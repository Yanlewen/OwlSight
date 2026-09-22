"""Factor-conditioned dual-input monitor with trajectory-shared evidence K/V."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .config import FACTOR_PATH_INTERVENTIONS, FactorMonitorArchitecture, load_architecture
from .encoding import (
    CRITERION_DIFFERENCE_DECISION_MODE,
    SINGLE_ENDPOINT_DECISION_MODE,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ModuleNotFoundError:  # Keep config/renderer commands lightweight.
    torch = None
    nn = None
    F = None


RELATION_TO_ID = {"Intrinsic": 0, "Extrinsic": 1}
ROUTE_TO_ID = {"cot_direct": 0, "context_grounded": 1}
DIRECT_ROUTE_ID = ROUTE_TO_ID["cot_direct"]
GROUNDED_ROUTE_ID = ROUTE_TO_ID["context_grounded"]


def _require_torch() -> None:
    if torch is None or nn is None or F is None:
        raise RuntimeError("FactorMonitor requires the factor-monitor training dependencies")


def _last_hidden(output: Any) -> Any:
    hidden_states = getattr(output, "hidden_states", None)
    if hidden_states:
        return hidden_states[-1]
    hidden = getattr(output, "last_hidden_state", None)
    if hidden is not None:
        return hidden
    if isinstance(output, Mapping):
        states = output.get("hidden_states")
        if states:
            return states[-1]
        if output.get("last_hidden_state") is not None:
            return output["last_hidden_state"]
    raise RuntimeError("backbone did not expose final hidden states")


def _hidden_size(backbone: Any) -> int:
    configs = [getattr(backbone, "config", None)]
    base = getattr(backbone, "base_model", None)
    if base is not None:
        configs.append(getattr(base, "config", None))
    for config in configs:
        if config is None:
            continue
        for candidate in (config, getattr(config, "text_config", None)):
            value = getattr(candidate, "hidden_size", None)
            if value is not None:
                return int(value)
    raise ValueError("cannot determine backbone hidden_size")


def _padded_token_chunk_ranges(
    sequence_lengths: list[int], budget: int | None
) -> list[tuple[int, int, int]]:
    """Group padded rows without exceeding a backbone-forward token budget."""

    ranges: list[tuple[int, int, int]] = []
    start = 0
    while start < len(sequence_lengths):
        end = start
        padded_tokens = 0
        while end < len(sequence_lengths):
            candidate_length = sequence_lengths[end]
            candidate_width = max(padded_tokens, candidate_length)
            candidate_volume = candidate_width * (end - start + 1)
            if budget is not None and candidate_volume > budget:
                if end == start:
                    raise RuntimeError(
                        "one joint sequence exceeds the configured backbone forward "
                        f"token budget: length={candidate_length}, budget={budget}"
                    )
                break
            padded_tokens = candidate_width
            end += 1
        ranges.append((start, end, padded_tokens))
        start = end
    return ranges


if nn is not None:

    class BackboneEncoder(nn.Module):
        """Use one causal LM for separate or joint monitor-input forwards."""

        def __init__(self, backbone: nn.Module, expected_hidden_size: int) -> None:
            super().__init__()
            actual = _hidden_size(backbone)
            if actual != expected_hidden_size:
                raise ValueError(
                    f"backbone hidden_size={actual} does not match architecture={expected_hidden_size}"
                )
            self.backbone = backbone
            self.hidden_size = actual
            self.factor_sequences_encoded = 0
            self.evidence_sequences_encoded = 0
            self.endpoint_sequences_encoded = 0
            self.joint_sequences_encoded = 0

        def _encode_hidden_states(
            self,
            input_ids: Any,
            attention_mask: Any,
            *,
            role: str,
            token_additions: Any | None = None,
        ) -> tuple[Any, ...]:
            if input_ids.ndim != 2 or attention_mask.shape != input_ids.shape:
                raise ValueError("backbone inputs must have matching [batch, tokens] shapes")
            if torch.any(attention_mask.sum(dim=-1) <= 0):
                raise ValueError("each backbone sequence must contain an attended token")
            backbone_inputs: dict[str, Any]
            if token_additions is None:
                backbone_inputs = {"input_ids": input_ids}
            else:
                if token_additions.shape != (*input_ids.shape, self.hidden_size):
                    raise ValueError(
                        "token additions must match [batch,tokens,hidden_size]"
                    )
                embedding_getter = getattr(self.backbone, "get_input_embeddings", None)
                if embedding_getter is None:
                    base_getter = getattr(self.backbone, "get_base_model", None)
                    if base_getter is None:
                        raise RuntimeError(
                            "backbone cannot expose token embeddings for role inputs"
                        )
                    embedding_getter = getattr(
                        base_getter(), "get_input_embeddings", None
                    )
                if embedding_getter is None:
                    raise RuntimeError(
                        "base model cannot expose token embeddings for role inputs"
                    )
                embedding_layer = embedding_getter()
                inputs_embeds = embedding_layer(input_ids)
                backbone_inputs = {
                    "inputs_embeds": inputs_embeds
                    + token_additions.to(dtype=inputs_embeds.dtype)
                }
            output = self.backbone(
                **backbone_inputs,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )
            hidden = _last_hidden(output)
            if hidden.shape[:2] != input_ids.shape or hidden.shape[-1] != self.hidden_size:
                raise RuntimeError("backbone returned an unexpected hidden-state shape")
            if role == "factor":
                self.factor_sequences_encoded += int(input_ids.shape[0])
            elif role == "evidence":
                self.evidence_sequences_encoded += int(input_ids.shape[0])
            elif role == "endpoint":
                self.endpoint_sequences_encoded += int(input_ids.shape[0])
            elif role == "joint":
                self.joint_sequences_encoded += int(input_ids.shape[0])
            else:
                raise ValueError(f"unknown backbone role: {role}")
            hidden_states = getattr(output, "hidden_states", None)
            if hidden_states is None and isinstance(output, Mapping):
                hidden_states = output.get("hidden_states")
            if not hidden_states:
                hidden_states = (hidden,)
            return tuple(hidden_states)

        def encode(
            self,
            input_ids: Any,
            attention_mask: Any,
            *,
            role: str,
            token_additions: Any | None = None,
        ) -> Any:
            return self._encode_hidden_states(
                input_ids,
                attention_mask,
                role=role,
                token_additions=token_additions,
            )[-1]

        def encode_selected_layers(
            self,
            input_ids: Any,
            attention_mask: Any,
            *,
            role: str,
            layers_below_final: tuple[int, ...],
            token_additions: Any | None = None,
        ) -> tuple[Any, dict[int, Any]]:
            if not layers_below_final or any(
                isinstance(offset, bool) or offset <= 0 for offset in layers_below_final
            ):
                raise ValueError("layers_below_final must contain positive integers")
            if len(set(layers_below_final)) != len(layers_below_final):
                raise ValueError("layers_below_final must not contain duplicates")
            states = self._encode_hidden_states(
                input_ids,
                attention_mask,
                role=role,
                token_additions=token_additions,
            )
            maximum = max(layers_below_final)
            if len(states) <= maximum:
                raise RuntimeError(
                    "backbone exposes too few hidden states for layer-mixed endpoint: "
                    f"states={len(states)}, requested_below_final={maximum}"
                )
            selected = {
                offset: states[-(offset + 1)] for offset in layers_below_final
            }
            for offset, hidden in selected.items():
                if hidden.shape[:2] != input_ids.shape or hidden.shape[-1] != self.hidden_size:
                    raise RuntimeError(
                        f"backbone layer {offset} below final has an unexpected shape"
                    )
            return states[-1], selected


    class FactorGuidedEvidenceReadout(nn.Module):
        """Pool endpoint evidence with a query derived only from the Factor span."""

        def __init__(
            self,
            hidden_size: int,
            *,
            readout_dim: int,
            interaction_hidden_dim: int,
        ) -> None:
            super().__init__()
            if readout_dim <= 0 or interaction_hidden_dim <= 0:
                raise ValueError("guided-readout dimensions must be positive")
            self.readout_dim = readout_dim
            self.query_projection = nn.Linear(hidden_size, readout_dim)
            self.key_projection = nn.Linear(hidden_size, readout_dim)
            self.interaction_projection = nn.Linear(
                readout_dim * 2, interaction_hidden_dim
            )
            self.output_projection = nn.Linear(interaction_hidden_dim, 1)
            nn.init.zeros_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        def forward(
            self,
            hidden: Any,
            *,
            factor_span_mask: Any,
            evidence_span_mask: Any,
        ) -> Any:
            if hidden.ndim != 3:
                raise ValueError("guided readout expects hidden states [N,T,H]")
            expected_mask_shape = hidden.shape[:2]
            if factor_span_mask.shape != expected_mask_shape:
                raise ValueError("Factor span mask must match hidden [N,T]")
            if evidence_span_mask.shape != expected_mask_shape:
                raise ValueError("Evidence span mask must match hidden [N,T]")
            factor_mask = factor_span_mask.bool()
            evidence_mask = evidence_span_mask.bool()
            factor_counts = factor_mask.long().sum(dim=-1)
            evidence_counts = evidence_mask.long().sum(dim=-1)
            if torch.any(factor_counts <= 0):
                raise ValueError("guided readout requires a non-empty Factor span")
            if torch.any(evidence_counts <= 0):
                raise ValueError(
                    "guided readout requires non-empty Context-or-CoT evidence"
                )

            hidden = hidden.to(dtype=self.query_projection.weight.dtype)

            factor_hidden = (
                hidden * factor_mask.unsqueeze(-1).to(dtype=hidden.dtype)
            ).sum(dim=1) / factor_counts.unsqueeze(-1).to(dtype=hidden.dtype)
            query = self.query_projection(factor_hidden)
            keys = self.key_projection(hidden)
            scores = torch.einsum("nd,ntd->nt", query, keys) / (
                self.readout_dim ** 0.5
            )
            scores = scores.float().masked_fill(~evidence_mask, float("-inf"))
            attention = torch.softmax(scores, dim=-1).to(dtype=keys.dtype)
            evidence = torch.einsum("nt,ntd->nd", attention, keys)
            interactions = torch.cat(
                (query * evidence, torch.abs(query - evidence)), dim=-1
            )
            interaction_hidden = F.gelu(
                self.interaction_projection(interactions)
            )
            return self.output_projection(interaction_hidden).squeeze(-1)


    class SignedCriterionPanel(nn.Module):
        """Signed applied-minus-not-applied cosine residual over canonical fields."""

        def __init__(self, hidden_size: int, *, projection_dim: int) -> None:
            super().__init__()
            if projection_dim <= 0:
                raise ValueError("signed criterion panel projection_dim must be positive")
            self.shared_projection = nn.Linear(hidden_size, projection_dim, dtype=torch.float32)
            self.endpoint_projection = nn.Linear(hidden_size, projection_dim, dtype=torch.float32)
            self.residual_scale = nn.Parameter(torch.zeros((), dtype=torch.float32))

        @staticmethod
        def _masked_mean(hidden: Any, mask: Any, *, name: str) -> Any:
            if mask.shape != hidden.shape[:2]:
                raise ValueError(f"{name} span mask must match hidden [N,T]")
            selected = mask.bool()
            counts = selected.long().sum(dim=-1)
            if torch.any(counts <= 0):
                raise ValueError(f"signed criterion panel requires a non-empty {name} span")
            weights = selected.unsqueeze(-1).to(dtype=hidden.dtype)
            return (hidden * weights).sum(dim=1) / counts.unsqueeze(-1).to(hidden.dtype)

        def forward(
            self,
            hidden: Any,
            endpoint_hidden: Any,
            *,
            factor_span_mask: Any,
            relation_span_mask: Any,
            applied_span_mask: Any,
            not_applied_span_mask: Any,
        ) -> Any:
            if hidden.ndim != 3 or endpoint_hidden.shape != (
                hidden.shape[0], hidden.shape[2]
            ):
                raise ValueError("signed criterion panel hidden shapes are invalid")
            endpoint_hidden = endpoint_hidden.float()
            pooled = {
                name: self._masked_mean(hidden, mask, name=name).float()
                for name, mask in (
                    ("Factor", factor_span_mask),
                    ("Relation", relation_span_mask),
                    ("Applied Criterion", applied_span_mask),
                    ("Not-Applied Criterion", not_applied_span_mask),
                )
            }
            shared = self.shared_projection
            common = shared(pooled["Factor"]) + shared(pooled["Relation"])
            applied = F.normalize(
                F.gelu(common + shared(pooled["Applied Criterion"])), dim=-1
            )
            not_applied = F.normalize(
                F.gelu(common + shared(pooled["Not-Applied Criterion"])), dim=-1
            )
            endpoint = F.normalize(self.endpoint_projection(endpoint_hidden), dim=-1)
            difference = (endpoint * applied).sum(dim=-1) - (
                endpoint * not_applied
            ).sum(dim=-1)
            return self.residual_scale * difference


    class LayerMixedEndpoint(nn.Module):
        """Zero-initialized logit residuals from fixed intermediate endpoint layers."""

        def __init__(self, hidden_size: int, *, layers_below_final: tuple[int, ...]) -> None:
            super().__init__()
            if not layers_below_final or len(set(layers_below_final)) != len(
                layers_below_final
            ):
                raise ValueError("layer-mixed endpoint requires unique selected layers")
            self.layers_below_final = tuple(layers_below_final)
            self.normalizations = nn.ModuleDict(
                {
                    str(offset): nn.LayerNorm(hidden_size, dtype=torch.float32)
                    for offset in self.layers_below_final
                }
            )
            self.heads = nn.ModuleDict(
                {
                    str(offset): nn.Linear(hidden_size, 1, dtype=torch.float32)
                    for offset in self.layers_below_final
                }
            )
            self.residual_scales = nn.ParameterDict(
                {
                    str(offset): nn.Parameter(torch.zeros((), dtype=torch.float32))
                    for offset in self.layers_below_final
                }
            )

        def forward(self, endpoint_hidden_by_layer: Mapping[int, Any]) -> Any:
            if set(endpoint_hidden_by_layer) != set(self.layers_below_final):
                raise ValueError("layer-mixed endpoint hidden-state selection differs")
            residual = None
            for offset in self.layers_below_final:
                key = str(offset)
                hidden = endpoint_hidden_by_layer[offset].float()
                value = self.heads[key](self.normalizations[key](hidden)).squeeze(-1)
                value = self.residual_scales[key] * value
                residual = value if residual is None else residual + value
            return residual


    class FeedForward(nn.Module):
        def __init__(self, dim: int, hidden_dim: int, dropout: float) -> None:
            super().__init__()
            self.layers = nn.Sequential(
                nn.Linear(dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, dim),
                nn.Dropout(dropout),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value)


    class FactorEncoderLayer(nn.Module):
        def __init__(self, config: FactorMonitorArchitecture) -> None:
            super().__init__()
            dim = config.reader_dim
            heads = config.num_attention_heads
            self.heads = heads
            self.self_attention = nn.MultiheadAttention(
                dim, heads, dropout=config.dropout, batch_first=True
            )
            self.cross_attention = nn.MultiheadAttention(
                dim, heads, dropout=config.dropout, batch_first=True
            )
            self.ffn = FeedForward(dim, config.ffn_dim, config.dropout)
            self.self_norm = nn.LayerNorm(dim)
            self.cross_norm = nn.LayerNorm(dim)
            self.ffn_norm = nn.LayerNorm(dim)

        def forward(self, queries: Any, factor_hidden: Any, field_masks: Any) -> Any:
            residual = queries
            attended, _ = self.self_attention(queries, queries, queries, need_weights=False)
            queries = self.self_norm(residual + attended)
            allowed = field_masks.bool()
            if torch.any(allowed.sum(dim=-1) <= 0):
                raise ValueError("every Factor query must have at least one field token")
            batch, num_queries, tokens = allowed.shape
            attention_mask = (~allowed).unsqueeze(1).expand(batch, self.heads, num_queries, tokens)
            attention_mask = attention_mask.reshape(batch * self.heads, num_queries, tokens)
            residual = queries
            attended, _ = self.cross_attention(
                queries,
                factor_hidden,
                factor_hidden,
                attn_mask=attention_mask,
                need_weights=False,
            )
            queries = self.cross_norm(residual + attended)
            return self.ffn_norm(queries + self.ffn(queries))


    class FactorEncoder(nn.Module):
        def __init__(self, config: FactorMonitorArchitecture) -> None:
            super().__init__()
            self.num_queries = config.num_queries
            self.queries = nn.Parameter(torch.empty(config.num_queries, config.reader_dim))
            nn.init.normal_(self.queries, mean=0.0, std=config.reader_dim**-0.5)
            self.layers = nn.ModuleList(
                FactorEncoderLayer(config) for _ in range(config.num_layers)
            )

        def forward(self, factor_hidden: Any, field_masks: Any) -> Any:
            if factor_hidden.ndim != 3 or field_masks.ndim != 3:
                raise ValueError("FactorEncoder expects [records,tokens,dim] and [records,3,tokens]")
            if field_masks.shape[0] != factor_hidden.shape[0] or field_masks.shape[1] != self.num_queries:
                raise ValueError("Factor field-mask shape does not match hidden states")
            queries = self.queries.unsqueeze(0).expand(factor_hidden.shape[0], -1, -1)
            for layer in self.layers:
                queries = layer(queries, factor_hidden, field_masks)
            return queries


    class FactorFieldMeanPool(nn.Module):
        """Deterministically pool the three canonical Factor text fields."""

        def forward(self, factor_hidden: Any, field_masks: Any) -> Any:
            if factor_hidden.ndim != 3 or field_masks.ndim != 3:
                raise ValueError(
                    "FactorFieldMeanPool expects [records,tokens,dim] and "
                    "[records,3,tokens]"
                )
            if (
                field_masks.shape[0] != factor_hidden.shape[0]
                or field_masks.shape[1] != 3
                or field_masks.shape[2] != factor_hidden.shape[1]
            ):
                raise ValueError("Factor field-mask shape does not match hidden states")
            weights = field_masks.bool().to(dtype=factor_hidden.dtype)
            counts = weights.sum(dim=-1, keepdim=True)
            if torch.any(counts <= 0):
                raise ValueError("every Factor field must contain at least one token")
            pooled = torch.einsum("rft,rtd->rfd", weights, factor_hidden)
            return pooled / counts


    class GroupedCrossAttention(nn.Module):
        """Cross-attention over shared or per-query Evidence K/V tensors."""

        def __init__(self, dim: int, heads: int, head_dim: int, dropout: float) -> None:
            super().__init__()
            if dim != heads * head_dim:
                raise ValueError("attention dim must equal heads * head_dim")
            self.heads = heads
            self.head_dim = head_dim
            self.scale = head_dim**-0.5
            self.query = nn.Linear(dim, dim)
            self.key = nn.Linear(dim, dim)
            self.value = nn.Linear(dim, dim)
            self.output = nn.Linear(dim, dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, queries: Any, evidence: Any, evidence_mask: Any) -> Any:
            if queries.ndim != 4 or evidence.ndim not in {3, 4}:
                raise ValueError(
                    "grouped attention expects queries [B,Q,T,D] and Evidence "
                    "[B,N,D] or query-conditioned Evidence [B,Q,N,D]"
                )
            batch, records, query_count, dim = queries.shape
            if evidence.ndim == 3:
                if evidence_mask.ndim != 2 or evidence.shape[:2] != evidence_mask.shape:
                    raise ValueError("shared Evidence batch/mask shapes do not match")
            elif (
                evidence_mask.ndim != 3
                or evidence.shape[:3] != evidence_mask.shape
                or evidence.shape[:2] != (batch, records)
            ):
                raise ValueError(
                    "query-conditioned Evidence batch/record/mask shapes do not match"
                )
            if evidence.shape[0] != batch:
                raise ValueError("Evidence batch does not match grouped queries")
            if torch.any(evidence_mask.bool().sum(dim=-1) <= 0):
                raise ValueError("every Evidence row must expose at least one token")
            q = self.query(queries).reshape(
                batch, records, query_count, self.heads, self.head_dim
            ).permute(0, 1, 3, 2, 4)
            if evidence.ndim == 3:
                k = self.key(evidence).reshape(
                    batch, evidence.shape[1], self.heads, self.head_dim
                ).permute(0, 2, 1, 3)
                v = self.value(evidence).reshape(
                    batch, evidence.shape[1], self.heads, self.head_dim
                ).permute(0, 2, 1, 3)
                scores = torch.einsum("bqhtd,bhnd->bqhtn", q, k) * self.scale
                mask = evidence_mask.bool()[:, None, None, None, :]
                probabilities = self.dropout(
                    torch.softmax(
                        scores.masked_fill(~mask, torch.finfo(scores.dtype).min),
                        dim=-1,
                    )
                )
                attended = torch.einsum("bqhtn,bhnd->bqhtd", probabilities, v)
            else:
                k = self.key(evidence).reshape(
                    batch,
                    records,
                    evidence.shape[2],
                    self.heads,
                    self.head_dim,
                ).permute(0, 1, 3, 2, 4)
                v = self.value(evidence).reshape(
                    batch,
                    records,
                    evidence.shape[2],
                    self.heads,
                    self.head_dim,
                ).permute(0, 1, 3, 2, 4)
                scores = torch.einsum("bqhtd,bqhnd->bqhtn", q, k) * self.scale
                mask = evidence_mask.bool()[:, :, None, None, :]
                probabilities = self.dropout(
                    torch.softmax(
                        scores.masked_fill(~mask, torch.finfo(scores.dtype).min),
                        dim=-1,
                    )
                )
                attended = torch.einsum("bqhtn,bqhnd->bqhtd", probabilities, v)
            attended = attended.permute(0, 1, 3, 2, 4).reshape(
                batch, records, query_count, dim
            )
            return self.output(attended)


    class EvidenceLocatorLayer(nn.Module):
        def __init__(self, config: FactorMonitorArchitecture, *, query_self_attention: bool) -> None:
            super().__init__()
            dim = config.reader_dim
            self.query_self_attention = query_self_attention
            self.self_attention = (
                nn.MultiheadAttention(
                    dim,
                    config.num_attention_heads,
                    dropout=config.dropout,
                    batch_first=True,
                )
                if query_self_attention
                else None
            )
            self.cross_attention = GroupedCrossAttention(
                dim,
                config.num_attention_heads,
                config.head_dim,
                config.dropout,
            )
            self.ffn = FeedForward(dim, config.ffn_dim, config.dropout)
            self.self_norm = nn.LayerNorm(dim) if query_self_attention else None
            self.cross_norm = nn.LayerNorm(dim)
            self.ffn_norm = nn.LayerNorm(dim)

        def forward(
            self,
            queries: Any,
            evidence: Any,
            evidence_mask: Any,
            query_mask: Any,
            *,
            return_cross_attention_readout: bool = False,
        ) -> Any:
            if self.self_attention is not None:
                batch, records, query_count, dim = queries.shape
                flat = queries.reshape(batch * records, query_count, dim)
                attended, _ = self.self_attention(flat, flat, flat, need_weights=False)
                queries = self.self_norm(flat + attended).reshape(batch, records, query_count, dim)
            safe_evidence_mask = evidence_mask
            if evidence.ndim == 4:
                empty = evidence_mask.bool().sum(dim=-1) <= 0
                if torch.any(empty & query_mask.bool()):
                    raise ValueError("a valid query has no visible Evidence tokens")
                if torch.any(empty):
                    safe_evidence_mask = evidence_mask.bool().clone()
                    empty_indices = empty.nonzero(as_tuple=True)
                    safe_evidence_mask[empty_indices[0], empty_indices[1], 0] = True
            attended = self.cross_attention(queries, evidence, safe_evidence_mask)
            evidence_readout = (
                self.cross_norm(attended)
                if return_cross_attention_readout
                else None
            )
            queries = self.cross_norm(queries + attended)
            queries = self.ffn_norm(queries + self.ffn(queries))
            valid = query_mask[:, :, None, None].to(queries.dtype)
            queries = queries * valid
            if evidence_readout is not None:
                return queries, evidence_readout * valid
            return queries


    class EvidenceLocator(nn.Module):
        def __init__(self, config: FactorMonitorArchitecture, *, query_self_attention: bool) -> None:
            super().__init__()
            self.layers = nn.ModuleList(
                EvidenceLocatorLayer(config, query_self_attention=query_self_attention)
                for _ in range(config.num_layers)
            )

        def forward(
            self,
            queries: Any,
            evidence: Any,
            evidence_mask: Any,
            query_mask: Any,
            *,
            return_cross_attention_readout: bool = False,
        ) -> Any:
            evidence_readout = None
            final_index = len(self.layers) - 1
            for index, layer in enumerate(self.layers):
                if return_cross_attention_readout and index == final_index:
                    queries, evidence_readout = layer(
                        queries,
                        evidence,
                        evidence_mask,
                        query_mask,
                        return_cross_attention_readout=True,
                    )
                else:
                    queries = layer(queries, evidence, evidence_mask, query_mask)
            if return_cross_attention_readout:
                if evidence_readout is None:
                    raise RuntimeError("CoT Evidence readout was not produced")
                return queries, evidence_readout
            return queries


    class ContextLatentAggregator(nn.Module):
        def forward(self, latents: Any) -> Any:
            if latents.ndim != 4 or latents.shape[2] != 1:
                raise ValueError("Phase 1 ContextLatentAggregator requires [B,Q,1,D]")
            return latents[:, :, 0, :]


    class GroundedQueryFusion(nn.Module):
        def __init__(self, config: FactorMonitorArchitecture) -> None:
            super().__init__()
            dim = config.reader_dim
            self.mlp = nn.Sequential(
                nn.Linear(2 * dim, config.fusion_hidden_dim),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.fusion_hidden_dim, dim),
                nn.Dropout(config.dropout),
            )

        def forward(self, queries: Any, context: Any) -> Any:
            expanded = context.unsqueeze(2).expand(-1, -1, queries.shape[2], -1)
            return queries + self.mlp(torch.cat((queries, expanded), dim=-1))


    class Matcher(nn.Module):
        def __init__(self, input_dim: int, hidden_dims: tuple[int, int], dropout: float) -> None:
            super().__init__()
            first, second = hidden_dims
            self.layers = nn.Sequential(
                nn.Linear(input_dim, first),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(first, second),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(second, 1),
            )

        def forward(self, value: Any) -> Any:
            return self.layers(value).squeeze(-1)


    class FactorMonitor(nn.Module):
        """Pure-logit model; labels and hierarchy metadata stay in the Trainer."""

        CUSTOM_PREFIXES = (
            "factor_projection.",
            "evidence_projection.",
            "factor_encoder.",
            "context_locator.",
            "cot_locator.",
            "grounded_query_fusion.",
            "relation_embedding.",
            "direct_matcher.",
            "grounded_matcher.",
        )

        def __init__(
            self,
            backbone: nn.Module,
            architecture: FactorMonitorArchitecture,
            *,
            joint_encoding: bool = False,
            max_padded_tokens_per_backbone_forward: int | None = None,
        ) -> None:
            super().__init__()
            if (
                max_padded_tokens_per_backbone_forward is not None
                and max_padded_tokens_per_backbone_forward <= 0
            ):
                raise ValueError(
                    "max_padded_tokens_per_backbone_forward must be positive"
                )
            if not joint_encoding and max_padded_tokens_per_backbone_forward is not None:
                raise ValueError(
                    "a backbone forward token budget requires joint FactorMonitor encoding"
                )
            self.architecture = architecture
            self.joint_encoding = bool(joint_encoding)
            self.max_padded_tokens_per_backbone_forward = (
                max_padded_tokens_per_backbone_forward
            )
            self.config = getattr(backbone, "config", None)
            self.main_input_name = (
                "joint_input_ids" if self.joint_encoding else "evidence_input_ids"
            )
            self.accepts_loss_kwargs = False
            self.supports_gradient_checkpointing = bool(
                getattr(backbone, "supports_gradient_checkpointing", False)
            )
            self.backbone_encoder = BackboneEncoder(backbone, architecture.hidden_size)
            self.factor_projection = nn.Linear(architecture.hidden_size, architecture.reader_dim)
            self.evidence_projection = nn.Linear(architecture.hidden_size, architecture.reader_dim)
            if architecture.factor_encoder_mode == "learned_queries":
                self.factor_encoder = FactorEncoder(architecture)
            elif architecture.factor_encoder_mode == "field_mean_pool":
                self.factor_encoder = FactorFieldMeanPool()
            else:  # Defensive for manually constructed architectures.
                raise ValueError(
                    "unknown factor_encoder_mode: "
                    f"{architecture.factor_encoder_mode!r}"
                )
            self.context_locator = EvidenceLocator(architecture, query_self_attention=False)
            self.context_latent_aggregator = ContextLatentAggregator()
            self.cot_locator = EvidenceLocator(
                architecture,
                query_self_attention=architecture.cot_query_self_attention,
            )
            self.grounded_query_fusion = GroundedQueryFusion(architecture)
            self.relation_embedding = nn.Embedding(len(architecture.relations), architecture.relation_dim)
            direct_matcher_input_dim = (
                architecture.reader_dim + architecture.relation_dim
                if architecture.direct_matcher_mode == "criterion_margin"
                else (
                    (6 if architecture.direct_matcher_include_raw_factor else 3)
                    * architecture.reader_dim
                    + architecture.relation_dim
                )
            )
            self.direct_matcher = Matcher(
                direct_matcher_input_dim,
                architecture.direct_hidden_dims,
                architecture.dropout,
            )
            self.grounded_matcher = Matcher(
                7 * architecture.reader_dim + architecture.relation_dim,
                architecture.grounded_hidden_dims,
                architecture.dropout,
            )

        @property
        def backbone(self) -> nn.Module:
            return self.backbone_encoder.backbone

        @property
        def is_gradient_checkpointing(self) -> bool:
            return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

        def gradient_checkpointing_enable(
            self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
        ) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method(
                gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {}
            )

        def gradient_checkpointing_disable(self) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_disable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method()

        def _validate_inputs(self, values: Mapping[str, Any]) -> tuple[int, int]:
            evidence_ids = values["evidence_input_ids"]
            factor_ids = values["factor_input_ids"]
            query_mask = values["query_mask"]
            if evidence_ids.ndim != 2 or factor_ids.ndim != 3 or query_mask.ndim != 2:
                raise ValueError("FactorMonitor expects evidence [B,N], Factor [B,Q,F], mask [B,Q]")
            batch, records = query_mask.shape
            if evidence_ids.shape[0] != batch or factor_ids.shape[:2] != (batch, records):
                raise ValueError("FactorMonitor B/Q dimensions do not match")
            if torch.any(query_mask.bool().sum(dim=-1) <= 0):
                raise ValueError("each trajectory must contain at least one valid query record")
            for name in (
                "factor_attention_mask",
                "factor_span_mask",
                "applied_span_mask",
                "not_applied_span_mask",
            ):
                if values[name].shape != factor_ids.shape:
                    raise ValueError(f"{name} must match factor_input_ids")
            for name in ("evidence_attention_mask", "context_span_mask", "cot_span_mask"):
                if values[name].shape != evidence_ids.shape:
                    raise ValueError(f"{name} must match evidence_input_ids")
            for name in ("relation", "evidence_route"):
                if values[name].shape != query_mask.shape:
                    raise ValueError(f"{name} must match query_mask")
            return batch, records

        def _validate_joint_inputs(
            self, values: Mapping[str, Any]
        ) -> tuple[int, int]:
            joint_ids = values["joint_input_ids"]
            query_mask = values["query_mask"]
            if joint_ids.ndim != 3 or query_mask.ndim != 2:
                raise ValueError(
                    "joint FactorMonitor expects input [B,Q,T] and mask [B,Q]"
                )
            batch, records = query_mask.shape
            if joint_ids.shape[:2] != (batch, records):
                raise ValueError("joint input B/Q dimensions do not match query_mask")
            if torch.any(query_mask.bool().sum(dim=-1) <= 0):
                raise ValueError("each trajectory must contain at least one valid query record")
            for name in (
                "joint_attention_mask",
                "factor_span_mask",
                "applied_span_mask",
                "not_applied_span_mask",
                "context_span_mask",
                "cot_span_mask",
            ):
                if values[name].shape != joint_ids.shape:
                    raise ValueError(f"{name} must match joint_input_ids")
            for name in ("relation", "evidence_route"):
                if values[name].shape != query_mask.shape:
                    raise ValueError(f"{name} must match query_mask")
            return batch, records

        def _encode_factors(
            self,
            factor_input_ids: Any,
            factor_attention_mask: Any,
            field_masks: Any,
            query_mask: Any,
        ) -> Any:
            batch, records, tokens = factor_input_ids.shape
            valid = query_mask.bool().reshape(-1)
            flat_ids = factor_input_ids.reshape(batch * records, tokens)[valid]
            flat_attention = factor_attention_mask.reshape(batch * records, tokens)[valid]
            flat_fields = field_masks.reshape(batch * records, 3, tokens)[valid]
            hidden = self.backbone_encoder.encode(flat_ids, flat_attention, role="factor")
            hidden = hidden.to(dtype=self.factor_projection.weight.dtype)
            queries = self.factor_encoder(self.factor_projection(hidden), flat_fields)
            output = torch.zeros(
                batch * records,
                3,
                self.architecture.reader_dim,
                dtype=queries.dtype,
                device=queries.device,
            )
            output[valid] = queries
            return output.reshape(batch, records, 3, self.architecture.reader_dim)

        def _encode_joint(
            self,
            joint_input_ids: Any,
            joint_attention_mask: Any,
            field_masks: Any,
            query_mask: Any,
        ) -> tuple[Any, Any]:
            """Encode one Factor+Evidence sequence per query in the same backbone pass."""

            batch, records, tokens = joint_input_ids.shape
            valid = query_mask.bool().reshape(-1)
            flat_ids = joint_input_ids.reshape(batch * records, tokens)[valid]
            flat_attention = joint_attention_mask.reshape(batch * records, tokens)[valid]
            flat_fields = field_masks.reshape(batch * records, 3, tokens)[valid]
            lengths = [
                int(value)
                for value in flat_attention.long().sum(dim=-1).detach().cpu().tolist()
            ]
            ranges = _padded_token_chunk_ranges(
                lengths, self.max_padded_tokens_per_backbone_forward
            )
            valid_queries: list[Any] = []
            valid_evidence = None
            for start, end, chunk_tokens in ranges:
                hidden = self.backbone_encoder.encode(
                    flat_ids[start:end, :chunk_tokens],
                    flat_attention[start:end, :chunk_tokens],
                    role="joint",
                )
                factor_hidden = self.factor_projection(
                    hidden.to(dtype=self.factor_projection.weight.dtype)
                )
                valid_queries.append(
                    self.factor_encoder(
                        factor_hidden,
                        flat_fields[start:end, :, :chunk_tokens],
                    )
                )
                projected_evidence = self.evidence_projection(
                    hidden.to(dtype=self.evidence_projection.weight.dtype)
                )
                if valid_evidence is None:
                    valid_evidence = projected_evidence.new_zeros(
                        flat_ids.shape[0],
                        tokens,
                        self.architecture.reader_dim,
                    )
                valid_evidence[start:end, :chunk_tokens] = projected_evidence
            queries = torch.cat(valid_queries, dim=0)
            if valid_evidence is None:
                raise RuntimeError("joint encoding received no valid query records")
            factor_output = torch.zeros(
                batch * records,
                3,
                self.architecture.reader_dim,
                dtype=queries.dtype,
                device=queries.device,
            )
            factor_output[valid] = queries
            evidence_output = torch.zeros(
                batch * records,
                tokens,
                self.architecture.reader_dim,
                dtype=valid_evidence.dtype,
                device=valid_evidence.device,
            )
            evidence_output[valid] = valid_evidence
            return (
                factor_output.reshape(
                    batch, records, 3, self.architecture.reader_dim
                ),
                evidence_output.reshape(
                    batch, records, tokens, self.architecture.reader_dim
                ),
            )

        def _locate_direct_evidence(
            self,
            factor_queries: Any,
            evidence_hidden: Any,
            evidence_mask: Any,
            query_mask: Any,
        ) -> Any:
            if self.architecture.direct_evidence_mode == "locator_state":
                return self.cot_locator(
                    factor_queries,
                    evidence_hidden,
                    evidence_mask,
                    query_mask,
                )
            if self.architecture.direct_evidence_mode == "cross_attention_readout":
                _, evidence_readout = self.cot_locator(
                    factor_queries,
                    evidence_hidden,
                    evidence_mask,
                    query_mask,
                    return_cross_attention_readout=True,
                )
                return evidence_readout
            raise RuntimeError(
                "unknown direct_evidence_mode: "
                f"{self.architecture.direct_evidence_mode!r}"
            )

        @staticmethod
        def _validate_direct_matcher_inputs(
            located_evidence: Any,
            factor_queries: Any,
            relation_hidden: Any,
        ) -> None:
            if located_evidence.shape != factor_queries.shape:
                raise ValueError(
                    "direct matcher located-Evidence and Factor-query shapes differ"
                )
            if located_evidence.shape[-2] != 3:
                raise ValueError("direct matcher requires three canonical Factor states")
            if relation_hidden.shape[:-1] != located_evidence.shape[:-2]:
                raise ValueError("direct matcher Relation leading dimensions differ")

        def _direct_matcher_features(
            self,
            located_evidence: Any,
            factor_queries: Any,
            relation_hidden: Any,
        ) -> Any:
            if self.architecture.direct_matcher_mode != "concat_binary":
                raise RuntimeError(
                    "concatenated direct-matcher features require concat_binary mode"
                )
            self._validate_direct_matcher_inputs(
                located_evidence, factor_queries, relation_hidden
            )
            parts = [located_evidence.flatten(start_dim=located_evidence.ndim - 2)]
            if self.architecture.direct_matcher_include_raw_factor:
                parts.append(factor_queries.flatten(start_dim=factor_queries.ndim - 2))
            parts.append(relation_hidden)
            return torch.cat(parts, dim=-1)

        def _direct_matcher_logits(
            self,
            located_evidence: Any,
            factor_queries: Any,
            relation_hidden: Any,
        ) -> Any:
            self._validate_direct_matcher_inputs(
                located_evidence, factor_queries, relation_hidden
            )
            if self.architecture.direct_matcher_mode == "concat_binary":
                return self.direct_matcher(
                    self._direct_matcher_features(
                        located_evidence,
                        factor_queries,
                        relation_hidden,
                    )
                )
            if self.architecture.direct_matcher_mode == "criterion_margin":
                applied_features = torch.cat(
                    (located_evidence[..., 1, :], relation_hidden), dim=-1
                )
                not_applied_features = torch.cat(
                    (located_evidence[..., 2, :], relation_hidden), dim=-1
                )
                return self.direct_matcher(applied_features) - self.direct_matcher(
                    not_applied_features
                )
            raise RuntimeError(
                "unknown direct_matcher_mode: "
                f"{self.architecture.direct_matcher_mode!r}"
            )

        def direct_factor_path_logits(
            self,
            *,
            evidence_input_ids: Any,
            evidence_attention_mask: Any,
            context_span_mask: Any,
            cot_span_mask: Any,
            factor_input_ids: Any,
            factor_attention_mask: Any,
            factor_span_mask: Any,
            applied_span_mask: Any,
            not_applied_span_mask: Any,
            query_mask: Any,
            relation: Any,
            evidence_route: Any,
            matched_factor_indices: Any,
            shuffled_factor_indices: Any,
            source_query_mask: Any,
            interventions: tuple[str, ...],
        ) -> dict[str, Any]:
            """Attribute direct-route logits to Locator and direct-matcher paths."""

            if self.joint_encoding:
                raise RuntimeError(
                    "Factor path decomposition is undefined for joint backbone encoding"
                )
            values = locals()
            batch, records = self._validate_inputs(values)
            if tuple(interventions) != tuple(dict.fromkeys(interventions)):
                raise ValueError("factor path interventions must be unique")
            unknown = set(interventions) - set(FACTOR_PATH_INTERVENTIONS)
            if unknown:
                raise ValueError(f"unknown factor path interventions: {sorted(unknown)}")
            if not interventions or interventions[0] != "matched":
                raise ValueError("factor path interventions must begin with matched")
            if matched_factor_indices.shape != shuffled_factor_indices.shape:
                raise ValueError("matched/shuffled Factor index shapes differ")
            if source_query_mask.shape != matched_factor_indices.shape:
                raise ValueError("source query mask does not match Factor indices")
            if matched_factor_indices.shape[0] != batch:
                raise ValueError("Factor path indices do not match the evidence batch")
            query_mask = query_mask.bool()
            source_query_mask = source_query_mask.bool()
            if torch.any(evidence_route[query_mask] != DIRECT_ROUTE_ID):
                raise ValueError("factor path attribution currently supports cot_direct only")
            for name, indices in (
                ("matched", matched_factor_indices),
                ("shuffled", shuffled_factor_indices),
            ):
                selected = indices[source_query_mask]
                if torch.any((selected < 0) | (selected >= records)):
                    raise ValueError(f"{name} Factor indices are out of range")

            safe_matched = matched_factor_indices.clamp(min=0, max=records - 1)
            safe_shuffled = shuffled_factor_indices.clamp(min=0, max=records - 1)
            batch_indices = torch.arange(batch, device=query_mask.device).unsqueeze(1)
            if torch.any(
                ~query_mask[batch_indices, safe_matched][source_query_mask]
            ) or torch.any(
                ~query_mask[batch_indices, safe_shuffled][source_query_mask]
            ):
                raise ValueError("factor path indices must select valid query records")

            field_masks = torch.stack(
                (factor_span_mask, applied_span_mask, not_applied_span_mask), dim=2
            ).bool()
            if torch.any(field_masks & ~factor_attention_mask.bool().unsqueeze(2)):
                raise ValueError("Factor field masks may not include padding")
            factor_queries = self._encode_factors(
                factor_input_ids,
                factor_attention_mask,
                field_masks,
                query_mask,
            )
            matched_queries = factor_queries[batch_indices, safe_matched]
            shuffled_queries = factor_queries[batch_indices, safe_shuffled]
            matched_relations = relation[batch_indices, safe_matched]
            shuffled_relations = relation[batch_indices, safe_shuffled]
            if torch.any(
                matched_relations[source_query_mask]
                != shuffled_relations[source_query_mask]
            ):
                raise ValueError("matched/shuffled Factors must preserve Relation")

            evidence_hidden = self.backbone_encoder.encode(
                evidence_input_ids, evidence_attention_mask, role="evidence"
            )
            evidence_hidden = self.evidence_projection(
                evidence_hidden.to(dtype=self.evidence_projection.weight.dtype)
            )
            cot_mask = cot_span_mask.bool() & evidence_attention_mask.bool()
            all_located = self._locate_direct_evidence(
                factor_queries,
                evidence_hidden,
                cot_mask,
                query_mask,
            )
            matched_located = all_located[batch_indices, safe_matched]
            shuffled_located = all_located[batch_indices, safe_shuffled]
            relation_hidden = self.relation_embedding(
                matched_relations.clamp(
                    min=0, max=len(self.architecture.relations) - 1
                )
            )

            components = {
                "matched": (matched_located, matched_queries),
                "full_shuffled": (shuffled_located, shuffled_queries),
                "locator_only_shuffled": (shuffled_located, matched_queries),
                "matcher_only_shuffled": (matched_located, shuffled_queries),
                "zero_matcher_factor": (
                    matched_located,
                    torch.zeros_like(matched_queries),
                ),
                "zero_located_evidence": (
                    torch.zeros_like(matched_located),
                    matched_queries,
                ),
            }
            logits: dict[str, Any] = {}
            for name in interventions:
                located, matcher_factor = components[name]
                values = self._direct_matcher_logits(
                    located,
                    matcher_factor,
                    relation_hidden,
                )
                logits[name] = values.masked_fill(~source_query_mask, 0.0)
            return logits

        def forward(
            self,
            *,
            evidence_input_ids: Any = None,
            evidence_attention_mask: Any = None,
            context_span_mask: Any = None,
            cot_span_mask: Any = None,
            factor_input_ids: Any = None,
            factor_attention_mask: Any = None,
            factor_span_mask: Any = None,
            applied_span_mask: Any = None,
            not_applied_span_mask: Any = None,
            query_mask: Any = None,
            relation: Any = None,
            evidence_route: Any = None,
            joint_input_ids: Any = None,
            joint_attention_mask: Any = None,
        ) -> Any:
            values = locals()
            if self.joint_encoding:
                self._validate_joint_inputs(values)
                if evidence_input_ids is not None or factor_input_ids is not None:
                    raise ValueError(
                        "joint FactorMonitor encoding does not accept separate inputs"
                    )
                evidence_attention_mask = joint_attention_mask
                factor_attention_mask = joint_attention_mask
            else:
                self._validate_inputs(values)
                if joint_input_ids is not None or joint_attention_mask is not None:
                    raise ValueError(
                        "separate FactorMonitor encoding does not accept joint inputs"
                    )
            query_mask = query_mask.bool()
            valid_relations = relation[query_mask]
            valid_routes = evidence_route[query_mask]
            if torch.any((valid_relations < 0) | (valid_relations >= len(self.architecture.relations))):
                raise ValueError("relation contains an invalid ID")
            if torch.any((valid_routes < 0) | (valid_routes >= len(self.architecture.routes))):
                raise ValueError("evidence_route contains an invalid ID")
            if torch.any(context_span_mask.bool() & cot_span_mask.bool()):
                raise ValueError("Context and CoT span masks must not overlap")
            if torch.any((context_span_mask.bool() | cot_span_mask.bool()) & ~evidence_attention_mask.bool()):
                raise ValueError("evidence span masks may not include padding")

            field_masks = torch.stack(
                (factor_span_mask, applied_span_mask, not_applied_span_mask), dim=2
            ).bool()
            if torch.any(field_masks & ~factor_attention_mask.bool().unsqueeze(2)):
                raise ValueError("Factor field masks may not include padding")
            if self.joint_encoding:
                factor_queries, evidence_hidden = self._encode_joint(
                    joint_input_ids,
                    joint_attention_mask,
                    field_masks,
                    query_mask,
                )
            else:
                factor_queries = self._encode_factors(
                    factor_input_ids,
                    factor_attention_mask,
                    field_masks,
                    query_mask,
                )
                evidence_hidden = self.backbone_encoder.encode(
                    evidence_input_ids, evidence_attention_mask, role="evidence"
                )
                evidence_hidden = evidence_hidden.to(
                    dtype=self.evidence_projection.weight.dtype
                )
                evidence_hidden = self.evidence_projection(evidence_hidden)
            direct_mask = query_mask & (evidence_route == DIRECT_ROUTE_ID)
            grounded_mask = query_mask & (evidence_route == GROUNDED_ROUTE_ID)
            shape = (*factor_queries.shape[:2], self.architecture.reader_dim)
            direct_evidence = factor_queries.new_zeros(
                *factor_queries.shape[:3], self.architecture.reader_dim
            )
            context_pooled = factor_queries.new_zeros(*shape)
            grounded_evidence = factor_queries.new_zeros(
                *factor_queries.shape[:3], self.architecture.reader_dim
            )
            direct_trajectories = direct_mask.any(dim=1).nonzero(as_tuple=False).flatten()
            if direct_trajectories.numel() > 0:
                selected_direct = self._locate_direct_evidence(
                    factor_queries.index_select(0, direct_trajectories),
                    evidence_hidden.index_select(0, direct_trajectories),
                    (cot_span_mask.bool() & evidence_attention_mask.bool()).index_select(
                        0, direct_trajectories
                    ),
                    direct_mask.index_select(0, direct_trajectories),
                )
                direct_evidence = direct_evidence.index_copy(
                    0, direct_trajectories, selected_direct
                )
            grounded_trajectories = grounded_mask.any(dim=1).nonzero(
                as_tuple=False
            ).flatten()
            if grounded_trajectories.numel() > 0:
                selected_factor = factor_queries.index_select(0, grounded_trajectories)
                selected_mask = grounded_mask.index_select(0, grounded_trajectories)
                context_latents = self.context_locator(
                    selected_factor[:, :, :1, :],
                    evidence_hidden.index_select(0, grounded_trajectories),
                    (
                        context_span_mask.bool() & evidence_attention_mask.bool()
                    ).index_select(0, grounded_trajectories),
                    selected_mask,
                )
                selected_context = self.context_latent_aggregator(context_latents)
                selected_grounded_queries = self.grounded_query_fusion(
                    selected_factor, selected_context
                )
                selected_grounded = self.cot_locator(
                    selected_grounded_queries,
                    evidence_hidden.index_select(0, grounded_trajectories),
                    (cot_span_mask.bool() & evidence_attention_mask.bool()).index_select(
                        0, grounded_trajectories
                    ),
                    selected_mask,
                )
                context_pooled = context_pooled.index_copy(
                    0, grounded_trajectories, selected_context
                )
                grounded_evidence = grounded_evidence.index_copy(
                    0, grounded_trajectories, selected_grounded
                )
            relation_hidden = self.relation_embedding(relation.clamp(min=0, max=len(self.architecture.relations) - 1))
            grounded_features = torch.cat(
                (
                    context_pooled[grounded_mask],
                    grounded_evidence[grounded_mask].flatten(start_dim=1),
                    factor_queries[grounded_mask].flatten(start_dim=1),
                    relation_hidden[grounded_mask],
                ),
                dim=-1,
            )
            direct_logits = None
            if torch.any(direct_mask):
                direct_logits = self._direct_matcher_logits(
                    direct_evidence[direct_mask],
                    factor_queries[direct_mask],
                    relation_hidden[direct_mask],
                )
            grounded_logits = None
            if grounded_features.shape[0] > 0:
                grounded_logits = self.grounded_matcher(grounded_features)
            branch_logits = [
                values for values in (direct_logits, grounded_logits) if values is not None
            ]
            if not branch_logits:
                raise RuntimeError("FactorMonitor batch contains no routed queries")
            logits = factor_queries.new_zeros(
                query_mask.shape, dtype=branch_logits[0].dtype
            )
            if direct_logits is not None:
                logits = logits.masked_scatter(
                    direct_mask, direct_logits
                )
            if grounded_logits is not None:
                logits = logits.masked_scatter(
                    grounded_mask, grounded_logits
                )
            return logits.to(dtype=self.direct_matcher.layers[-1].weight.dtype)

        def custom_state_dict(self) -> dict[str, Any]:
            return {
                key: value
                for key, value in self.state_dict().items()
                if key.startswith(self.CUSTOM_PREFIXES)
            }

        def load_custom_state_dict(self, state: Mapping[str, Any]) -> None:
            expected = set(self.custom_state_dict())
            actual = set(state)
            if actual != expected:
                raise ValueError(
                    f"custom checkpoint keys differ: missing={sorted(expected-actual)}, "
                    f"unexpected={sorted(actual-expected)}"
                )
            current = self.state_dict()
            current.update(state)
            self.load_state_dict(current, strict=True)

        @classmethod
        def from_architecture_file(cls, backbone: nn.Module, path: str | Path) -> "FactorMonitor":
            return cls(backbone, load_architecture(Path(path)))


    class EndpointBinaryMonitor(nn.Module):
        """M0 early-fusion endpoint baseline with trajectory-grouped logits."""

        CUSTOM_PREFIXES = (
            "endpoint_head.",
            "endpoint_role_embeddings",
            "factor_guided_readout.",
            "signed_criterion_panel.",
            "layer_mixed_endpoint.",
        )

        def __init__(
            self,
            backbone: nn.Module,
            architecture: FactorMonitorArchitecture,
            *,
            max_padded_tokens_per_backbone_forward: int | None = None,
            endpoint_role_embedding_fields: tuple[str, ...] | None = None,
            endpoint_decision_mode: str = SINGLE_ENDPOINT_DECISION_MODE,
            factor_guided_readout_config: Mapping[str, Any] | None = None,
            signed_criterion_panel_config: Mapping[str, Any] | None = None,
            layer_mixed_config: Mapping[str, Any] | None = None,
        ) -> None:
            super().__init__()
            if (
                max_padded_tokens_per_backbone_forward is not None
                and max_padded_tokens_per_backbone_forward <= 0
            ):
                raise ValueError(
                    "max_padded_tokens_per_backbone_forward must be positive"
                )
            if endpoint_decision_mode not in {
                SINGLE_ENDPOINT_DECISION_MODE,
                CRITERION_DIFFERENCE_DECISION_MODE,
            }:
                raise ValueError(f"unsupported endpoint decision mode: {endpoint_decision_mode}")
            if (
                endpoint_decision_mode == CRITERION_DIFFERENCE_DECISION_MODE
                and endpoint_role_embedding_fields is not None
            ):
                raise ValueError("criterion-difference mode does not support role embeddings")
            guided_enabled = bool(
                factor_guided_readout_config
                and factor_guided_readout_config.get("enabled")
            )
            panel_enabled = bool(
                signed_criterion_panel_config
                and signed_criterion_panel_config.get("enabled")
            )
            layer_mixed_enabled = bool(
                layer_mixed_config and layer_mixed_config.get("enabled")
            )
            if guided_enabled and endpoint_decision_mode != SINGLE_ENDPOINT_DECISION_MODE:
                raise ValueError("guided readout requires single-endpoint decision mode")
            if guided_enabled and endpoint_role_embedding_fields is not None:
                raise ValueError("guided readout and endpoint role embeddings are separate experiments")
            enabled_deltas = sum(
                bool(value)
                for value in (
                    endpoint_role_embedding_fields is not None,
                    guided_enabled,
                    panel_enabled,
                    layer_mixed_enabled,
                    endpoint_decision_mode == CRITERION_DIFFERENCE_DECISION_MODE,
                )
            )
            if enabled_deltas > 1:
                raise ValueError("endpoint architecture deltas must be enabled one at a time")
            if (panel_enabled or layer_mixed_enabled) and endpoint_decision_mode != SINGLE_ENDPOINT_DECISION_MODE:
                raise ValueError("panel and layer-mixed readouts require single-endpoint mode")
            self.architecture = architecture
            self.config = getattr(backbone, "config", None)
            self.main_input_name = "endpoint_input_ids"
            self.accepts_loss_kwargs = False
            self.supports_gradient_checkpointing = bool(
                getattr(backbone, "supports_gradient_checkpointing", False)
            )
            self.backbone_encoder = BackboneEncoder(backbone, architecture.hidden_size)
            self.max_padded_tokens_per_backbone_forward = (
                max_padded_tokens_per_backbone_forward
            )
            self.endpoint_role_embedding_fields = endpoint_role_embedding_fields
            self.endpoint_decision_mode = endpoint_decision_mode
            if endpoint_role_embedding_fields is not None:
                if len(endpoint_role_embedding_fields) != 6:
                    raise ValueError("endpoint role embedding requires exactly six fields")
                self.endpoint_role_embeddings = nn.Parameter(
                    torch.zeros(
                        len(endpoint_role_embedding_fields),
                        architecture.hidden_size,
                        dtype=torch.float32,
                    )
                )
            else:
                self.register_parameter("endpoint_role_embeddings", None)
            self.endpoint_head = nn.Linear(
                architecture.hidden_size, 1, dtype=torch.float32
            )
            self.factor_guided_readout_config = (
                dict(factor_guided_readout_config) if guided_enabled else None
            )
            self.factor_guided_readout = (
                FactorGuidedEvidenceReadout(
                    architecture.hidden_size,
                    readout_dim=int(factor_guided_readout_config["readout_dim"]),
                    interaction_hidden_dim=int(
                        factor_guided_readout_config["interaction_hidden_dim"]
                    ),
                )
                if guided_enabled
                else None
            )
            self.signed_criterion_panel_config = (
                dict(signed_criterion_panel_config) if panel_enabled else None
            )
            self.signed_criterion_panel = (
                SignedCriterionPanel(
                    architecture.hidden_size,
                    projection_dim=int(signed_criterion_panel_config["projection_dim"]),
                )
                if panel_enabled
                else None
            )
            self.layer_mixed_config = (
                dict(layer_mixed_config) if layer_mixed_enabled else None
            )
            self.layer_mixed_endpoint = (
                LayerMixedEndpoint(
                    architecture.hidden_size,
                    layers_below_final=tuple(
                        int(value) for value in layer_mixed_config["layers_below_final"]
                    ),
                )
                if layer_mixed_enabled
                else None
            )

        @property
        def backbone(self) -> nn.Module:
            return self.backbone_encoder.backbone

        @property
        def is_gradient_checkpointing(self) -> bool:
            return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

        def gradient_checkpointing_enable(
            self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
        ) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {})

        def gradient_checkpointing_disable(self) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_disable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method()

        def forward(
            self,
            *,
            endpoint_input_ids: Any,
            endpoint_attention_mask: Any,
            endpoint_role_ids: Any | None = None,
            factor_span_mask: Any | None = None,
            relation_span_mask: Any | None = None,
            applied_span_mask: Any | None = None,
            not_applied_span_mask: Any | None = None,
            context_span_mask: Any | None = None,
            cot_span_mask: Any | None = None,
            query_mask: Any,
        ) -> Any:
            expected_dimensions = (
                4
                if self.endpoint_decision_mode == CRITERION_DIFFERENCE_DECISION_MODE
                else 3
            )
            if endpoint_input_ids.ndim != expected_dimensions:
                expected_shape = (
                    "[B,Q,2,T]"
                    if expected_dimensions == 4
                    else "[B,Q,T]"
                )
                raise ValueError(
                    f"EndpointBinaryMonitor expects input_ids {expected_shape} "
                    f"for {self.endpoint_decision_mode}"
                )
            if endpoint_attention_mask.shape != endpoint_input_ids.shape:
                raise ValueError("endpoint attention mask must match input IDs")
            if query_mask.shape != endpoint_input_ids.shape[:2]:
                raise ValueError("query_mask must match endpoint [B,Q]")
            if endpoint_role_ids is not None and endpoint_role_ids.shape != endpoint_input_ids.shape:
                raise ValueError("endpoint role IDs must match input IDs")
            if self.endpoint_role_embeddings is not None and endpoint_role_ids is None:
                raise ValueError("enabled endpoint role embeddings require endpoint role IDs")
            guided_masks = (factor_span_mask, context_span_mask, cot_span_mask)
            if self.factor_guided_readout is not None:
                if any(mask is None for mask in guided_masks):
                    raise ValueError(
                        "enabled guided readout requires Factor, Context, and CoT span masks"
                    )
                for name, mask in zip(
                    ("Factor", "Context", "CoT"), guided_masks, strict=True
                ):
                    if mask.shape != endpoint_input_ids.shape:
                        raise ValueError(f"{name} span mask must match endpoint input IDs")
                    if torch.any(mask.bool() & ~endpoint_attention_mask.bool()):
                        raise ValueError(f"{name} span mask must be inside attention mask")
            panel_masks = (
                factor_span_mask,
                relation_span_mask,
                applied_span_mask,
                not_applied_span_mask,
            )
            if self.signed_criterion_panel is not None:
                if any(mask is None for mask in panel_masks):
                    raise ValueError(
                        "enabled signed criterion panel requires all four canonical field masks"
                    )
                for name, mask in zip(
                    ("Factor", "Relation", "Applied", "Not-Applied"),
                    panel_masks,
                    strict=True,
                ):
                    if mask.shape != endpoint_input_ids.shape:
                        raise ValueError(f"{name} span mask must match endpoint input IDs")
                    if torch.any(mask.bool() & ~endpoint_attention_mask.bool()):
                        raise ValueError(f"{name} span mask must be inside attention mask")
            valid_queries = query_mask.bool().reshape(-1)
            if torch.any(query_mask.bool().sum(dim=-1) <= 0):
                raise ValueError("every trajectory must contain at least one valid query")
            batch, records = endpoint_input_ids.shape[:2]
            tokens = endpoint_input_ids.shape[-1]
            if self.endpoint_decision_mode == CRITERION_DIFFERENCE_DECISION_MODE:
                if endpoint_input_ids.shape[2] != 2:
                    raise ValueError("criterion-difference endpoint must contain two branches")
                valid_sequences = (
                    query_mask.bool().unsqueeze(-1).expand(batch, records, 2).reshape(-1)
                )
                flattened_sequences = batch * records * 2
            else:
                valid_sequences = valid_queries
                flattened_sequences = batch * records
            flat_ids = endpoint_input_ids.reshape(flattened_sequences, tokens)[
                valid_sequences
            ]
            flat_attention = endpoint_attention_mask.reshape(
                flattened_sequences, tokens
            )[valid_sequences]
            flat_role_ids = (
                endpoint_role_ids.reshape(flattened_sequences, tokens)[valid_sequences]
                if endpoint_role_ids is not None
                else None
            )
            flat_factor_mask = (
                factor_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if factor_span_mask is not None
                else None
            )
            flat_relation_mask = (
                relation_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if relation_span_mask is not None
                else None
            )
            flat_applied_mask = (
                applied_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if applied_span_mask is not None
                else None
            )
            flat_not_applied_mask = (
                not_applied_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if not_applied_span_mask is not None
                else None
            )
            flat_context_mask = (
                context_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if context_span_mask is not None
                else None
            )
            flat_cot_mask = (
                cot_span_mask.reshape(flattened_sequences, tokens)[valid_sequences]
                if cot_span_mask is not None
                else None
            )
            sequence_lengths = flat_attention.long().sum(dim=-1)
            sequence_length_values = [
                int(value) for value in sequence_lengths.detach().cpu().tolist()
            ]
            budget = self.max_padded_tokens_per_backbone_forward
            chunk_ranges = _padded_token_chunk_ranges(
                sequence_length_values, budget
            )

            endpoint_chunks: list[Any] = []
            guided_delta_chunks: list[Any] = []
            panel_delta_chunks: list[Any] = []
            selected_layer_chunks: dict[int, list[Any]] = (
                {
                    offset: []
                    for offset in self.layer_mixed_endpoint.layers_below_final
                }
                if self.layer_mixed_endpoint is not None
                else {}
            )
            for start, end, chunk_tokens in chunk_ranges:
                chunk_ids = flat_ids[start:end, :chunk_tokens]
                chunk_attention = flat_attention[start:end, :chunk_tokens]
                token_additions = None
                if self.endpoint_role_embeddings is not None:
                    chunk_role_ids = flat_role_ids[start:end, :chunk_tokens]
                    if torch.any(chunk_role_ids < -1) or torch.any(
                        chunk_role_ids >= self.endpoint_role_embeddings.shape[0]
                    ):
                        raise ValueError("endpoint role IDs fall outside the registered mapping")
                    role_mask = chunk_role_ids >= 0
                    token_additions = self.endpoint_role_embeddings[
                        chunk_role_ids.clamp(min=0)
                    ] * role_mask.unsqueeze(-1)
                if self.layer_mixed_endpoint is None:
                    hidden = self.backbone_encoder.encode(
                        chunk_ids,
                        chunk_attention,
                        role="endpoint",
                        token_additions=token_additions,
                    )
                    selected_hidden = {}
                else:
                    hidden, selected_hidden = self.backbone_encoder.encode_selected_layers(
                        chunk_ids,
                        chunk_attention,
                        role="endpoint",
                        layers_below_final=self.layer_mixed_endpoint.layers_below_final,
                        token_additions=token_additions,
                    )
                endpoint_indices = chunk_attention.long().sum(dim=-1) - 1
                row_indices = torch.arange(hidden.shape[0], device=hidden.device)
                chunk_endpoint_hidden = hidden[row_indices, endpoint_indices]
                endpoint_chunks.append(chunk_endpoint_hidden)
                for offset, layer_hidden in selected_hidden.items():
                    selected_layer_chunks[offset].append(
                        layer_hidden[row_indices, endpoint_indices]
                    )
                if self.factor_guided_readout is not None:
                    chunk_factor_mask = flat_factor_mask[start:end, :chunk_tokens]
                    chunk_evidence_mask = (
                        flat_context_mask[start:end, :chunk_tokens]
                        | flat_cot_mask[start:end, :chunk_tokens]
                    )
                    guided_delta_chunks.append(
                        self.factor_guided_readout(
                            hidden,
                            factor_span_mask=chunk_factor_mask,
                            evidence_span_mask=chunk_evidence_mask,
                        )
                    )
                if self.signed_criterion_panel is not None:
                    panel_delta_chunks.append(
                        self.signed_criterion_panel(
                            hidden,
                            chunk_endpoint_hidden,
                            factor_span_mask=flat_factor_mask[start:end, :chunk_tokens],
                            relation_span_mask=flat_relation_mask[start:end, :chunk_tokens],
                            applied_span_mask=flat_applied_mask[start:end, :chunk_tokens],
                            not_applied_span_mask=flat_not_applied_mask[
                                start:end, :chunk_tokens
                            ],
                        )
                    )
            endpoint_hidden = torch.cat(endpoint_chunks, dim=0)
            scores_valid = self.endpoint_head(endpoint_hidden.float()).squeeze(-1)
            if guided_delta_chunks:
                scores_valid = scores_valid + torch.cat(guided_delta_chunks, dim=0)
            if panel_delta_chunks:
                scores_valid = scores_valid + torch.cat(panel_delta_chunks, dim=0)
            if self.layer_mixed_endpoint is not None:
                scores_valid = scores_valid + self.layer_mixed_endpoint(
                    {
                        offset: torch.cat(chunks, dim=0)
                        for offset, chunks in selected_layer_chunks.items()
                    }
                )
            if self.endpoint_decision_mode == CRITERION_DIFFERENCE_DECISION_MODE:
                branch_scores = scores_valid.reshape(-1, 2)
                logits_valid = branch_scores[:, 0] - branch_scores[:, 1]
            else:
                logits_valid = scores_valid
            logits = logits_valid.new_zeros(batch * records)
            logits[valid_queries] = logits_valid
            return logits.reshape(batch, records)

        def custom_state_dict(self) -> dict[str, Any]:
            return {
                key: value
                for key, value in self.state_dict().items()
                if key.startswith(self.CUSTOM_PREFIXES)
            }

        def load_custom_state_dict(self, state: Mapping[str, Any]) -> None:
            expected = set(self.custom_state_dict())
            actual = set(state)
            if actual != expected:
                raise ValueError(
                    f"custom checkpoint keys differ: missing={sorted(expected-actual)}, "
                    f"unexpected={sorted(actual-expected)}"
                )
            current = self.state_dict()
            current.update(state)
            self.load_state_dict(current, strict=True)


    class CausalLMChoiceMonitor(nn.Module):
        """One-token A/B SFT through the pretrained causal-LM output head."""

        CUSTOM_PREFIXES: tuple[str, ...] = ()

        def __init__(
            self,
            backbone: nn.Module,
            architecture: FactorMonitorArchitecture,
            *,
            choice_token_ids: Mapping[str, int],
            max_padded_tokens_per_backbone_forward: int | None = None,
        ) -> None:
            super().__init__()
            if set(choice_token_ids) != {"A", "B"}:
                raise ValueError("causal choice monitor requires exactly A and B token IDs")
            if choice_token_ids["A"] == choice_token_ids["B"]:
                raise ValueError("causal choice A and B token IDs must differ")
            if (
                max_padded_tokens_per_backbone_forward is not None
                and max_padded_tokens_per_backbone_forward <= 0
            ):
                raise ValueError("causal choice backbone token budget must be positive")
            self.architecture = architecture
            self.config = getattr(backbone, "config", None)
            self.main_input_name = "endpoint_input_ids"
            self.accepts_loss_kwargs = False
            self.supports_gradient_checkpointing = bool(
                getattr(backbone, "supports_gradient_checkpointing", False)
            )
            self._backbone = backbone
            self.max_padded_tokens_per_backbone_forward = (
                max_padded_tokens_per_backbone_forward
            )
            self.register_buffer(
                "choice_token_ids",
                torch.tensor(
                    [int(choice_token_ids["A"]), int(choice_token_ids["B"])],
                    dtype=torch.long,
                ),
                persistent=True,
            )

        @property
        def backbone(self) -> nn.Module:
            return self._backbone

        @property
        def is_gradient_checkpointing(self) -> bool:
            return bool(getattr(self.backbone, "is_gradient_checkpointing", False))

        def gradient_checkpointing_enable(
            self, gradient_checkpointing_kwargs: dict[str, Any] | None = None
        ) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_enable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method(gradient_checkpointing_kwargs=gradient_checkpointing_kwargs or {})

        def gradient_checkpointing_disable(self) -> None:
            method = getattr(self.backbone, "gradient_checkpointing_disable", None)
            if method is None:
                raise RuntimeError("backbone does not implement gradient checkpointing")
            method()

        def _decoder_and_lm_head(self) -> tuple[Any, Any]:
            get_base = getattr(self.backbone, "get_base_model", None)
            causal_lm = get_base() if callable(get_base) else self.backbone
            decoder = getattr(causal_lm, "model", None)
            get_output = getattr(causal_lm, "get_output_embeddings", None)
            lm_head = get_output() if callable(get_output) else None
            if decoder is None or not callable(decoder):
                raise RuntimeError(
                    "causal choice SFT requires a causal LM exposing its decoder as .model"
                )
            if lm_head is None or not callable(lm_head):
                raise RuntimeError(
                    "causal choice SFT requires pretrained output embeddings"
                )
            return decoder, lm_head

        def forward(
            self,
            *,
            endpoint_input_ids: Any,
            endpoint_attention_mask: Any,
            endpoint_role_ids: Any | None = None,
            query_mask: Any,
            choice_labels: Any | None = None,
        ) -> Any:
            del endpoint_role_ids
            if endpoint_input_ids.ndim != 3:
                raise ValueError("causal choice monitor expects input_ids [B,Q,T]")
            if endpoint_attention_mask.shape != endpoint_input_ids.shape:
                raise ValueError("causal choice attention mask must match input IDs")
            if query_mask.shape != endpoint_input_ids.shape[:2]:
                raise ValueError("query_mask must match causal choice [B,Q]")
            if torch.any(query_mask.bool().sum(dim=-1) <= 0):
                raise ValueError("every trajectory must contain at least one valid query")
            if choice_labels is not None:
                if choice_labels.shape != query_mask.shape:
                    raise ValueError("choice labels must match causal choice [B,Q]")
                valid_labels = choice_labels[query_mask.bool()]
                if torch.any((valid_labels < 0) | (valid_labels > 1)):
                    raise ValueError("valid causal choice labels must be zero or one")

            batch, records, tokens = endpoint_input_ids.shape
            valid_queries = query_mask.bool().reshape(-1)
            flat_ids = endpoint_input_ids.reshape(batch * records, tokens)[valid_queries]
            flat_attention = endpoint_attention_mask.reshape(batch * records, tokens)[
                valid_queries
            ]
            lengths = flat_attention.long().sum(dim=-1)
            if torch.any(lengths <= 0):
                raise ValueError("every causal choice prompt must contain an attended token")
            length_values = [int(value) for value in lengths.detach().cpu().tolist()]
            chunk_ranges = _padded_token_chunk_ranges(
                length_values, self.max_padded_tokens_per_backbone_forward
            )
            decoder, lm_head = self._decoder_and_lm_head()
            choice_logits_chunks: list[Any] = []
            vocabulary_logits_chunks: list[Any] = []
            for start, end, chunk_tokens in chunk_ranges:
                chunk_ids = flat_ids[start:end, :chunk_tokens]
                chunk_attention = flat_attention[start:end, :chunk_tokens]
                output = decoder(
                    input_ids=chunk_ids,
                    attention_mask=chunk_attention,
                    output_hidden_states=False,
                    return_dict=True,
                    use_cache=False,
                )
                hidden = getattr(output, "last_hidden_state", None)
                if hidden is None and isinstance(output, Mapping):
                    hidden = output.get("last_hidden_state")
                if hidden is None or hidden.shape[:2] != chunk_ids.shape:
                    raise RuntimeError("causal LM decoder returned invalid hidden states")
                endpoint_indices = chunk_attention.long().sum(dim=-1) - 1
                row_indices = torch.arange(hidden.shape[0], device=hidden.device)
                final_hidden = hidden[row_indices, endpoint_indices]
                vocabulary_logits = lm_head(final_hidden).float()
                if vocabulary_logits.ndim != 2:
                    raise RuntimeError("causal LM head must return [N,V] logits")
                selected = vocabulary_logits.index_select(
                    -1, self.choice_token_ids.to(vocabulary_logits.device)
                )
                choice_logits_chunks.append(selected[:, 1] - selected[:, 0])
                if choice_labels is not None:
                    vocabulary_logits_chunks.append(vocabulary_logits)

            logits_valid = torch.cat(choice_logits_chunks, dim=0)
            logits = logits_valid.new_zeros(batch * records)
            logits[valid_queries] = logits_valid
            logits = logits.reshape(batch, records)
            if choice_labels is None:
                return logits

            vocabulary_logits = torch.cat(vocabulary_logits_chunks, dim=0)
            valid_label_ids = choice_labels.reshape(-1)[valid_queries].long()
            target_token_ids = self.choice_token_ids.to(valid_label_ids.device)[
                valid_label_ids
            ]
            losses_valid = F.cross_entropy(
                vocabulary_logits,
                target_token_ids,
                reduction="none",
            )
            query_losses = losses_valid.new_zeros(batch * records)
            query_losses[valid_queries] = losses_valid
            return logits, query_losses.reshape(batch, records)

        def custom_state_dict(self) -> dict[str, Any]:
            return {"choice_token_ids": self.choice_token_ids}

        def load_custom_state_dict(self, state: Mapping[str, Any]) -> None:
            if set(state) != {"choice_token_ids"}:
                raise ValueError("causal choice checkpoint must contain choice_token_ids")
            value = state["choice_token_ids"].to(
                device=self.choice_token_ids.device,
                dtype=self.choice_token_ids.dtype,
            )
            if not torch.equal(value, self.choice_token_ids):
                raise ValueError("causal choice checkpoint changed A/B token IDs")


else:

    class FactorMonitor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class EndpointBinaryMonitor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


    class CausalLMChoiceMonitor:  # pragma: no cover
        def __init__(self, *_: Any, **__: Any) -> None:
            _require_torch()


def assert_factor_monitor_trainable_parameters(model: Any) -> dict[str, int]:
    """Allow only LoRA parameters and the frozen list of custom modules."""

    allowed_custom = FactorMonitor.CUSTOM_PREFIXES if nn is not None else ()
    counts = {"lora": 0, "custom": 0, "unexpected": 0}
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            counts["lora"] += parameter.numel()
        elif name.startswith(allowed_custom):
            counts["custom"] += parameter.numel()
        else:
            counts["unexpected"] += parameter.numel()
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"unexpected trainable parameters: {unexpected}")
    if counts["custom"] == 0:
        raise RuntimeError("FactorMonitor has no trainable custom parameters")
    if counts["lora"] == 0:
        raise RuntimeError("FactorMonitor has no trainable LoRA parameters")
    return counts


def assert_endpoint_monitor_trainable_parameters(model: Any) -> dict[str, int]:
    """Allow only the shared LoRA adapter and registered endpoint modules."""

    allowed_custom = EndpointBinaryMonitor.CUSTOM_PREFIXES if nn is not None else ()
    counts = {"lora": 0, "custom": 0, "unexpected": 0}
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            counts["lora"] += parameter.numel()
        elif name.startswith(allowed_custom):
            counts["custom"] += parameter.numel()
        else:
            counts["unexpected"] += parameter.numel()
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(f"unexpected trainable M0 parameters: {unexpected}")
    if counts["custom"] == 0 or counts["lora"] == 0:
        raise RuntimeError(f"M0 requires non-empty LoRA and endpoint head: {counts}")
    return counts


def assert_causal_choice_trainable_parameters(model: Any) -> dict[str, int]:
    """Require LoRA-only training with no custom classification parameters."""

    counts = {"lora": 0, "custom": 0, "unexpected": 0}
    unexpected: list[str] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if "lora_" in name:
            counts["lora"] += parameter.numel()
        else:
            counts["unexpected"] += parameter.numel()
            unexpected.append(name)
    if unexpected:
        raise RuntimeError(
            f"causal choice SFT permits only LoRA trainable parameters: {unexpected}"
        )
    if counts["lora"] == 0:
        raise RuntimeError("causal choice SFT has no trainable LoRA parameters")
    return counts
