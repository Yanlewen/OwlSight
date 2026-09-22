"""Phase-1 hierarchical query → Factor instance → trajectory objective."""

from __future__ import annotations

from typing import Any

try:
    import torch
    import torch.nn.functional as F
except ModuleNotFoundError:
    torch = None
    F = None


def hierarchical_binary_loss(
    logits: Any,
    labels: Any,
    query_mask: Any,
    factor_instance_index: Any,
    factor_level: Any,
    family_id: Any,
    loss_weight: Any | None = None,
) -> dict[str, Any]:
    """Average levels within instance, instances within trajectory, then trajectories."""

    if torch is None or F is None:
        raise RuntimeError("hierarchical_binary_loss requires torch")
    expected = logits.shape
    for name, value in (
        ("labels", labels),
        ("query_mask", query_mask),
        ("factor_instance_index", factor_instance_index),
        ("factor_level", factor_level),
        ("family_id", family_id),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if logits.ndim != 2:
        raise ValueError("hierarchical loss expects [B,Q]")
    mask = query_mask.bool()
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every trajectory must have at least one valid query")
    if torch.any((labels[mask] < 0) | (labels[mask] > 1)):
        raise ValueError("valid labels must be in [0,1]")
    per_query = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return hierarchical_query_loss(
        per_query,
        query_mask,
        factor_instance_index,
        factor_level,
        family_id,
        loss_weight,
    )


def hierarchical_query_loss(
    per_query_loss: Any,
    query_mask: Any,
    factor_instance_index: Any,
    factor_level: Any,
    family_id: Any,
    loss_weight: Any | None = None,
) -> dict[str, Any]:
    """Aggregate arbitrary per-query losses with the validation hierarchy."""

    if torch is None:
        raise RuntimeError("hierarchical_query_loss requires torch")
    expected = per_query_loss.shape
    for name, value in (
        ("query_mask", query_mask),
        ("factor_instance_index", factor_instance_index),
        ("factor_level", factor_level),
        ("family_id", family_id),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if per_query_loss.ndim != 2:
        raise ValueError("hierarchical query loss expects [B,Q]")
    mask = query_mask.bool()
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every trajectory must have at least one valid query")
    if torch.any(~torch.isfinite(per_query_loss[mask])):
        raise ValueError("valid per-query losses must be finite")
    if loss_weight is not None:
        if loss_weight.shape != expected:
            raise ValueError(f"loss_weight must have shape {expected}")
        weights = loss_weight.to(
            per_query_loss.device, dtype=per_query_loss.dtype
        )
        if torch.any(~torch.isfinite(weights[mask])) or torch.any(weights[mask] <= 0):
            raise ValueError("valid loss weights must be positive and finite")
        per_query_loss = per_query_loss * weights
    trajectory_losses: list[Any] = []
    instance_counts: list[int] = []
    for batch_index in range(per_query_loss.shape[0]):
        valid = mask[batch_index]
        indices = factor_instance_index[batch_index, valid].long()
        levels = factor_level[batch_index, valid].long()
        families = family_id[batch_index, valid].long()
        if torch.any(indices < 0) or torch.any(levels < 0) or torch.any(families < 0):
            raise ValueError("valid hierarchy metadata may not use padding sentinel -1")
        if torch.unique(families).numel() != 1:
            raise ValueError("one trajectory may not cross families")
        unique_indices = torch.unique(indices, sorted=True)
        expected_indices = torch.arange(
            unique_indices.numel(), device=indices.device, dtype=indices.dtype
        )
        if not torch.equal(unique_indices, expected_indices):
            raise ValueError("factor_instance_index must be local and contiguous from zero")
        factor_losses: list[Any] = []
        for instance in unique_indices:
            selected = indices == instance
            selected_levels = levels[selected]
            if torch.unique(selected_levels).numel() != selected_levels.numel():
                raise ValueError("a Factor instance may contain at most one query per level")
            factor_losses.append(
                per_query_loss[batch_index, valid][selected].mean()
            )
        trajectory_losses.append(torch.stack(factor_losses).mean())
        instance_counts.append(len(factor_losses))
    stacked = torch.stack(trajectory_losses)
    return {
        "loss": stacked.mean(),
        "trajectory_losses": stacked.detach(),
        "valid_query_count": int(mask.sum().item()),
        "factor_instance_count": int(sum(instance_counts)),
        "trajectory_count": per_query_loss.shape[0],
    }


def factor_instance_uniform_binary_loss(
    logits: Any,
    labels: Any,
    query_mask: Any,
    factor_instance_index: Any,
    factor_level: Any,
    family_id: Any,
    training_unit_index: Any,
    loss_weight: Any | None = None,
) -> dict[str, Any]:
    """Average levels per instance, then instances per selected contrast unit.

    The sampler selects Factor-instance anchors. A complete contrast unit may
    span two trajectory rows, so the trajectory layer is intentionally absent.
    Under DDP, the differentiable local sum is scaled by the actual global unit
    draw count rather than by the number of ranks.
    """

    if torch is None or F is None:
        raise RuntimeError("factor_instance_uniform_binary_loss requires torch")
    expected = logits.shape
    for name, value in (
        ("labels", labels),
        ("query_mask", query_mask),
        ("factor_instance_index", factor_instance_index),
        ("factor_level", factor_level),
        ("family_id", family_id),
        ("training_unit_index", training_unit_index),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if logits.ndim != 2:
        raise ValueError("factor-instance-uniform loss expects [B,Q]")
    mask = query_mask.bool()
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every trajectory must have at least one valid query")
    if torch.any((labels[mask] < 0) | (labels[mask] > 1)):
        raise ValueError("valid labels must be in [0,1]")
    per_query = F.binary_cross_entropy_with_logits(
        logits,
        labels.to(logits.device, dtype=logits.dtype),
        reduction="none",
    )
    return factor_instance_uniform_query_loss(
        per_query,
        query_mask,
        factor_instance_index,
        factor_level,
        family_id,
        training_unit_index,
        loss_weight,
    )


def factor_instance_uniform_query_loss(
    per_query_loss: Any,
    query_mask: Any,
    factor_instance_index: Any,
    factor_level: Any,
    family_id: Any,
    training_unit_index: Any,
    loss_weight: Any | None = None,
) -> dict[str, Any]:
    """Aggregate arbitrary per-query losses with the frozen P1 hierarchy."""

    if torch is None:
        raise RuntimeError("factor_instance_uniform_query_loss requires torch")
    expected = per_query_loss.shape
    for name, value in (
        ("query_mask", query_mask),
        ("factor_instance_index", factor_instance_index),
        ("factor_level", factor_level),
        ("family_id", family_id),
        ("training_unit_index", training_unit_index),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if per_query_loss.ndim != 2:
        raise ValueError("factor-instance-uniform query loss expects [B,Q]")
    mask = query_mask.bool()
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every trajectory must have at least one valid query")
    if torch.any(~torch.isfinite(per_query_loss[mask])):
        raise ValueError("valid per-query losses must be finite")
    if loss_weight is not None:
        if loss_weight.shape != expected:
            raise ValueError(f"loss_weight must have shape {expected}")
        weights = loss_weight.to(
            per_query_loss.device, dtype=per_query_loss.dtype
        )
        if torch.any(~torch.isfinite(weights[mask])) or torch.any(weights[mask] <= 0):
            raise ValueError("valid loss weights must be positive and finite")
        per_query = per_query_loss * weights
    else:
        per_query = per_query_loss
    losses_by_unit: dict[int, list[Any]] = {}
    unit_families: dict[int, int] = {}
    instance_count = 0
    for batch_index in range(per_query_loss.shape[0]):
        valid = mask[batch_index]
        indices = factor_instance_index[batch_index, valid].long()
        levels = factor_level[batch_index, valid].long()
        families = family_id[batch_index, valid].long()
        units = training_unit_index[batch_index, valid].long()
        if torch.any(indices < 0) or torch.any(levels < 0) or torch.any(families < 0):
            raise ValueError("valid hierarchy metadata may not use padding sentinel -1")
        if torch.any(units < 0) or torch.unique(units).numel() != 1:
            raise ValueError("one trajectory row must belong to exactly one selected unit")
        if torch.unique(families).numel() != 1:
            raise ValueError("one trajectory may not cross families")
        unit = int(units[0].item())
        family = int(families[0].item())
        if unit in unit_families and unit_families[unit] != family:
            raise ValueError("one selected unit may not cross families")
        unit_families[unit] = family
        unique_indices = torch.unique(indices, sorted=True)
        expected_indices = torch.arange(
            unique_indices.numel(), device=indices.device, dtype=indices.dtype
        )
        if not torch.equal(unique_indices, expected_indices):
            raise ValueError("factor_instance_index must be local and contiguous from zero")
        selected_query_losses = per_query[batch_index, valid]
        for instance in unique_indices:
            selected = indices == instance
            selected_levels = levels[selected]
            if torch.unique(selected_levels).numel() != selected_levels.numel():
                raise ValueError("a Factor instance may contain at most one query per level")
            losses_by_unit.setdefault(unit, []).append(
                selected_query_losses[selected].mean()
            )
            instance_count += 1
    unit_ids = sorted(losses_by_unit)
    if unit_ids != list(range(len(unit_ids))):
        raise ValueError("training_unit_index must be local and contiguous from zero")
    unit_losses = torch.stack(
        [torch.stack(losses_by_unit[unit]).mean() for unit in unit_ids]
    )
    local_count = torch.tensor(
        float(unit_losses.numel()),
        device=per_query_loss.device,
        dtype=per_query_loss.dtype,
    )
    global_count = local_count.detach().clone()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    loss = unit_losses.sum() * world_size / global_count
    return {
        "loss": loss,
        "training_unit_losses": unit_losses.detach(),
        "valid_query_count": int(mask.sum().item()),
        "factor_instance_count": instance_count,
        "trajectory_count": per_query_loss.shape[0],
        "training_unit_count": len(unit_ids),
        "global_training_unit_count": int(global_count.item()),
    }


def mil_factor_instance_uniform_binary_loss(
    logits: Any,
    labels: Any,
    query_mask: Any,
    mil_bag_index: Any,
    mil_factor_instance_index: Any,
    factor_level: Any,
    family_id: Any,
    training_unit_index: Any,
    *,
    pooling: str,
    loss_weight: Any | None = None,
    training_unit_weight: Any | None = None,
    teacher_logits: Any | None = None,
    mil_is_long: Any | None = None,
    teacher_temperature: float | None = None,
    teacher_loss_weight: float | None = None,
) -> dict[str, Any]:
    """Pool window logits per source query before Factor-instance averaging.

    A direct short-context query is represented by a one-window bag, making
    this objective exactly equivalent to BCE-with-logits for that query.  The
    remaining hierarchy matches :func:`factor_instance_uniform_binary_loss`:
    source-query bags are averaged within a Factor instance, then Factor
    instances are averaged within the selected training unit.
    """

    if torch is None or F is None:
        raise RuntimeError("mil_factor_instance_uniform_binary_loss requires torch")
    expected = logits.shape
    for name, value in (
        ("labels", labels),
        ("query_mask", query_mask),
        ("mil_bag_index", mil_bag_index),
        ("mil_factor_instance_index", mil_factor_instance_index),
        ("factor_level", factor_level),
        ("family_id", family_id),
        ("training_unit_index", training_unit_index),
    ):
        if value.shape != expected:
            raise ValueError(f"{name} must have shape {expected}")
    if logits.ndim != 2:
        raise ValueError("MIL factor-instance loss expects [B,Q]")
    if pooling not in {"top2_probability_mean", "top2_logmeanexp"}:
        raise ValueError(f"unsupported MIL pooling: {pooling!r}")
    teacher_values = (
        teacher_logits,
        mil_is_long,
        teacher_temperature,
        teacher_loss_weight,
    )
    teacher_enabled = any(value is not None for value in teacher_values)
    if teacher_enabled and any(value is None for value in teacher_values):
        raise ValueError("Teacher distillation arguments must be provided together")
    if teacher_enabled:
        if teacher_logits.shape != expected or mil_is_long.shape != expected:
            raise ValueError("Teacher logits and long flags must match MIL logits")
        if teacher_temperature != 2.0 or teacher_loss_weight != 0.25:
            raise ValueError("Teacher distillation requires T=2 and loss weight 0.25")

    mask = query_mask.bool()
    if torch.any(mask.sum(dim=-1) <= 0):
        raise ValueError("every materialized trajectory must contain a valid query")
    flat_logits = logits[mask]
    flat_labels = labels[mask].to(logits.device, dtype=logits.dtype)
    flat_bags = mil_bag_index[mask].long()
    flat_instances = mil_factor_instance_index[mask].long()
    flat_levels = factor_level[mask].long()
    flat_families = family_id[mask].long()
    flat_units = training_unit_index[mask].long()
    flat_teacher = None
    flat_long = None
    if teacher_enabled:
        flat_teacher = teacher_logits[mask].to(logits.device, dtype=torch.float32)
        flat_long = mil_is_long[mask].long()
        if torch.any((flat_long < 0) | (flat_long > 1)):
            raise ValueError("valid MIL long flags must be zero or one")
        if torch.any(~torch.isfinite(flat_teacher[flat_long == 1])):
            raise ValueError("every valid long window requires a finite Teacher logit")
        if torch.any(torch.isfinite(flat_teacher[flat_long == 0])):
            raise ValueError("short MIL queries must not carry Teacher logits")
    if torch.any((flat_labels < 0) | (flat_labels > 1)):
        raise ValueError("valid labels must be in [0,1]")
    for name, value in (
        ("mil_bag_index", flat_bags),
        ("mil_factor_instance_index", flat_instances),
        ("factor_level", flat_levels),
        ("family_id", flat_families),
        ("training_unit_index", flat_units),
    ):
        if torch.any(value < 0):
            raise ValueError(f"valid {name} values may not use padding sentinel -1")
    flat_weights = None
    if loss_weight is not None:
        if loss_weight.shape != expected:
            raise ValueError(f"loss_weight must have shape {expected}")
        flat_weights = loss_weight[mask].to(logits.device, dtype=logits.dtype)
        if torch.any(~torch.isfinite(flat_weights)) or torch.any(flat_weights <= 0):
            raise ValueError("valid loss weights must be positive and finite")

    unique_bags = torch.unique(flat_bags, sorted=True)
    expected_bags = torch.arange(
        unique_bags.numel(), device=flat_bags.device, dtype=flat_bags.dtype
    )
    if not torch.equal(unique_bags, expected_bags):
        raise ValueError("mil_bag_index must be local and contiguous from zero")

    losses_by_instance: dict[int, list[Any]] = {}
    instance_units: dict[int, int] = {}
    instance_families: dict[int, int] = {}
    instance_levels: dict[int, set[int]] = {}
    bag_sizes: list[int] = []
    pooled_logits: list[Any] = []
    teacher_losses_by_instance: dict[int, list[Any]] = {}
    teacher_instance_units: dict[int, int] = {}
    teacher_bags_by_unit: dict[int, int] = {}
    for bag in unique_bags:
        selected = flat_bags == bag
        bag_logits = flat_logits[selected]
        bag_sizes.append(int(bag_logits.numel()))
        k = min(2, int(bag_logits.numel()))
        top = torch.topk(bag_logits, k=k).values
        if pooling == "top2_logmeanexp":
            pooled_logit = torch.logsumexp(top, dim=0) - torch.log(
                torch.tensor(float(k), device=top.device, dtype=top.dtype)
            )
            bag_loss = F.binary_cross_entropy_with_logits(
                pooled_logit, flat_labels[selected][0], reduction="none"
            )
        else:
            probability = torch.sigmoid(top).mean()
            bag_loss = F.binary_cross_entropy(
                probability, flat_labels[selected][0], reduction="none"
            )
            pooled_logit = torch.logit(
                probability.clamp(
                    min=torch.finfo(probability.dtype).eps,
                    max=1.0 - torch.finfo(probability.dtype).eps,
                )
            )
        pooled_logits.append(pooled_logit)

        def one_value(name: str, value: Any) -> Any:
            unique = torch.unique(value[selected])
            if unique.numel() != 1:
                raise ValueError(f"one source-query bag must have one {name}")
            return unique[0]

        label = one_value("label", flat_labels)
        if label.item() != flat_labels[selected][0].item():
            raise ValueError("one source-query bag must have one label")
        instance = int(one_value("Factor instance", flat_instances).item())
        level = int(one_value("Factor level", flat_levels).item())
        family = int(one_value("family", flat_families).item())
        unit = int(one_value("training unit", flat_units).item())
        is_long = (
            int(one_value("long flag", flat_long).item())
            if flat_long is not None
            else 0
        )
        if flat_weights is not None:
            weight = one_value("loss weight", flat_weights)
            bag_loss = bag_loss * weight
        else:
            weight = None
        if teacher_enabled and is_long == 1 and int(label.item()) == 1:
            if bag_logits.numel() < 2:
                raise ValueError(
                    "applied long Teacher bags require at least two active windows"
                )
            assert flat_teacher is not None
            selected_teacher = flat_teacher[selected]
            teacher_log_probabilities = F.log_softmax(
                selected_teacher / float(teacher_temperature), dim=0
            )
            teacher_probabilities = teacher_log_probabilities.exp().detach()
            student_log_probabilities = F.log_softmax(
                bag_logits.float() / float(teacher_temperature), dim=0
            )
            teacher_loss = torch.sum(
                teacher_probabilities
                * (teacher_log_probabilities.detach() - student_log_probabilities)
            )
            if weight is not None:
                teacher_loss = teacher_loss * weight.float()
            teacher_losses_by_instance.setdefault(instance, []).append(teacher_loss)
            teacher_instance_units[instance] = unit
            teacher_bags_by_unit[unit] = teacher_bags_by_unit.get(unit, 0) + 1
        if instance in instance_units and instance_units[instance] != unit:
            raise ValueError("one Factor instance may not cross training units")
        if instance in instance_families and instance_families[instance] != family:
            raise ValueError("one Factor instance may not cross families")
        levels = instance_levels.setdefault(instance, set())
        if level in levels:
            raise ValueError("a Factor instance may contain at most one bag per level")
        levels.add(level)
        instance_units[instance] = unit
        instance_families[instance] = family
        losses_by_instance.setdefault(instance, []).append(bag_loss)

    unique_instances = sorted(losses_by_instance)
    if unique_instances != list(range(len(unique_instances))):
        raise ValueError("mil_factor_instance_index must be local and contiguous from zero")
    losses_by_unit: dict[int, list[Any]] = {}
    unit_families: dict[int, int] = {}
    for instance in unique_instances:
        unit = instance_units[instance]
        family = instance_families[instance]
        if unit in unit_families and unit_families[unit] != family:
            raise ValueError("one selected training unit may not cross families")
        unit_families[unit] = family
        losses_by_unit.setdefault(unit, []).append(
            torch.stack(losses_by_instance[instance]).mean()
        )
    unit_ids = sorted(losses_by_unit)
    if unit_ids != list(range(len(unit_ids))):
        raise ValueError("training_unit_index must be local and contiguous from zero")
    unit_losses = torch.stack(
        [torch.stack(losses_by_unit[unit]).mean() for unit in unit_ids]
    )
    if training_unit_weight is None:
        unit_weights = torch.ones_like(unit_losses)
    else:
        if training_unit_weight.ndim != 1 or training_unit_weight.numel() != len(unit_ids):
            raise ValueError(
                "training_unit_weight must contain one value per local training unit"
            )
        unit_weights = training_unit_weight.to(
            logits.device, dtype=logits.dtype
        )
        if torch.any(~torch.isfinite(unit_weights)) or torch.any(
            (unit_weights != 0) & (unit_weights != 1)
        ):
            raise ValueError("training_unit_weight values must be finite zero or one")
    local_count = torch.tensor(
        float(unit_weights.sum().item()), device=logits.device, dtype=logits.dtype
    )
    global_count = local_count.detach().clone()
    world_size = 1
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_reduce(global_count, op=torch.distributed.ReduceOp.SUM)
        world_size = torch.distributed.get_world_size()
    if float(global_count.item()) <= 0:
        raise ValueError("one synchronized MIL microbatch must contain a real training unit")
    mil_loss = (unit_losses * unit_weights).sum() * world_size / global_count
    teacher_loss = logits.sum() * 0.0
    teacher_global_mean = torch.zeros(
        (), device=logits.device, dtype=torch.float32
    )
    teacher_global_units = torch.zeros(
        (), device=logits.device, dtype=torch.float32
    )
    teacher_global_bags = torch.zeros((), device=logits.device, dtype=torch.float32)
    if teacher_enabled:
        teacher_losses_by_unit: dict[int, list[Any]] = {}
        for teacher_instance, instance_losses in teacher_losses_by_instance.items():
            teacher_unit = teacher_instance_units[teacher_instance]
            teacher_losses_by_unit.setdefault(teacher_unit, []).append(
                torch.stack(instance_losses).mean()
            )
        teacher_unit_ids = sorted(teacher_losses_by_unit)
        teacher_unit_losses = (
            torch.stack(
                [
                    torch.stack(teacher_losses_by_unit[unit]).mean()
                    for unit in teacher_unit_ids
                ]
            )
            if teacher_unit_ids
            else logits.new_empty((0,), dtype=torch.float32)
        )
        teacher_unit_weights = (
            torch.stack([unit_weights[unit].float() for unit in teacher_unit_ids])
            if teacher_unit_ids
            else logits.new_empty((0,), dtype=torch.float32)
        )
        teacher_global_bags = torch.stack(
            [
                unit_weights[unit].float() * float(teacher_bags_by_unit[unit])
                for unit in teacher_unit_ids
            ]
        ).sum() if teacher_unit_ids else torch.zeros(
            (), device=logits.device, dtype=torch.float32
        )
        local_teacher_sum = (teacher_unit_losses * teacher_unit_weights).sum()
        teacher_global_units = teacher_unit_weights.sum().detach()
        global_teacher_sum = local_teacher_sum.detach().clone()
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                teacher_global_units, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                teacher_global_bags, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                global_teacher_sum, op=torch.distributed.ReduceOp.SUM
            )
        if float(teacher_global_units.item()) > 0:
            teacher_loss = (
                local_teacher_sum * world_size / teacher_global_units
            ).to(logits.dtype)
            teacher_global_mean = global_teacher_sum / teacher_global_units
    loss = (
        mil_loss + float(teacher_loss_weight) * teacher_loss
        if teacher_enabled
        else mil_loss
    )
    return {
        "loss": loss,
        "mil_loss": mil_loss.detach(),
        "teacher_kl_loss": teacher_global_mean.detach(),
        "weighted_teacher_kl_loss": (
            teacher_global_mean.detach() * float(teacher_loss_weight or 0.0)
        ),
        "teacher_eligible_query_count": int(teacher_global_bags.item()),
        "teacher_eligible_unit_count": int(teacher_global_units.item()),
        "training_unit_losses": unit_losses.detach(),
        "pooled_logits": torch.stack(pooled_logits).detach(),
        "bag_sizes": tuple(bag_sizes),
        "valid_window_query_count": int(mask.sum().item()),
        "source_query_count": len(bag_sizes),
        "factor_instance_count": len(unique_instances),
        "training_unit_count": int(unit_weights.sum().item()),
        "sync_padding_unit_count": int((unit_weights == 0).sum().item()),
        "global_training_unit_count": int(global_count.item()),
    }
