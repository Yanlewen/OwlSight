"""Reusable browser-only blind-audit rendering and overlay validation."""

from __future__ import annotations

import base64
import json
from html import escape
from pathlib import Path
from typing import Any, Iterable

from .config import ConfigurationError
from .io import read_jsonl


def validate_reference_reviews(
    queue: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    decisions: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Validate one hidden reference row for every blind-audit item."""

    allowed = set(decisions)
    if len(queue) != len(references):
        raise ConfigurationError("audit queue and reference-review row counts differ")
    by_id = {row.get("audit_id"): row for row in references}
    if len(by_id) != len(references) or None in by_id:
        raise ConfigurationError("reference reviews contain duplicate or missing audit IDs")
    for item in queue:
        audit_id = item.get("audit_id")
        reference = by_id.get(audit_id)
        if reference is None or reference.get("queue_item_hash") != item.get(
            "queue_item_hash"
        ):
            raise ConfigurationError(
                f"reference review is missing or stale for {audit_id}"
            )
        if reference.get("decision") not in allowed:
            raise ConfigurationError(
                f"reference review has invalid decision for {audit_id}"
            )
    return by_id


def render_blind_audit_html(
    queue: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    decisions: Iterable[str],
    title: str,
    heading: str,
    instructions: str,
    storage_namespace: str,
    reveal_heading: str,
    reveal_button: str,
    hidden_notice: str,
    diagnostic_heading: str,
    task_context_heading: str = "Task context（solver-visible）",
    factor_heading: str = "Factor",
    model_cot_heading: str = "Model thinking",
    invalid_label: str = "factor_invalid",
    evidence_heading: str = "Evidence spans（JSON string array，可为空）",
    rationale_heading: str = "Rationale",
    export_name: str = "owner_overlay.jsonl",
) -> str:
    """Render a self-contained audit page whose reference stays hidden initially."""

    allowed = tuple(decisions)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ConfigurationError("audit decisions must be unique and non-empty")
    validate_reference_reviews(queue, references, decisions=allowed)
    payload = base64.b64encode(
        json.dumps(
            {
                "queue": queue,
                "references": references,
                "decisions": allowed,
                "text": {
                    "title": title,
                    "heading": heading,
                    "instructions": instructions,
                    "storage_namespace": storage_namespace,
                    "reveal_heading": reveal_heading,
                    "reveal_button": reveal_button,
                    "hidden_notice": hidden_notice,
                    "diagnostic_heading": diagnostic_heading,
                    "task_context_heading": task_context_heading,
                    "factor_heading": factor_heading,
                    "model_cot_heading": model_cot_heading,
                    "invalid_label": invalid_label,
                    "evidence_heading": evidence_heading,
                    "rationale_heading": rationale_heading,
                    "export_name": export_name,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    searchable_text = " | ".join(
        (
            heading,
            instructions,
            reveal_heading,
            reveal_button,
            hidden_notice,
            task_context_heading,
            factor_heading,
            model_cot_heading,
            invalid_label,
            evidence_heading,
            rationale_heading,
            *allowed,
        )
    )
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
body{margin:0;background:#f5f2ea;color:#27231d;font:15px/1.5 system-ui,sans-serif}header{position:sticky;top:0;background:#f5f2eaf2;border-bottom:1px solid #d7d0c3;padding:12px 3vw;z-index:2}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.bar h1{font-size:19px;margin:0 auto 0 0}button,select,input,textarea{font:inherit}button,select,input{padding:7px 10px;border:1px solid #cfc6b7;border-radius:7px;background:white}.active{background:#315d52;color:white}.reveal{background:#70531b;color:white}.notice{padding:10px;border:1px dashed #b99b5a;border-radius:8px;background:#fff8df}.reference{margin-top:14px;border:2px solid #b99b5a;background:#fffaf0}.badge{display:inline-block;padding:2px 7px;border-radius:12px;background:#eee5cf;margin-right:5px}main{max-width:1400px;margin:18px auto;padding:0 3vw 60px}.card{background:#fffdf7;border:1px solid #d7d0c3;border-radius:12px;padding:16px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{border:1px solid #ddd4c5;border-radius:8px;padding:10px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:55vh;overflow:auto;background:#f8f5ee;padding:10px}.choices{display:flex;gap:7px;flex-wrap:wrap;margin:12px 0}textarea{width:100%;min-height:70px}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><!-- __SEARCHABLE_TEXT__ --><header><div class="bar"><h1 id="heading"></h1><span id="progress"></span><select id="filter"><option value="all">全部</option><option value="unreviewed">未完成盲选/揭示</option></select><button id="prev">上一条</button><span id="position"></span><button id="next">下一条</button><input id="annotator" placeholder="annotator ID"><button id="export">导出 owner_overlay.jsonl</button></div><div id="instructions"></div></header><main id="app"></main><script>
const payload=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__PAYLOAD__'),c=>c.charCodeAt(0))));const rows=payload.queue,referenceById=Object.fromEntries(payload.references.map(x=>[x.audit_id,x])),decisions=payload.decisions,text=payload.text;document.getElementById('heading').textContent=text.heading;document.getElementById('instructions').textContent=text.instructions;const key=text.storage_namespace+':'+rows.map(x=>x.queue_item_hash).join(':');let state=JSON.parse(localStorage.getItem(key)||'{}'),visible=[],cursor=0;
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function persist(){localStorage.setItem(key,JSON.stringify(state))}function save(){persist();render()};function filtered(){return rows.filter(r=>document.getElementById('filter').value==='all'||!state[r.audit_id]?.decision||!state[r.audit_id]?.revealed)}
function referencePanel(r,s){if(!s.decision)return `<div class="notice"><b>${esc(text.hidden_notice)}</b></div>`;if(!s.revealed)return `<div class="notice"><b>你的盲选已保存：${esc(s.decision)}</b><br><button class="reveal" onclick="revealReview('${r.audit_id}')">${esc(text.reveal_button)}</button></div>`;const a=referenceById[r.audit_id];return `<section class="panel reference"><h3>${esc(text.reveal_heading)}</h3><p><span class="badge">${esc(a.decision)}</span><span class="badge">confidence=${esc(a.confidence)}</span><span class="badge">${esc(a.review_method)}</span></p><p>${esc(a.rationale)}</p><h3>CoT evidence spans</h3><pre>${esc(JSON.stringify(a.evidence_spans||[],null,2))}</pre><details><summary>${esc(text.diagnostic_heading)}</summary><pre>${esc(JSON.stringify(a.judge_diagnostic||{},null,2))}</pre></details></section>`}
function card(r){const s=state[r.audit_id]||{};return `<article class="card"><h2>${esc(r.audit_id)} · ${esc(r.source)} · ${esc(r.factor_level)}</h2><div class="grid"><section class="panel"><h3>${esc(text.task_context_heading)}</h3><pre>${esc(r.task_context)}</pre><h3>${esc(text.factor_heading)}</h3><pre>${esc(JSON.stringify(r.factor_spec,null,2))}</pre></section><section class="panel"><h3>${esc(text.model_cot_heading)}</h3><pre>${esc(r.model_cot)}</pre></section></div><div class="choices">${decisions.map(d=>`<button class="${s.decision===d?'active':''}" onclick="choose('${r.audit_id}','${d}')">${esc(d)}</button>`).join('')}</div><label><input type="checkbox" ${s.factor_invalid?'checked':''} onchange="field('${r.audit_id}','factor_invalid',this.checked)"> ${esc(text.invalid_label)}</label><p>${esc(text.evidence_heading)}</p><textarea oninput="field('${r.audit_id}','evidence',this.value)">${esc(s.evidence||'[]')}</textarea><p>${esc(text.rationale_heading)}</p><textarea oninput="field('${r.audit_id}','rationale',this.value)">${esc(s.rationale||'')}</textarea>${referencePanel(r,s)}</article>`}
function render(){visible=filtered();if(cursor>=visible.length)cursor=Math.max(0,visible.length-1);document.getElementById('app').innerHTML=visible.length?card(visible[cursor]):'<p>当前筛选没有条目。</p>';document.getElementById('position').textContent=visible.length?`${cursor+1}/${visible.length}`:'0/0';const chosen=rows.filter(r=>state[r.audit_id]?.decision).length,revealed=rows.filter(r=>state[r.audit_id]?.revealed).length;document.getElementById('progress').textContent=`已选 ${chosen}/${rows.length} · 已揭示 ${revealed}/${rows.length}`};function choose(id,v){state[id]={...(state[id]||{}),decision:v};save()}function revealReview(id){if(!state[id]?.decision)return;state[id]={...(state[id]||{}),revealed:true};save()}function field(id,k,v){state[id]={...(state[id]||{}),[k]:v};persist()}
document.getElementById('prev').onclick=()=>{if(visible.length){cursor=(cursor-1+visible.length)%visible.length;render()}};document.getElementById('next').onclick=()=>{if(visible.length){cursor=(cursor+1)%visible.length;render()}};document.getElementById('filter').onchange=()=>{cursor=0;render()};document.getElementById('export').onclick=()=>{const annotator=document.getElementById('annotator').value.trim();if(!annotator){alert('请填写 annotator ID');return}const out=[];for(const r of rows){const s=state[r.audit_id]||{};if(!s.decision){alert('仍有未审核条目');return}let spans;try{spans=JSON.parse(s.evidence||'[]')}catch(e){alert(r.audit_id+' evidence 不是合法 JSON');return}if(!Array.isArray(spans)){alert(r.audit_id+' evidence 必须是 array');return}out.push({audit_id:r.audit_id,queue_item_hash:r.queue_item_hash,decision:s.decision,factor_invalid:!!s.factor_invalid,evidence_spans:spans,rationale:(s.rationale||'').trim(),annotator})}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}));a.download=text.export_name;a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''.replace("__PAYLOAD__", payload).replace("__TITLE__", escape(title)).replace("__SEARCHABLE_TEXT__", escape(searchable_text))


def render_grouped_blind_audit_html(
    groups: list[dict[str, Any]],
    references: list[dict[str, Any]],
    *,
    decisions: Iterable[str],
    title: str,
    heading: str,
    instructions: str,
    storage_namespace: str,
    objective_max_chars: int,
    objective_editable: bool = True,
    historical_audit_visible: bool = True,
    item_heading: str = "Path",
    reference_heading: str = "隐藏预审",
    diagnostic_heading: str = "Judge diagnostics",
    reveal_button: str = "揭示预审与旧 path 审计",
    revealed_notice: str = "预审已揭示；查看后仍可修改盲选。",
    progress_item_label: str = "paths",
) -> str:
    """Render a grouped audit where one CoT is shared by several Factors."""

    allowed = tuple(decisions)
    if not allowed or len(allowed) != len(set(allowed)):
        raise ConfigurationError("audit decisions must be unique and non-empty")
    if objective_max_chars <= 0:
        raise ConfigurationError("objective_max_chars must be positive")
    items = [item for group in groups for item in group.get("items", [])]
    validate_reference_reviews(items, references, decisions=allowed)
    group_ids = [group.get("audit_group_id") for group in groups]
    if len(group_ids) != len(set(group_ids)) or None in group_ids:
        raise ConfigurationError("grouped audit contains duplicate or missing group IDs")
    payload = base64.b64encode(
        json.dumps(
            {
                "groups": groups,
                "references": references,
                "decisions": allowed,
                "objective_max_chars": objective_max_chars,
                "options": {
                    "objective_editable": objective_editable,
                    "historical_audit_visible": historical_audit_visible,
                    "item_heading": item_heading,
                    "reference_heading": reference_heading,
                    "diagnostic_heading": diagnostic_heading,
                    "reveal_button": reveal_button,
                    "revealed_notice": revealed_notice,
                    "progress_item_label": progress_item_label,
                },
                "text": {
                    "heading": heading,
                    "instructions": instructions,
                    "storage_namespace": storage_namespace,
                },
            },
            ensure_ascii=False,
            separators=(",", ":"),
        ).encode("utf-8")
    ).decode("ascii")
    return r'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>__TITLE__</title><style>
body{margin:0;background:#f5f2ea;color:#27231d;font:15px/1.5 system-ui,sans-serif}header{position:sticky;top:0;background:#f5f2eaf2;border-bottom:1px solid #d7d0c3;padding:12px 3vw;z-index:2}.bar{display:flex;gap:8px;align-items:center;flex-wrap:wrap}.bar h1{font-size:19px;margin:0 auto 0 0}button,select,input,textarea{font:inherit}button,select,input{padding:7px 10px;border:1px solid #cfc6b7;border-radius:7px;background:white}.active{background:#315d52;color:white}.reveal{background:#70531b;color:white}.notice{padding:10px;border:1px dashed #b99b5a;border-radius:8px;background:#fff8df}.reference{margin-top:10px;border:2px solid #b99b5a;background:#fffaf0}.badge{display:inline-block;padding:2px 7px;border-radius:12px;background:#eee5cf;margin-right:5px}main{max-width:1500px;margin:18px auto;padding:0 3vw 60px}.card,.path{background:#fffdf7;border:1px solid #d7d0c3;border-radius:12px;padding:16px;margin-bottom:14px}.grid{display:grid;grid-template-columns:1fr 1fr;gap:12px}.panel{border:1px solid #ddd4c5;border-radius:8px;padding:10px;min-width:0}pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:55vh;overflow:auto;background:#f8f5ee;padding:10px}textarea{width:100%;min-height:60px}.choices{display:flex;gap:7px;flex-wrap:wrap;margin:10px 0}@media(max-width:850px){.grid{grid-template-columns:1fr}}
</style></head><body><header><div class="bar"><h1 id="heading"></h1><span id="progress"></span><select id="filter"><option value="all">全部</option><option value="unreviewed">未完成</option></select><button id="prev">上一题</button><span id="position"></span><button id="next">下一题</button><input id="annotator" placeholder="annotator ID"><button id="export">导出 owner_overlay.jsonl</button></div><div id="instructions"></div></header><main id="app"></main><script>
const payload=JSON.parse(new TextDecoder().decode(Uint8Array.from(atob('__PAYLOAD__'),c=>c.charCodeAt(0))));const groups=payload.groups,refs=Object.fromEntries(payload.references.map(x=>[x.audit_id,x])),decisions=payload.decisions,text=payload.text,maxObjective=payload.objective_max_chars,options=payload.options;document.getElementById('heading').textContent=text.heading;document.getElementById('instructions').textContent=text.instructions;const key=text.storage_namespace+':'+groups.map(x=>x.queue_item_hash).join(':');let state=JSON.parse(localStorage.getItem(key)||'{}'),cursor=0,visible=[];
const esc=x=>String(x??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));function gs(g){return state[g.audit_group_id]||{items:{}}}function complete(g){const s=gs(g);return g.items.every(i=>s.items?.[i.audit_id]?.decision)}function persist(){localStorage.setItem(key,JSON.stringify(state))}function field(gid,k,v){state[gid]={...(state[gid]||{}),[k]:v,items:{...(state[gid]?.items||{})}};persist()}function itemField(gid,id,k,v){const g=state[gid]||{},items={...(g.items||{})};items[id]={...(items[id]||{}),[k]:v};state[gid]={...g,items};persist()}function choose(gid,id,v){itemField(gid,id,'decision',v);render()}function reveal(gid){const g=groups.find(x=>x.audit_group_id===gid);if(!complete(g))return;field(gid,'revealed',true);render()}function filtered(){return groups.filter(g=>document.getElementById('filter').value==='all'||!complete(g)||!gs(g).revealed)}
function reference(i,s){if(!s.revealed)return '';const r=refs[i.audit_id];return `<section class="panel reference"><b>${esc(options.reference_heading)}</b><p><span class="badge">${esc(r.decision)}</span><span class="badge">${esc(r.confidence)}</span><span class="badge">${esc(r.review_method)}</span></p><p>${esc(r.rationale)}</p><pre>${esc(JSON.stringify(r.evidence_spans||[],null,2))}</pre><details><summary>${esc(options.diagnostic_heading)}</summary><pre>${esc(JSON.stringify(r.judge_diagnostic||{},null,2))}</pre></details></section>`}
function pathCard(g,i,s){const x=s.items?.[i.audit_id]||{};return `<section class="path"><h3>${esc(options.item_heading)} ${esc(i.path_index)} · ${esc(i.audit_id)}</h3><pre>${esc(JSON.stringify(i.factor_spec,null,2))}</pre><div class="choices">${decisions.map(d=>`<button class="${x.decision===d?'active':''}" onclick="choose('${g.audit_group_id}','${i.audit_id}','${d}')">${esc(d)}</button>`).join('')}</div><label><input type="checkbox" ${x.factor_invalid?'checked':''} onchange="itemField('${g.audit_group_id}','${i.audit_id}','factor_invalid',this.checked)"> factor_invalid</label><p>Evidence spans（JSON string array）</p><textarea oninput="itemField('${g.audit_group_id}','${i.audit_id}','evidence',this.value)">${esc(x.evidence||'[]')}</textarea><p>Rationale</p><textarea oninput="itemField('${g.audit_group_id}','${i.audit_id}','rationale',this.value)">${esc(x.rationale||'')}</textarea>${reference(i,s)}</section>`}
function card(g){const s=gs(g),objective=s.objective??g.proposed_objective;const objectivePanel=options.objective_editable?`<h3>L1 task objective（必须为题面原文片段，最多 ${maxObjective} 字符）</h3><textarea oninput="field('${g.audit_group_id}','objective',this.value)">${esc(objective)}</textarea><p>${esc((g.objective_audit_reasons||[]).join(', '))}</p>`:'';const revealPanel=!complete(g)?`<div class="notice">请先完成本题全部 ${esc(options.item_heading)} 的盲选。</div>`:!s.revealed?`<div class="notice">盲选已保存。<button class="reveal" onclick="reveal('${g.audit_group_id}')">${esc(options.reveal_button)}</button></div>`:options.historical_audit_visible?`<div class="notice"><b>旧 path 审计：</b>${esc(JSON.stringify(g.historical_audit))}<p>若最终题级 OR 与旧审计冲突，请填写 override rationale。</p><textarea oninput="field('${g.audit_group_id}','historical_override_rationale',this.value)">${esc(s.historical_override_rationale||'')}</textarea></div>`:`<div class="notice">${esc(options.revealed_notice)}</div>`;return `<article class="card"><h2>${esc(g.audit_group_id)} · ${esc(g.benchmark_source)} · ${esc(g.task_id)}</h2><div class="grid"><section class="panel"><h3>Task Context</h3><pre>${esc(g.task_context)}</pre>${objectivePanel}</section><section class="panel"><h3>Model thinking</h3><pre>${esc(g.model_cot)}</pre></section></div>${revealPanel}${g.items.map(i=>pathCard(g,i,s)).join('')}</article>`}
function render(){visible=filtered();if(cursor>=visible.length)cursor=Math.max(0,visible.length-1);document.getElementById('app').innerHTML=visible.length?card(visible[cursor]):'<p>当前筛选没有条目。</p>';document.getElementById('position').textContent=visible.length?`${cursor+1}/${visible.length}`:'0/0';document.getElementById('progress').textContent=`已完成 ${groups.filter(complete).length}/${groups.length} 题 · ${groups.reduce((n,g)=>n+g.items.filter(i=>gs(g).items?.[i.audit_id]?.decision).length,0)}/${groups.reduce((n,g)=>n+g.items.length,0)} ${options.progress_item_label}`}
document.getElementById('prev').onclick=()=>{if(visible.length){cursor=(cursor-1+visible.length)%visible.length;render()}};document.getElementById('next').onclick=()=>{if(visible.length){cursor=(cursor+1)%visible.length;render()}};document.getElementById('filter').onchange=()=>{cursor=0;render()};document.getElementById('export').onclick=()=>{const annotator=document.getElementById('annotator').value.trim();if(!annotator){alert('请填写 annotator ID');return}const out=[];for(const g of groups){const s=gs(g);if(!complete(g)){alert(g.audit_group_id+' 尚未完成');return}let objective='';if(options.objective_editable){objective=(s.objective??g.proposed_objective).trim();if(!objective||objective.length>maxObjective||!g.normalized_task_context.includes(objective)){alert(g.audit_group_id+' objective 必须是题面原文片段且不超过限制');return}}for(const i of g.items){const x=s.items[i.audit_id];let spans;try{spans=JSON.parse(x.evidence||'[]')}catch(e){alert(i.audit_id+' evidence 不是合法 JSON');return}if(!Array.isArray(spans)){alert(i.audit_id+' evidence 必须是 array');return}const row={audit_id:i.audit_id,queue_item_hash:i.queue_item_hash,audit_group_id:g.audit_group_id,decision:x.decision,factor_invalid:!!x.factor_invalid,evidence_spans:spans,rationale:(x.rationale||'').trim(),annotator};if(options.objective_editable){row.task_objective=objective;row.historical_override_rationale=(s.historical_override_rationale||'').trim()}out.push(row)}}const a=document.createElement('a');a.href=URL.createObjectURL(new Blob([out.map(x=>JSON.stringify(x)).join('\n')+'\n'],{type:'application/jsonl'}));a.download='owner_overlay.jsonl';a.click();URL.revokeObjectURL(a.href)};render();
</script></body></html>'''.replace("__PAYLOAD__", payload).replace("__TITLE__", escape(title))


def load_owner_overlay(
    path: Path,
    queue: list[dict[str, Any]],
    *,
    decisions: Iterable[str],
) -> dict[str, dict[str, Any]]:
    """Load a complete or partial owner overlay and reject stale evidence."""

    if not path.is_file():
        return {}
    allowed = set(decisions)
    expected = {row.get("audit_id"): row for row in queue}
    result: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(path):
        audit_id = row.get("audit_id")
        if audit_id not in expected or audit_id in result:
            raise ConfigurationError(
                f"owner overlay has unknown or duplicate audit_id: {audit_id}"
            )
        if row.get("queue_item_hash") != expected[audit_id].get("queue_item_hash"):
            raise ConfigurationError(f"owner overlay hash mismatch for {audit_id}")
        if row.get("decision") not in allowed or not isinstance(
            row.get("factor_invalid"), bool
        ):
            raise ConfigurationError(
                f"owner overlay has invalid decision/factor_invalid for {audit_id}"
            )
        spans = row.get("evidence_spans", [])
        cot = expected[audit_id].get("model_cot")
        if not isinstance(spans, list) or any(
            not isinstance(span, str) or not isinstance(cot, str) or span not in cot
            for span in spans
        ):
            raise ConfigurationError(
                f"owner overlay evidence must be exact model-COT spans for {audit_id}"
            )
        result[str(audit_id)] = row
    return result
