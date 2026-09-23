#!/usr/bin/env python3
"""Build the public evaluation-data release.

Training data is not included. Every core and supplemental evaluation dataset
is included:

- redistributable benchmarks are copied with their task text and CoT;
- copyrighted task statements are replaced by ``[PROBLEM]``;
- ``problem_links.jsonl`` stores the official URL, source or task id, statement
  hash, and reversible slice offsets.

AIME statements come from the local OpenAI environment files. Those files are
read only to recover ``url`` and ``source``; their problem text is never
written to the public tree. MASK, SkillEval, and SHADE-Arena use the same
placeholder. Their statement is the trajectory ``task_context``, and the link
is the official dataset plus ``task_id``.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = (
    REPO_ROOT
    / "release/iclr2027_anonymous_source/review_required/data/evaluation/core"
)
DEFAULT_OUTPUT = REPO_ROOT / "release/iclr2027_eval_public"
EVAL_ROOT = REPO_ROOT.parent / "eval"
PROBLEM_TOKEN = "[PROBLEM]"
MIN_PROBLEM_SLICE = 64

PUBLIC_DATASETS = {
    "process_gpqa": {
        "license": "CC-BY-4.0 data; OpenAI evaluation code Apache-2.0",
        "attribution": "https://huggingface.co/datasets/Idavidrein/gpqa",
        "note": "Problem text is included, matching the OpenAI monitorability release.",
    },
    "sycobench_judge": {
        "license": "CC-BY-4.0",
        "attribution": "SycoBench-600",
        "note": "Task text, CoT, factors, and labels are included.",
    },
    "ifbench": {
        "license": "ODC-BY-1.0",
        "attribution": "https://github.com/allenai/IFBench",
        "note": "Attribute AllenAI and note that some outputs come from third-party models.",
    },
    "hallulens": {
        "license": "CC-BY-NC",
        "attribution": "https://github.com/facebookresearch/HalluLens",
        "note": "Non-commercial slice. Do not relicense as Apache-2.0.",
    },
    "confaide": {
        "license": "MIT",
        "attribution": "https://github.com/skywalker023/confaide",
        "note": "Task text, CoT, factors, and labels are included.",
    },
    "jailbreakbench": {
        "license": "MIT",
        "attribution": "https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors",
        "note": "Task text, CoT, factors, and labels are included.",
    },
    "openai_wmdp": {
        "license": "OpenAI evaluation code Apache-2.0; WMDP MIT",
        "attribution": "https://github.com/centerforaisafety/wmdp",
        "note": "Task text, CoT, factors, labels, and scoring plans are included.",
    },
    "reward_hacking": {
        "license": "MIT",
        "attribution": "https://huggingface.co/datasets/metr-evals/malt-public",
        "note": "Task text, CoT, factors, labels, and complete_eval are included. Upstream hosting is gated; the license is MIT.",
    },
    "sandbagging": {
        "license": "MIT",
        "attribution": "https://huggingface.co/datasets/metr-evals/malt-public",
        "note": "Task text, CoT, factors, and labels are included. Upstream hosting is gated; the license is MIT.",
    },
}

TASK_STATEMENT_DATASETS = {
    "mask_qwen": {
        "license": "no formal dataset license; task statements are not redistributed",
        "attribution": "https://huggingface.co/datasets/cais/MASK",
        "note": "The task statement is replaced with [PROBLEM]. CoT, factors, and labels are included.",
    },
    "skilleval": {
        "license": "no umbrella license; task statements are not redistributed",
        "attribution": "https://huggingface.co/datasets/Artemis0430/skilleval-v1",
        "note": "The task statement is replaced with [PROBLEM]. CoT, factors, and labels are included.",
    },
    "shadearena": {
        "license": "public split MIT; this snapshot is not limited to that split",
        "attribution": "https://github.com/jkutaso/SHADE-Arena",
        "note": "The task statement is replaced with [PROBLEM]. CoT, factors, and labels are included.",
    },
}

SUPPLEMENTAL_PUBLIC_BASE = {
    "glm53_ifbench": "ifbench",
    "kimi_k26_ifbench": "ifbench",
    "glm53_process_gpqa": "process_gpqa",
    "kimi_k26_process_gpqa": "process_gpqa",
    "glm53_sycobench": "sycobench_judge",
    "kimi_k26_sycobench": "sycobench_judge",
}

AIME_LICENSE = {
    "license": "AIME problem statements are not redistributed",
    "attribution": "Official url and source in problem_links.jsonl",
    "note": "Contest statements are replaced with [PROBLEM]. CoT, factors, and labels are included.",
}

SYCOPHANCY_LICENSE = {
    "license": "AIME statements withheld; DAFT, Scruples, and GPQA text included",
    "attribution": "AIME url and source in problem_links.jsonl; other rows follow the OpenAI monitorability release",
    "note": "Only rows whose query source is aime replace the contest statement with [PROBLEM].",
}

AIME_CATALOGS = {
    "process": EVAL_ROOT / "environments/process/process_aime_math.jsonl",
    "intervention": EVAL_ROOT / "environments/intervention/aime_with_problem.jsonl",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return sha256_bytes(text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")))
            handle.write("\n")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_catalog(path: Path) -> list[dict[str, Any]]:
    catalog = []
    for row in read_jsonl(path):
        problem = row.get("problem")
        if not isinstance(problem, str) or not problem:
            continue
        catalog.append(
            {
                "problem": problem,
                "url": row.get("url"),
                "source": row.get("source"),
                "problem_sha256": sha256_text(problem),
            }
        )
    if not catalog:
        raise RuntimeError(f"{path} contains no AIME problem statements")
    return catalog


def match_problem(text: str, catalog: list[dict[str, Any]]) -> dict[str, Any]:
    hits = [item for item in catalog if item["problem"] in text]
    if not hits:
        raise RuntimeError("no AIME problem statement matches this row")
    longest = max(len(item["problem"]) for item in hits)
    longest_hits = [item for item in hits if len(item["problem"]) == longest]
    identities = {(item["url"], item["problem_sha256"]) for item in longest_hits}
    if len(identities) != 1:
        raise RuntimeError("AIME problem match is ambiguous")
    return longest_hits[0]


_ROLL_BASE = 1315423911
_ROLL_MOD = 1 << 64


def _roll_table(problem: str) -> dict[int, list[int]]:
    width = MIN_PROBLEM_SLICE
    if len(problem) < width:
        return {}
    value = 0
    for char in problem[:width]:
        value = (value * _ROLL_BASE + ord(char)) % _ROLL_MOD
    table: dict[int, list[int]] = {value: [0]}
    power = pow(_ROLL_BASE, width - 1, _ROLL_MOD)
    for index in range(width, len(problem)):
        value = (value - ord(problem[index - width]) * power) % _ROLL_MOD
        value = (value * _ROLL_BASE + ord(problem[index])) % _ROLL_MOD
        table.setdefault(value, []).append(index - width + 1)
    return table


def _advance_hash(value: int, previous: str, incoming: str, power: int) -> int:
    value = (value - ord(previous) * power) % _ROLL_MOD
    return (value * _ROLL_BASE + ord(incoming)) % _ROLL_MOD


def problem_slices(
    text: str,
    problem: str,
    table: dict[int, list[int]] | None = None,
) -> list[dict[str, int]]:
    width = MIN_PROBLEM_SLICE
    if len(problem) < width or len(text) < width:
        return []
    if table is None:
        table = _roll_table(problem)
    slices: list[dict[str, int]] = []
    value = 0
    for char in text[:width]:
        value = (value * _ROLL_BASE + ord(char)) % _ROLL_MOD
    power = pow(_ROLL_BASE, width - 1, _ROLL_MOD)
    index = 0
    last = len(text) - width
    while index <= last:
        matched: int | None = None
        window = text[index : index + width]
        for start in table.get(value, ()):
            if window == problem[start : start + width]:
                matched = start
                break
        if matched is None:
            if index == last:
                break
            value = _advance_hash(value, text[index], text[index + width], power)
            index += 1
            continue
        text_end = index + width
        problem_end = matched + width
        while (
            text_end < len(text)
            and problem_end < len(problem)
            and text[text_end] == problem[problem_end]
        ):
            text_end += 1
            problem_end += 1
        slices.append({"start": matched, "end": problem_end})
        if text_end > last:
            break
        while index < text_end:
            value = _advance_hash(value, text[index], text[index + width], power)
            index += 1
    return slices


def _self_check_slices() -> None:
    problem = "abcdefghijklmnopqrstuvwxyz" * 4
    samples = [
        "xx" + problem[10:90] + "yy" + problem + "zz",
        problem + "||" + problem[20:100],
        "no shared long fragment here",
        problem[:80] + "QQ" + problem[30:],
    ]

    def reference(text: str) -> list[dict[str, int]]:
        found: list[dict[str, int]] = []
        cursor = 0
        while cursor < len(text):
            window = text[cursor : cursor + MIN_PROBLEM_SLICE]
            if len(window) < MIN_PROBLEM_SLICE:
                break
            start = problem.find(window)
            if start < 0:
                cursor += 1
                continue
            text_end = cursor + MIN_PROBLEM_SLICE
            problem_end = start + MIN_PROBLEM_SLICE
            while (
                text_end < len(text)
                and problem_end < len(problem)
                and text[text_end] == problem[problem_end]
            ):
                text_end += 1
                problem_end += 1
            found.append({"start": start, "end": problem_end})
            cursor = text_end
        return found

    for sample in samples:
        if problem_slices(sample, problem) != reference(sample):
            raise RuntimeError("problem-slice matcher failed its self-check")


def apply_slices(text: str, problem: str, slices: list[dict[str, int]]) -> str:
    if not slices:
        return text
    pieces: list[str] = []
    cursor = 0
    for item in slices:
        excerpt = problem[item["start"] : item["end"]]
        found = text.find(excerpt, cursor)
        if found < 0:
            raise RuntimeError("problem slice disappeared while redacting")
        pieces.append(text[cursor:found])
        pieces.append(PROBLEM_TOKEN)
        cursor = found + len(excerpt)
    pieces.append(text[cursor:])
    return "".join(pieces)


def redact_text(
    text: str,
    problem: str,
    table: dict[int, list[int]],
) -> tuple[str, list[dict[str, int]]]:
    if PROBLEM_TOKEN in text:
        raise RuntimeError("row already contains the problem placeholder")
    slices = problem_slices(text, problem, table)
    return apply_slices(text, problem, slices), slices


def restore_text(text: str, problem: str, slices: list[dict[str, int]]) -> str:
    pieces = text.split(PROBLEM_TOKEN)
    if len(pieces) != len(slices) + 1:
        raise RuntimeError("placeholder count does not match the redaction record")
    restored = [pieces[0]]
    for piece, item in zip(pieces[1:], slices):
        restored.append(problem[item["start"] : item["end"]])
        restored.append(piece)
    return "".join(restored)


def redact_strings(
    value: Any,
    problem: str,
    path: str,
    records: list[dict[str, Any]],
    table: dict[int, list[int]],
) -> Any:
    if isinstance(value, str):
        redacted, slices = redact_text(value, problem, table)
        if slices:
            records.append({"path": path, "slices": slices})
        return redacted
    if isinstance(value, list):
        return [
            redact_strings(item, problem, f"{path}[{index}]", records, table)
            for index, item in enumerate(value)
        ]
    if isinstance(value, dict):
        return {
            key: redact_strings(item, problem, f"{path}.{key}", records, table)
            for key, item in value.items()
        }
    return value


def copy_tree(source: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(source, destination)


def replace_hashes(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, dict):
        return {key: replace_hashes(item, mapping) for key, item in value.items()}
    if isinstance(value, list):
        return [replace_hashes(item, mapping) for item in value]
    if isinstance(value, str) and value in mapping:
        return mapping[value]
    return value


def refresh_hashes(source_dir: Path, destination: Path, relative_paths: list[str]) -> None:
    mapping = {
        sha256_file(source_dir / relative): sha256_file(destination / relative)
        for relative in relative_paths
    }
    for path in destination.rglob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        write_json(path, replace_hashes(payload, mapping))


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for line in path.open(encoding="utf-8") if line.strip())


def dataset_counts(directory: Path) -> dict[str, int]:
    counts = {
        "trajectories": count_jsonl(directory / "trajectories.jsonl"),
        "queries": count_jsonl(directory / "eval_queries.jsonl"),
    }
    nested = {
        str(path.relative_to(directory)): count_jsonl(path)
        for path in sorted(directory.rglob("*.jsonl"))
        if path.name in {"trajectories.jsonl", "eval_queries.jsonl"}
        and path.parent != directory
    }
    if nested:
        counts["nested"] = nested
    return counts


def copy_missing_files(source_dir: Path, destination: Path) -> None:
    for path in source_dir.rglob("*"):
        if not path.is_file():
            continue
        target = destination / path.relative_to(source_dir)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)


def public_spec(dataset_id: str, group: str | None = None) -> dict[str, Any]:
    base_id = SUPPLEMENTAL_PUBLIC_BASE.get(dataset_id, dataset_id)
    if base_id not in PUBLIC_DATASETS:
        raise RuntimeError(f"no redistribution license for {dataset_id}")
    spec = dict(PUBLIC_DATASETS[base_id])
    if group:
        spec["note"] = f"{spec['note']} Supplemental split: {group}."
    return spec


def task_spec(dataset_id: str, group: str | None = None) -> dict[str, Any]:
    if "skilleval" in dataset_id:
        key = "skilleval"
    elif "mask" in dataset_id:
        key = "mask_qwen"
    elif "shade" in dataset_id:
        key = "shadearena"
    else:
        raise RuntimeError(f"no task-statement policy for {dataset_id}")
    spec = dict(TASK_STATEMENT_DATASETS[key])
    if group:
        spec["note"] = f"{spec['note']} Supplemental split: {group}."
    return spec


def prepare_public_dataset(
    source_dir: Path,
    destination: Path,
    spec: dict[str, Any],
) -> dict[str, Any]:
    copy_tree(source_dir, destination)
    write_json(destination / "LICENSE.json", spec)
    report = {"license": spec["license"]}
    report.update(dataset_counts(destination))
    return report


def prepare_task_statement_dataset(
    source_dir: Path,
    destination: Path,
    spec: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    protected: list[str] = []
    rewritten: list[str] = []
    link_count = 0
    for trajectory_path in sorted(source_dir.rglob("trajectories.jsonl")):
        relative_dir = trajectory_path.parent.relative_to(source_dir)
        trajectories = read_jsonl(trajectory_path)
        query_path = trajectory_path.with_name("eval_queries.jsonl")
        queries = read_jsonl(query_path) if query_path.exists() else None
        problem_by_trajectory: dict[str, dict[str, Any]] = {}
        redacted_trajectories: list[dict[str, Any]] = []
        trajectory_links: list[dict[str, Any]] = []
        for row in trajectories:
            statement = row.get("task_context")
            if not isinstance(statement, str) or len(statement) < MIN_PROBLEM_SLICE:
                raise RuntimeError(f"task statement is missing or too short: {row.get('trajectory_id')}")
            record = {
                "problem": statement,
                "url": spec["attribution"],
                "source": row.get("task_id"),
                "problem_sha256": sha256_text(statement),
            }
            problem_by_trajectory[row["trajectory_id"]] = record
            protected.append(statement)
            redacted, link = redact_known_problem(row, record, row["trajectory_id"])
            if link is None or statement_in_row(redacted, statement):
                raise RuntimeError(f"task statement survived redaction: {row['trajectory_id']}")
            redacted_trajectories.append(redacted)
            trajectory_links.append(link)
        verify_restoration(
            trajectories,
            redacted_trajectories,
            trajectory_links,
            list(problem_by_trajectory.values()),
            "trajectory_id",
        )
        query_links: list[dict[str, Any]] = []
        redacted_queries: list[dict[str, Any]] | None = None
        if queries is not None:
            redacted_queries = []
            for row in queries:
                record = problem_by_trajectory[row["trajectory_id"]]
                redacted, link = redact_known_problem(row, record, row["query_id"])
                if statement_in_row(redacted, record["problem"]):
                    raise RuntimeError(f"task statement survived query redaction: {row['query_id']}")
                redacted_queries.append(redacted)
                if link is not None:
                    query_links.append(link)
            verify_restoration(
                queries,
                redacted_queries,
                query_links,
                list(problem_by_trajectory.values()),
                "query_id",
            )
        target = destination / relative_dir
        target.mkdir(parents=True, exist_ok=True)
        write_jsonl(target / "trajectories.jsonl", redacted_trajectories)
        rewritten.append(str(relative_dir / "trajectories.jsonl") if relative_dir.parts else "trajectories.jsonl")
        if redacted_queries is not None:
            write_jsonl(target / "eval_queries.jsonl", redacted_queries)
            rewritten.append(str(relative_dir / "eval_queries.jsonl") if relative_dir.parts else "eval_queries.jsonl")
        links = trajectory_links + query_links
        write_jsonl(target / "problem_links.jsonl", links)
        link_count += len(links)
    copy_missing_files(source_dir, destination)
    refresh_hashes(source_dir, destination, rewritten)
    write_json(destination / "LICENSE.json", spec)
    report = {"license": spec["license"], "redaction_records": link_count}
    report.update(dataset_counts(destination))
    return report, protected


def statement_in_row(value: Any, statement: str) -> bool:
    if isinstance(value, str):
        return statement in value
    if isinstance(value, list):
        return any(statement_in_row(item, statement) for item in value)
    if isinstance(value, dict):
        return any(statement_in_row(item, statement) for item in value.values())
    return False


def aime_trajectory_ids(query_path: Path) -> set[str]:
    return {
        row["trajectory_id"]
        for row in read_jsonl(query_path)
        if row.get("source") == "aime"
    }


def verify_restoration(
    original_rows: list[dict[str, Any]],
    redacted_rows: list[dict[str, Any]],
    links: list[dict[str, Any]],
    catalog: list[dict[str, Any]],
    id_field: str,
) -> None:
    by_hash = {item["problem_sha256"]: item["problem"] for item in catalog}
    link_by_id = {item["row_id"]: item for item in links}
    for original, redacted in zip(original_rows, redacted_rows):
        row_id = original[id_field]
        link = link_by_id.get(row_id)
        if link is None:
            if original != redacted:
                raise RuntimeError(f"unredacted row changed: {row_id}")
            continue
        problem = by_hash[link["problem_sha256"]]
        restored = copy.deepcopy(redacted)
        for record in link["redactions"]:
            restore_path(restored, record["path"], problem, record["slices"])
        if restored != original:
            raise RuntimeError(f"AIME restoration mismatch: {row_id}")


def descend(value: Any, part: str) -> Any:
    if "[" not in part:
        return value[part]
    key, index_text = part[:-1].split("[")
    return value[key][int(index_text)]


def restore_path(value: Any, path: str, problem: str, slices: list[dict[str, int]]) -> None:
    parts = path.removeprefix("$.").split(".")
    parent = value
    for part in parts[:-1]:
        parent = descend(parent, part)
    last = parts[-1]
    if "[" in last:
        key, index_text = last[:-1].split("[")
        index = int(index_text)
        parent[key][index] = restore_text(parent[key][index], problem, slices)
        return
    parent[last] = restore_text(parent[last], problem, slices)


def redact_known_problem(
    row: dict[str, Any],
    problem_record: dict[str, Any],
    row_id: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    records: list[dict[str, Any]] = []
    problem = problem_record["problem"]
    redacted = redact_strings(row, problem, "$", records, _roll_table(problem))
    if not records:
        return row, None
    return redacted, {
        "row_id": row_id,
        "task_id": row.get("task_id"),
        "url": problem_record["url"],
        "source": problem_record["source"],
        "problem_sha256": problem_record["problem_sha256"],
        "redactions": records,
    }


def write_redacted_dataset(
    source_dir: Path,
    destination: Path,
    filenames: list[str],
    redacted_by_file: dict[str, list[dict[str, Any]]],
    links: list[dict[str, Any]],
) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for filename in filenames:
        write_jsonl(destination / filename, redacted_by_file[filename])
    write_jsonl(destination / "problem_links.jsonl", links)
    for name in ("summary.json", "effective_config.json", "plan_manifest.json"):
        source = source_dir / name
        if source.exists():
            shutil.copy2(source, destination / name)
    plans = source_dir / "plans"
    if plans.is_dir():
        copy_tree(plans, destination / "plans")
    refresh_hashes(source_dir, destination, filenames)


def prepare_one_catalog_dataset(
    source_dir: Path,
    destination: Path,
    catalog: list[dict[str, Any]],
    spec: dict[str, Any],
    trajectory_filter: set[str] | None = None,
) -> dict[str, Any]:
    trajectories = read_jsonl(source_dir / "trajectories.jsonl")
    queries = read_jsonl(source_dir / "eval_queries.jsonl")
    selected = (
        {row["trajectory_id"] for row in trajectories}
        if trajectory_filter is None
        else trajectory_filter
    )
    problem_by_trajectory: dict[str, dict[str, Any]] = {}
    redacted_trajectories: list[dict[str, Any]] = []
    trajectory_links: list[dict[str, Any]] = []
    for row in trajectories:
        if row["trajectory_id"] not in selected:
            redacted_trajectories.append(row)
            continue
        matched = match_problem(row["task_context"], catalog)
        problem_by_trajectory[row["trajectory_id"]] = matched
        redacted, link = redact_known_problem(row, matched, row["trajectory_id"])
        if link is None or matched["problem"] in redacted["task_context"]:
            raise RuntimeError(f"problem statement survived redaction: {row['trajectory_id']}")
        redacted_trajectories.append(redacted)
        trajectory_links.append(link)
    verify_restoration(trajectories, redacted_trajectories, trajectory_links, catalog, "trajectory_id")

    redacted_queries: list[dict[str, Any]] = []
    query_links: list[dict[str, Any]] = []
    for row in queries:
        matched = problem_by_trajectory.get(row["trajectory_id"])
        if matched is None:
            redacted_queries.append(row)
            continue
        redacted, link = redact_known_problem(row, matched, row["query_id"])
        if statement_in_row(redacted, matched["problem"]):
            raise RuntimeError(f"problem statement survived query redaction: {row['query_id']}")
        redacted_queries.append(redacted)
        if link is not None:
            query_links.append(link)
    verify_restoration(queries, redacted_queries, query_links, catalog, "query_id")
    links = trajectory_links + query_links
    write_redacted_dataset(
        source_dir,
        destination,
        ["trajectories.jsonl", "eval_queries.jsonl"],
        {
            "trajectories.jsonl": redacted_trajectories,
            "eval_queries.jsonl": redacted_queries,
        },
        links,
    )
    copy_missing_files(source_dir, destination)
    refresh_hashes(source_dir, destination, ["trajectories.jsonl", "eval_queries.jsonl"])
    write_json(destination / "LICENSE.json", spec)
    report = {
        "license": spec["license"],
        "selected_trajectories": len(selected),
        "redaction_records": len(links),
    }
    report.update(dataset_counts(destination))
    return report


def assert_problems_absent(output: Path, problems: list[str]) -> None:
    unique = list(dict.fromkeys(problem for problem in problems if problem))
    for path in output.rglob("*"):
        if not path.is_file() or path.suffix not in {".json", ".jsonl", ".md"}:
            continue
        if path.name == "problem_links.jsonl":
            continue
        text = path.read_text(encoding="utf-8")
        for problem in unique:
            if problem in text:
                raise RuntimeError(f"protected statement remains in {path}")


def write_readme(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "README.md").write_text(
        """---
license: other
pretty_name: Factor Monitor Evaluation
---

# Public Factor Monitor evaluation data

Training rows are intentionally absent. Core and supplemental evaluation
artifacts are included.

## Layout

- `public/`: benchmarks whose license allows the task text to be redistributed.
  HalluLens remains CC-BY-NC and is not covered by Apache-2.0. MALT reward
  hacking and sandbagging are included in full under MIT.
- `problem_linked/`: copyrighted task statements are replaced with `[PROBLEM]`.
  CoT, factors, and labels remain. `problem_links.jsonl` stores the official
  URL, source or task id, statement hash, and reversible slice offsets.
- `public/supplemental/` and `problem_linked/supplemental/`: the same split for
  CoT-transform panels and the GLM / Kimi evaluation panels.

OpenAI sycophancy keeps DAFT, Scruples, and GPQA text. Only its AIME rows are
linked. MASK, SkillEval, and SHADE-Arena link the whole task statement. AIME
links use the official per-problem `url` and `source`.

## Restore linked statements

Fetch the statements yourself, then run:

```bash
python scripts/prepare_public_eval_release.py hydrate \\
  --public-dir release/iclr2027_eval_public \\
  --problems /path/to/locally-fetched-statements.jsonl \\
  --output /path/to/hydrated-eval
```

The input JSONL must contain `problem`, `url`, and `source`. Hydration restores
every stored hash present in that file and leaves the other placeholders in
place. A supplied statement whose hash does not match is rejected.

GPQA text is present because the OpenAI monitorability release publishes it.
AIME text is absent because that same release publishes only `url` and `source`.
""",
        encoding="utf-8",
    )


def export_dataset(
    dataset_id: str,
    source_dir: Path,
    output: Path,
    group: str | None,
    catalogs: dict[str, list[dict[str, Any]]],
    protected: list[str],
) -> tuple[str, dict[str, Any]]:
    print(f"export {group or 'core'}/{dataset_id}", flush=True)
    if dataset_id in PUBLIC_DATASETS or dataset_id in SUPPLEMENTAL_PUBLIC_BASE:
        bucket = "public"
        destination = output / bucket
        if group:
            destination = destination / "supplemental" / group
        report = prepare_public_dataset(
            source_dir,
            destination / dataset_id,
            public_spec(dataset_id, group),
        )
        return bucket, report
    if dataset_id == "process_aime" or dataset_id.endswith("process_aime"):
        bucket = "problem_linked"
        destination = output / bucket
        if group:
            destination = destination / "supplemental" / group
        report = prepare_one_catalog_dataset(
            source_dir,
            destination / dataset_id,
            catalogs["process"],
            AIME_LICENSE,
        )
        return bucket, report
    if dataset_id == "openai_sycophancy":
        report = prepare_one_catalog_dataset(
            source_dir,
            output / "problem_linked" / dataset_id,
            catalogs["intervention"],
            SYCOPHANCY_LICENSE,
            aime_trajectory_ids(source_dir / "eval_queries.jsonl"),
        )
        return "problem_linked", report
    if any(token in dataset_id for token in ("mask", "skilleval", "shade")):
        bucket = "problem_linked"
        destination = output / bucket
        if group:
            destination = destination / "supplemental" / group
        report, statements = prepare_task_statement_dataset(
            source_dir,
            destination / dataset_id,
            task_spec(dataset_id, group),
        )
        protected.extend(statements)
        return bucket, report
    raise RuntimeError(f"unclassified evaluation dataset: {dataset_id}")


def copy_group_metadata(group_dir: Path, output: Path) -> None:
    for name in ("README.md", "snapshot_manifest.json", "summary.json"):
        source = group_dir / name
        if not source.exists():
            continue
        for bucket in ("public", "problem_linked"):
            destination = output / bucket / "supplemental" / group_dir.name
            if not destination.exists():
                continue
            shutil.copy2(source, destination / name)


def build(source_root: Path, output: Path) -> dict[str, Any]:
    _self_check_slices()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True)
    catalogs = {name: load_catalog(path) for name, path in AIME_CATALOGS.items()}
    protected = [item["problem"] for catalog in catalogs.values() for item in catalog]
    report: dict[str, Any] = {
        "source": str(source_root),
        "training_data": "excluded",
        "public": {},
        "problem_linked": {},
    }
    for dataset_id in sorted(path.name for path in source_root.iterdir() if path.is_dir()):
        bucket, dataset_report = export_dataset(
            dataset_id,
            source_root / dataset_id,
            output,
            None,
            catalogs,
            protected,
        )
        report[bucket][dataset_id] = dataset_report
    supplemental = source_root.parent / "supplemental"
    if not supplemental.is_dir():
        raise RuntimeError(f"missing supplemental evaluation data: {supplemental}")
    report["public"]["supplemental"] = {}
    report["problem_linked"]["supplemental"] = {}
    for group_dir in sorted(path for path in supplemental.iterdir() if path.is_dir()):
        for dataset_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            bucket, dataset_report = export_dataset(
                dataset_dir.name,
                dataset_dir,
                output,
                group_dir.name,
                catalogs,
                protected,
            )
            report[bucket]["supplemental"].setdefault(group_dir.name, {})[dataset_dir.name] = dataset_report
        copy_group_metadata(group_dir, output)
    assert_problems_absent(output, protected)
    write_readme(output)
    write_json(output / "release_report.json", report)
    return report


def hydrate(public_dir: Path, problems_path: Path, output: Path) -> None:
    catalog = load_catalog(problems_path)
    by_hash = {item["problem_sha256"]: item["problem"] for item in catalog}
    if output.exists():
        shutil.rmtree(output)
    shutil.copytree(public_dir, output)
    restored = 0
    skipped = 0
    for link_path in output.rglob("problem_links.jsonl"):
        links = read_jsonl(link_path)
        dataset_dir = link_path.parent
        mutable: dict[str, list[dict[str, Any]]] = {}
        indexes: dict[str, dict[str, dict[str, Any]]] = {}
        for filename, id_field in (
            ("trajectories.jsonl", "trajectory_id"),
            ("eval_queries.jsonl", "query_id"),
        ):
            path = dataset_dir / filename
            if not path.exists():
                continue
            rows = read_jsonl(path)
            mutable[filename] = rows
            indexes[filename] = {row[id_field]: row for row in rows}
        for link in links:
            problem = by_hash.get(link["problem_sha256"])
            if problem is None:
                skipped += 1
                continue
            if sha256_text(problem) != link["problem_sha256"]:
                raise RuntimeError(f"fetched problem hash mismatch for {link['url']}")
            restored_here = False
            for rows_by_id in indexes.values():
                row = rows_by_id.get(link["row_id"])
                if row is None:
                    continue
                for record in link["redactions"]:
                    restore_path(row, record["path"], problem, record["slices"])
                restored_here = True
            if not restored_here:
                raise RuntimeError(f"redaction record has no row: {link['row_id']}")
            restored += 1
        for filename, rows in mutable.items():
            write_jsonl(dataset_dir / filename, rows)
    print(f"restored {restored} linked rows; left {skipped} placeholders without a supplied statement")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    build_parser = subparsers.add_parser("build")
    build_parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    build_parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    hydrate_parser = subparsers.add_parser("hydrate")
    hydrate_parser.add_argument("--public-dir", type=Path, default=DEFAULT_OUTPUT)
    hydrate_parser.add_argument("--problems", type=Path, required=True)
    hydrate_parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "build":
        report = build(args.source, args.output)
        print(json.dumps(report, indent=2, ensure_ascii=False))
        return
    hydrate(args.public_dir, args.problems, args.output)
    print(f"hydrated evaluation data: {args.output}")


if __name__ == "__main__":
    main()
