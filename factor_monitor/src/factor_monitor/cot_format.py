"""Canonical assistant-private-reasoning formatting for monitor inputs."""

from __future__ import annotations

from collections.abc import Iterable


ASSISTANT_STEP_PREFIX = "[ASSISTANT_STEP_"


def render_assistant_steps(steps: Iterable[str], *, start: int = 1) -> str:
    """Render non-empty private-reasoning blocks with stable step boundaries."""

    if isinstance(start, bool) or not isinstance(start, int) or start < 0:
        raise ValueError("start must be a non-negative integer")
    blocks: list[str] = []
    for step in steps:
        if not isinstance(step, str):
            raise TypeError("assistant reasoning steps must be strings")
        text = step.strip()
        if not text:
            continue
        blocks.append(f"[ASSISTANT_STEP_{start + len(blocks):03d}]\n{text}")
    return "\n\n".join(blocks)


def strip_reasoning_tag(line: str) -> str | None:
    """Return an ATIF reasoning line without its source-specific marker."""

    if not isinstance(line, str):
        return None
    text = line.strip()
    if not text.startswith("[REASONING]"):
        return None
    value = text[len("[REASONING]") :].strip()
    return value or None


def render_atif_reasoning_steps(step_lines: Iterable[Iterable[str]]) -> str:
    """Render ATIF step groups while excluding serialized tool calls."""

    steps: list[str] = []
    for lines in step_lines:
        reasoning = [value for line in lines if (value := strip_reasoning_tag(line))]
        if reasoning:
            steps.append("\n".join(reasoning))
    return render_assistant_steps(steps)


def render_tagged_reasoning_lines(model_cot: str) -> str:
    """Convert legacy flat ``[REASONING]``/``[TOOL_CALL]`` text to CoT-only."""

    if not isinstance(model_cot, str):
        raise TypeError("model_cot must be a string")
    steps: list[str] = []
    current: list[str] = []
    for raw_line in model_cot.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith("[TOOL_CALL]"):
            if current:
                steps.append("\n".join(current))
                current = []
            continue
        value = strip_reasoning_tag(line)
        if value is not None:
            current.append(value)
        elif not line.startswith("[TOOL_USE]"):
            current.append(line)
    if current:
        steps.append("\n".join(current))
    return render_assistant_steps(steps)


def normalize_assistant_step_markers(model_cot: str) -> str:
    """Renumber an already step-marked CoT without nesting old markers."""
    if not isinstance(model_cot, str):
        raise TypeError("model_cot must be a string")
    lines = model_cot.splitlines()
    blocks: list[str] = []
    current: list[str] = []
    for raw in lines:
        line = raw.strip()
        if line.startswith(ASSISTANT_STEP_PREFIX) and line.endswith("]"):
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        if line:
            current.append(line)
    if current:
        blocks.append("\n".join(current))
    return render_assistant_steps(blocks)


def normalize_unmarked_blocks(model_cot: str) -> str:
    """Mark paragraph-separated multi-step CoT, leaving one block canonical."""
    if not isinstance(model_cot, str):
        raise TypeError("model_cot must be a string")
    if "[ASSISTANT_STEP_" in model_cot:
        return normalize_assistant_step_markers(model_cot)
    return render_assistant_steps(model_cot.split("\n\n"))
