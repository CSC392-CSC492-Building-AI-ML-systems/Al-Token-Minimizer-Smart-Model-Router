"""Ordered compression stages, optional presets, and context rendering.

Each module in compressor/stages is automatically imported and registers its
functions with @stage. Requests may select an ordered stage list; presets are
convenient aliases for common combinations. Labels such as "code+concise" are
stored in telemetry, while the versioned fingerprint identifies cache behavior.
"""

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of a document from the knowledge base."""

    text: str
    source: str = ""


@dataclass(frozen=True)
class Prompt:
    """User text, optional system instructions, and knowledge-base chunks."""

    text: str
    system: str | None = None
    context: tuple[Chunk, ...] = ()

    def render(self):
        """Render context chunks before the question, preserving source labels."""
        if not self.context:
            return self.text
        chunks = "\n\n".join(f"[{c.source}]\n{c.text}" if c.source else c.text for c in self.context)
        return f"Context:\n{chunks}\n\nQuestion:\n{self.text}"


STAGES = {}
PIPELINES = {
    "none": (),
    "light": ("concise",),
    "code": ("code",),
    "full": ("code", "concise"),
}
DEFAULT_PIPELINE = "none"
# Bump when stage behavior changes; included in cache identity and telemetry.
PIPELINE_VERSION = "1"
CONCISE_INSTRUCTION = (
    "Use the shortest complete answer. Preserve required facts, caveats, code, "
    "and requested formatting. Follow any request for detail; avoid filler and repetition."
)


def stage(name):
    """Register the decorated function as a stage called `name`."""
    def register(fn):
        if name in STAGES:
            raise ValueError(f"stage {name!r} is defined twice")
        if not name or name == "none" or "+" in name:
            raise ValueError(f"stage name {name!r} must be nonempty, not 'none', and contain no '+'")
        STAGES[name] = fn
        return fn
    return register


def unknown_stages(stages):
    """The names in `stages` that are not registered."""
    return [name for name in stages if name not in STAGES]


def validate_pipeline(name: str) -> str:
    """Validate a named preset, rather than an arbitrary stage label."""
    if not isinstance(name, str) or name not in PIPELINES:
        raise ValueError(f"unknown compression pipeline {name!r}, expected one of {sorted(PIPELINES)}")
    return name


def label(stages):
    """How an ordered stage list is stored: 'a+b', or 'none' for no stages."""
    return "+".join(stages) or "none"


def parse_label(text):
    """The reverse of label(): 'a+b' -> ['a', 'b'], 'none' -> []."""
    return [] if text == "none" else text.split("+")


def resolve_stages(selection=()):
    """Resolve a preset, stage label, or ordered list/tuple into valid names."""
    if isinstance(selection, str):
        names = PIPELINES[selection] if selection in PIPELINES else parse_label(selection)
    elif isinstance(selection, (list, tuple)):
        names = selection
    else:
        raise ValueError("stages must be a preset, stage label, or a list or tuple of stage names")
    if any(not isinstance(name, str) for name in names):
        raise ValueError("every stage name must be a string")
    if unknown := unknown_stages(names):
        kind = "unknown compression pipeline or stages" if isinstance(selection, str) else "unknown stages"
        raise ValueError(f"{kind} {unknown}, expected some of {sorted(STAGES)}")
    return tuple(names)


def fingerprint(selection=()):
    """Describe effective stages and policy, sharing identity across aliases."""
    stages = resolve_stages(selection)
    return json.dumps(
        {"version": PIPELINE_VERSION, "stages": stages,
         "concise_instruction": CONCISE_INSTRUCTION if "concise" in stages else None},
        sort_keys=True, separators=(",", ":"),
    )


def compress(prompt, stages=()):
    """Run selected stages in order. The empty stage list is an exact no-op."""
    for name in resolve_stages(stages):
        prompt = STAGES[name](prompt)
    return prompt


# Stage modules import Prompt and @stage from here, so registration happens last.
from compressor import stages  # noqa: E402,F401
