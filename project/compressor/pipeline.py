"""Compression stages that POST /ask and the benchmark can switch between.

A stage is a function that takes a Prompt and returns a (shorter) Prompt.
Each request lists the stages it wants, e.g. ["strip_whitespace", "drop_comments"],
and they run in that order. No stages (the default) sends the prompt unchanged.

To add a stage, make a file in compressor/stages/ (every file there is loaded
automatically) and register the function with @stage:

    from compressor.pipeline import Prompt, stage

    @stage("strip_whitespace")
    def strip_whitespace(prompt: Prompt) -> Prompt:
        return replace(prompt, text=" ".join(prompt.text.split()))

then send its name in a request's stages. See compressor/stages/example.py
for a complete one.

The stages are stored with each request as label(stages), e.g. "strip_whitespace+drop_comments"
or "none" (the `compression` column), so the benchmark and the telemetry summary
can compare combinations of stages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Chunk:
    """A piece of a document from the knowledge base."""

    text: str
    source: str = ""  # where it came from, e.g. a file name


@dataclass(frozen=True)
class Prompt:
    """What gets sent to the model: the user's prompt, an optional system prompt
    and the knowledge-base chunks to answer from.

    Frozen, so a stage returns a new Prompt (dataclasses.replace) instead of editing this one.
    """

    text: str
    system: str | None = None
    context: tuple[Chunk, ...] = ()

    def render(self):
        """The text sent to the model: the chunks (if any), then the prompt."""
        if not self.context:
            return self.text
        chunks = "\n\n".join(f"[{c.source}]\n{c.text}" if c.source else c.text for c in self.context)
        return f"Context:\n{chunks}\n\nQuestion:\n{self.text}"


# stage name -> function(Prompt) -> Prompt, filled in by @stage
STAGES = {}


def stage(name):
    """Register the decorated function as a stage called `name`."""

    def register(fn):
        if name in STAGES:
            raise ValueError(f"stage {name!r} is defined twice")
        if name == "none" or "+" in name:  # would be mixed up with label()
            raise ValueError(f"stage name {name!r} can't be 'none' or contain '+'")
        STAGES[name] = fn
        return fn

    return register


def unknown_stages(stages):
    """The names in `stages` that aren't registered."""
    return [name for name in stages if name not in STAGES]


def compress(prompt, stages=()):
    """Run each stage in `stages` on `prompt`, in order. Raises ValueError for an unknown stage."""
    if unknown := unknown_stages(stages):
        raise ValueError(f"unknown stages {unknown}, expected some of {sorted(STAGES)}")
    for name in stages:
        prompt = STAGES[name](prompt)
    return prompt


def label(stages):
    """How a list of stages is stored and shown: "a+b", or "none" for no stages."""
    return "+".join(stages) or "none"


def parse_label(text):
    """The reverse of label(): "a+b" -> ["a", "b"], "none" -> []."""
    return [] if text == "none" else text.split("+")


from compressor import stages  # noqa: E402,F401  (at the bottom: the stage files import @stage from here)
