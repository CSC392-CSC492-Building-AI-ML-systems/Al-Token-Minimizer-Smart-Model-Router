"""Compression pipelines that POST /ask and the benchmark can switch between.

A stage is a function that takes a Prompt and returns a (shorter) Prompt.
A pipeline is a named list of stages, run in order. Only "none" exists for
now; when a real stage is written, add it to STAGES and make pipelines out of
it, e.g.:

    STAGES = {"strip_whitespace": strip_whitespace, "drop_comments": drop_comments}
    PIPELINES = {
        "none": (),
        "light": ("strip_whitespace",),
        "full": ("strip_whitespace", "drop_comments"),
    }

The pipeline name is stored with each request (the `compression` column), so
the benchmark and the telemetry summary can compare pipelines, or single stages.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class Prompt:
    """What gets sent to the model: the user's prompt and an optional system prompt.

    Frozen, so a stage returns a new Prompt (dataclasses.replace) instead of editing this one.
    """

    text: str
    system: str | None = None


# stage name -> function(Prompt) -> Prompt
STAGES = {}

# pipeline name -> stage names, run in order
PIPELINES = {
    "none": (),
}

DEFAULT_PIPELINE = "none"


def compress(prompt, pipeline=DEFAULT_PIPELINE):
    """Run every stage of `pipeline` on `prompt`. Raises ValueError for an unknown pipeline."""
    if pipeline not in PIPELINES:
        raise ValueError(f"unknown compression pipeline {pipeline!r}, expected one of {sorted(PIPELINES)}")
    for stage in PIPELINES[pipeline]:
        prompt = STAGES[stage](prompt)
    return prompt
