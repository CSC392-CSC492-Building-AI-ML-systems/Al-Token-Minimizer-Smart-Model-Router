"""Optional output guidance that retains caller requirements."""

from dataclasses import replace

from compressor import pipeline


@pipeline.stage("concise")
def concise(prompt: pipeline.Prompt) -> pipeline.Prompt:
    policy = pipeline.CONCISE_INSTRUCTION
    if prompt.system == policy or (prompt.system or "").startswith(policy + "\n\n"):
        return prompt
    system = policy
    if prompt.system:
        system += "\n\n" + prompt.system
    return replace(prompt, system=system)
