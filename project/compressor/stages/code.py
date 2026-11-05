"""Parser-checked code compression for the question and context chunks."""

from dataclasses import replace

from compressor.code import compress_code_blocks
from compressor.pipeline import Prompt, stage


@stage("code")
def code(prompt: Prompt) -> Prompt:
    return replace(
        prompt,
        text=compress_code_blocks(prompt.text),
        context=tuple(replace(chunk, text=compress_code_blocks(chunk.text)) for chunk in prompt.context),
    )
