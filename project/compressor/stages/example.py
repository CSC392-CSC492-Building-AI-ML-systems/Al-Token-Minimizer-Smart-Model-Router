"""An example stage to copy from. Send {"stages": ["passthrough"]} to POST /ask to run it."""

from compressor.pipeline import Prompt, stage


@stage("passthrough")
def passthrough(prompt: Prompt) -> Prompt:
    """example stage, this stage does literally nothing"""
    return prompt
