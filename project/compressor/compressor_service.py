from dataclasses import replace

from model_data.model_package import ModelPackage
from compressor.pipeline import Prompt, compress, label, resolve_stages, validate_pipeline

class CompressorService:
    """
    Compresses input text and calculates metrics.

    """
    input_text: str
    output_text: str | None
    compressed_text: str
    model_package: ModelPackage
    stages: list[str]
    pipeline: str

    def __init__(
        self,
        prompt: Prompt | str,
        model_package: ModelPackage,
        output: str | None = None,
        stages=(),
        *,
        pipeline: str | None = None,
    ):
        if pipeline is not None:
            validate_pipeline(pipeline)
            if stages:
                raise ValueError("choose either stages or a pipeline preset, not both")
            stages = pipeline
        resolved = resolve_stages(stages)
        original_prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
        compressed_prompt = compress(original_prompt, resolved)

        self.input_text = original_prompt.render()
        self.original_prompt = original_prompt
        self.output_text = output
        self.model_package = model_package
        self.stages = list(resolved)
        self.pipeline = pipeline if pipeline is not None else label(resolved)
        self.compressed_prompt = compressed_prompt
        self.compressed_text = compressed_prompt.render()

    def get_text_information(self):
        return {
            "uncompressed_text": self.input_text,
            "compressed_text": self.compressed_text,
            "output_text": self.output_text
        }

    def get_token_information(self):
        """
        Token information
        """
        tokenizer = self.model_package.tokenizer.count
        # Include system instructions, including concise guidance added by a stage.
        # Reuse identical counts within this request to avoid duplicate API calls.
        counts = {}

        def count_prompt(prompt):
            total = 0
            for text in (prompt.render(), prompt.system):
                if not text:
                    continue
                if text not in counts:
                    counts[text] = tokenizer(text)
                total += counts[text]
            return total

        uncompressed_input_tokens = count_prompt(self.original_prompt)
        compressed_input_tokens = count_prompt(self.compressed_prompt)
        if "code" in self.stages:
            # Fewer characters do not always mean fewer model tokens. Keep
            # concise guidance, but undo code compaction if its real count grew.
            original_text_tokens = counts.get(self.original_prompt.render(), 0)
            compact_text_tokens = counts.get(self.compressed_prompt.render(), 0)
            if compact_text_tokens > original_text_tokens:
                compressed_input_tokens -= compact_text_tokens - original_text_tokens
                self.compressed_prompt = replace(
                    self.compressed_prompt, text=self.original_prompt.text,
                    context=self.original_prompt.context,
                )
                self.compressed_text = self.original_prompt.render()
        # May be negative for a conciseness policy, which targets output tokens.
        tokens_saved = uncompressed_input_tokens - compressed_input_tokens

        return {
            "uncompressed_input_tokens" : uncompressed_input_tokens,
            "compressed_input_tokens" : compressed_input_tokens,
            "tokens_saved" : tokens_saved
        }


    def get_pricing_information(self):
        """ 
        Pricing information (this will have to be an estimate, not sure how to predict). 
        
        Pricing calculation depends on both input and output token counts and the models pricing 
        
        You can calculate compressed/uncompressed input cost in advance. 
        You cannot calculate compressed/uncompresesd output cost in advance (how would you know this without running the model first?) 
        
        Perhaps you can roughly calculate savings by doing this? : 

        savings = (uncompressed input cost + estimated uncompressed output cost) - (compressed input cost - actual output cost)
        
        You can pull some "$cost per million tokens" metrics for different models off their documentation websites. 
        """

        pricing = self.model_package.pricing
        if pricing is None:
            raise ValueError("Model pricing is not configured")
        tokens = self.get_token_information()
        output = self.output_text

        # Get token counts
        if output:
            output_tokens = self.model_package.tokenizer.count(output)

        uncompressed_tokens = tokens["uncompressed_input_tokens"]
        compressed_tokens = tokens["compressed_input_tokens"]

        # Calculate input costs
        uncompressed_input_cost = pricing.calculate_input_cost(uncompressed_tokens)
        compressed_input_cost = pricing.calculate_input_cost(compressed_tokens)

        # Calculate output cost
        if output:
            output_cost = pricing.calculate_total_cost(compressed_tokens, output_tokens)
        else:
            output_cost = None

        # Estimated savings from compression
        savings = uncompressed_input_cost - compressed_input_cost

        return {
            "uncompressed_input_cost": uncompressed_input_cost,
            "compressed_input_cost": compressed_input_cost,
            "output_cost": output_cost,
            "savings": savings
        }
