from model_data.model_package import ModelPackage
from compressor.pipeline import Prompt, compress

class CompressorService:
    """
    Compresses input text and calculates metrics.

    """
    input_text: str
    output_text: str | None
    compressed_text: str
    model_package: ModelPackage
    stages: list[str]

    def __init__(
        self,
        prompt: Prompt | str,
        model_package: ModelPackage,
        output: str | None = None,
        stages=(),
    ):
        original_prompt = prompt if isinstance(prompt, Prompt) else Prompt(prompt)
        compressed_prompt = compress(original_prompt, stages)

        # render(): the text actually sent, knowledge-base chunks included
        self.input_text = original_prompt.render()
        self.output_text = output
        self.model_package = model_package
        self.stages = list(stages)
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
        uncompressed = self.input_text
        compressed = self.compressed_text

        uncompressed_input_tokens, compressed_input_tokens = tokenizer(uncompressed), tokenizer(compressed)
        tokens_saved = uncompressed_input_tokens - compressed_input_tokens # if this becomes negative then we did not compress properly

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





    