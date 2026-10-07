from dataclasses import dataclass 

@dataclass
class ModelPricing:
    """
    LLM input and output fees. 

    Pricing unit convention: $cost per million tokens.

    input_cost: Cost per million input tokens.
    output_cost: Cost per million output tokens.
    """
    input_cost: float
    output_cost: float

    # Just added some random formulas, this is to be filled in later.
    
    def calculate_total_cost(self, input_tokens, output_tokens: int) -> float:
        return self.input_cost*input_tokens + self.output_cost*output_tokens

    def calculate_input_cost(self, input_tokens: int) -> float:
        return self.input_cost*input_tokens / 2

    def calculate_output_cost(self, output_tokens: int) -> float:
        return self.output_cost*output_tokens / 2

