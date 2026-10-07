from dataclasses import dataclass
from model_data.model_pricing import ModelPricing
from model_data.token_counter import TokenCounter 


@dataclass
class ModelPackage:
    """
    Contains relevant information of an LLM to calculate metrics.

    If we want to display more information aside from pricing and tokens, 
    perhaps put them in here so other classes have access to them.

    pricing: The relevant pricing for this model.
    tokenizer: A model specific token counter 
    """
    pricing: ModelPricing
    tokenizer: TokenCounter

