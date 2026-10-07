from model_data.token_counter import TokenCounter
from model_data.model_package import ModelPackage
from model_data.model_pricing import ModelPricing
from compressor.compressor_service import CompressorService

from google import genai
from dotenv import load_dotenv
import os

load_dotenv()

api_key = os.environ.get("GEMINI_API_KEY")
if not api_key:
    raise RuntimeError("GEMINI_API_KEY not set, copy .env.example to .env and add your key")
client = genai.Client(api_key=api_key)


# Add token counting logic (this class would be defined elsewhere)
class GeminiTokenCounter(TokenCounter):
    def __init__(self, client: genai):
        self.client = client

    # Counts input tokens without generating anything
    def count(self, text: str) -> int:
        response = self.client.models.count_tokens(
            model="gemini-3.5-flash",
            contents=text
        )

        return response.total_tokens

# Pricing (just adding some random defaults)
# input: $0.50 per million tokens
# output: $3.00 per million tokens
gemini_pricing = ModelPricing(
    input_cost=0.50, 
    output_cost=3.00
)

# Token counter
gemini_token_counter = GeminiTokenCounter(client)

# Package pricing and tokens. This design choice could be changed
gemini_package = ModelPackage(gemini_pricing, gemini_token_counter)

prompt = "Hello world!"

# Doesn't work because of high demand:
# response = client.models.generate_content(
#     model="gemini-3.5-flash",
#     contents=prompt
# )

# output = response.text
output = "Hello, how are you?"

# Compressed metrics
c = CompressorService(prompt, gemini_package, output)

print(c.get_text_information())
print(c.get_token_information())
print(c.get_pricing_information())
