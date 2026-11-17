from abc import abstractmethod


"""
Abstract class for TokenCounter.
"""
class TokenCounter:
    @abstractmethod
    def count(self, input: str) -> int:
        'Returns token count given an input'
        pass

# Temporary space for GeminiTokenCounter 
class GeminiTokenCounter(TokenCounter):
    def __init__(self, client, model="gemini-3.5-flash"):
        self.client = client
        self.model = model

    # Counts input tokens without generating anything
    def count(self, text: str) -> int:
        response = self.client.models.count_tokens(
            model=self.model,
            contents=text
        )

        return response.total_tokens