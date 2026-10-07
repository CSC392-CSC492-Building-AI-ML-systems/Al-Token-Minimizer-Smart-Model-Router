from abc import abstractmethod

"""
Abstract class for TokenCounter.
"""
class TokenCounter:
    @abstractmethod
    def count(self, input: str) -> int:
        'Returns token count given an input'
        pass
