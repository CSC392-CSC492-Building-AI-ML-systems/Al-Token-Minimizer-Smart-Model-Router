
class Compressor:
    """
    Text compression class.

    Logic for compressing test will be here
    """
    text: str
    def __init__(self, text: str):
        self.text = text

    def compress(self) -> str:
        """
        Compression function
        """

        # For testing purposes
        return self.text + "test"
    