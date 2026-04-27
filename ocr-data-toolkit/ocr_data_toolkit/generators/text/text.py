from typing import Tuple
from ...common.config import Config

class TextGenerator:
    """
    Text Generator for Synthetic OCR Data.

    This class is for generating synthetic text samples.
    """
    def __init__(
        self,
        language: str = Config.language,
    ) -> None:
        config = Config()
        if language not in config.supported_languages:
            raise ValueError(f"Language {language} is not supported. Supported languages are {config.supported_languages}")

        self.language = language

    def __call__(self) -> Tuple[str, str]:
        """
        Callable interface: generate a single text.

        Returns:
            Tuple[str, str]: A pair of texts where the first text is the generated ground truth text and the second text is the text as it will be inserted into the image (usually these are the same text).
        """
        raise NotImplementedError()
