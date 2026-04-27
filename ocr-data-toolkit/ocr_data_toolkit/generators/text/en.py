from typing import List, Dict, Tuple, Optional, Union
import string
import random

from ...helper.utils import generate_random_date
from ...common.config import Config
from ..text.text import TextGenerator


class ENGenerator(TextGenerator):
    """
    English word salad Text Generator for Synthetic OCR Data.

    This class generates synthetic text samples (random words, dates, numbers).

    Attributes:
        bag_of_words (List[str]): List of words for text synthesis.
        text_probs (Dict[str, float]): Probabilities for text, date, or number generation.
    """
    def __init__(
        self,
        bag_of_words: Optional[List[str]] = None,
        text_probs: Optional[Dict[str, float]] = None,
        punctuation: List[str] = ['-', '<', '/', ',', "'", ':', '&', '.', '(', ')'],
        max_num_words: int = 5,
        num_lines: Union[int, Tuple[int, int]] = 1,
    ):
        """
        Initialize the ENGenerator.

        Args:
            bag_of_words (List[str]): List of words for generating random text.
            text_probs (Dict[str, float]): Probabilities for generating text, date, or number.
            max_num_words: The maximum number of words to generate in the word salad.
            num_lines: The number of lines to generate in the text.
        """
        super().__init__('en')

        config = Config()
        if bag_of_words is None:
            words_path = config.supported_languages[self.language]["words_path"]
            bag_of_words = [x.replace("\n", "").strip() for x in open(words_path, "r").readlines() if x.strip() != ""][1:]
        if text_probs is None:
            text_probs = config.text_probs

        self.bag_of_words = bag_of_words
        self.text_probs = text_probs
        self.punctuations = punctuation
        self.max_num_words = max_num_words
        self.num_lines = num_lines

    def _generate_text(self) -> str:
        """
        Generate a random text sample, date, or number based on configured probabilities.

        Returns:
            str: Generated text sample.
        """
        # Choose what type of text to generate (text/date/number) based on probabilities

        toGenerate = random.choices(list(self.text_probs.keys()), weights=self.text_probs.values(), k=1)[0]
        if toGenerate == "text":
            num_lines = random.randint(*self.num_lines) if isinstance(self.num_lines, tuple) else self.num_lines
            total_words = random.randint(self.max_num_words * (num_lines), self.max_num_words * num_lines)
            words = random.choices(self.bag_of_words, k=total_words)
            lines = []
            words_per_line = total_words // num_lines
            for i in range(num_lines):
                if i == num_lines - 1:
                    # Last line is shorter
                    line_words = words[i * words_per_line : ]
                    if len(line_words) > 4:
                        line_words = line_words[:len(line_words) // 2]  # cut it short
                else:
                    line_words = words[i * words_per_line : (i + 1) * words_per_line]
                line = ' '.join(line_words)
                # Add optional punctuation
                if random.random() > 0.7 and len(line) > 3:
                    punct = random.choice(self.punctuations)
                    if punct != '<':
                        si = random.choice([*[i for i, x in enumerate(line) if x == ' '], 0, len(line) - 1])
                        ps = random.choice([f' {punct}', f' {punct} ', f'{punct} '])
                        line = line[:si] + ps + line[si + 1:]
                lines.append(line)

            to_case = random.choices([str.upper, str.lower, str.capitalize, str.title], weights=[0.3, 0.3, 0.2, 0.2], k=1)[0]
            text = to_case("\n".join(lines))

        elif toGenerate == "date":
            text = self._generate_date()
        else:
            text = self._generate_number()
        return text

    def _generate_date(self) -> str:
        """
        Generate a random date string using the helper function.

        Returns:
            str: Randomly generated date string.
        """
        return generate_random_date()

    def _generate_number(self) -> str:
        """
        Generate a random alphanumeric number string.

        Returns:
            str: Randomly generated number string.
        """
        # Build a string of random digits and uppercase letters, sometimes inserting a dash

        l = []
        for _ in range(random.randint(1, 15)):
            if random.random() > 0.3:
                letter = string.digits[random.randint(0, 9)]
            else:
                letter = random.choice([string.ascii_uppercase[random.randint(0, 25)], "-"])
            l.append(letter)
        return ''.join(l)

    def __call__(self) -> Tuple[str, str]:
        """
        Callable interface: generate a single text.

        Returns:
            Tuple[str, str]: The generated text twice.
        """
        text = self._generate_text()
        return (text, text)
