from typing import List, Dict, Tuple, Callable, Optional, Union
import random
from PIL import Image, ImageFont, ImageDraw, ImageFilter, ImageEnhance
import os
import glob
import logging

from ..common import AugmentationConfig
from ..helper.utils import (
    get_pil_font,
    add_background,
    get_max_char_dimensions
)
from ..helper.augmentation import Augmentation
from ..common.config import Config
from .text.text import TextGenerator


class DataGenerator:
    """
    Text-Image Generator for Synthetic OCR Data (given text).

    This class generates text samples and renders them onto images with configurable fonts, backgrounds, and augmentations. It is used for creating training data for OCR models.

    Attributes:
        text_gen (Callable[[], str] | Callable[[], Tuple[str, str]]): A text generator to use for putting text on images.
        fonts (List[str]): List of font file paths.
        backgrounds (List[str]): List of background image file paths.
        cfg (AugmentationConfig): Augmentation configuration object.
        augmentation (Augmentation): Augmentation operations.
    """
    def __init__(
        self,
        text_gen: TextGenerator,
        font_paths: Optional[Union[List[str], str]] = None,
        background_paths: Optional[Union[List[str], str]] = None,
        augmentation_config: Dict = None,
        logger: logging.RootLogger = logging.getLogger(__name__)
    ):
        """
        Initialize the ENGenerator.

        Args:
            text_gen (TextGenerator): A text generator to use for putting text on images.
            fonts_path (Optional[List[str] | str]): Either a list of font file paths or a path to a fonts directory. If None, use provided language default (text_gen.langauge).
            backgrounds (Optional[List[str] | str]): Either a list of background file paths or a path to a backgrounds directory. If None, use provided default.
            augmentation_config (Dict, optional): Augmentation configuration.
        """
        config = Config()

        self.text_gen = text_gen

        self.fonts = []
        if font_paths is None:
            font_paths = config.supported_languages[text_gen.language]["fonts_path"]
        if isinstance(font_paths, str):
            for font_path in glob.glob(os.path.join(font_paths, "**", "*.ttf"), recursive=True):
                self.fonts.append(font_path)
        else:
            self.fonts.extend(font_paths)
        logger.info("Fonts loaded.")

        self.backgrounds = []
        if background_paths is None:
            background_paths = config.backgrounds_path
        if isinstance(background_paths, str):
            for bg_name in os.listdir(background_paths):
                basename, ext = os.path.splitext(bg_name)
                if ext in ['.jpg', '.png', '.jpeg']:
                    self.backgrounds.append(os.path.join(config.backgrounds_path, bg_name))
        else:
            self.backgrounds.extend(background_paths)
        logger.info("Backgrounds loaded.")

        self.cfg = AugmentationConfig(augmentation_config)
        self.augmentation = Augmentation(self.cfg)

    def _estimate_image_size(self, text: str, font: ImageFont.ImageFont, letter_spacing: int) -> Tuple[int, int, int, int, int]:
        """
        Estimate the required image size for rendering the given text with the specified font and letter spacing.

        Args:
            text (str): Text to render.
            font (ImageFont.ImageFont): Font to use.
            letter_spacing (int): Extra spacing between letters.

        Returns:
            Tuple[int, int, int, int, int]: (final_width, final_height, margin_x, margin_y)
        """
        # Compute the width and height required to render the text, including margins

        lines = text.split("\n")
        img_w = 0
        ascent, descent = font.getmetrics()
        line_height = ascent + descent
        for line in lines:
            line_width = 0
            for char in line:
                bbox = font.getbbox(char)
                char_width = bbox[2] - bbox[0]
                line_width += (char_width + letter_spacing)
            img_w = max(img_w, line_width)
        img_h = line_height #* len(lines)

        max_char_width, max_char_height = get_max_char_dimensions(font)
        margin_x = int(random.uniform(*self.cfg["margin_x"]) * max_char_width)
        margin_y = int(random.uniform(*self.cfg["margin_y"]) * max_char_height)
        final_width = img_w + margin_x
        final_height = img_h + margin_y
        return final_width, final_height, margin_x, margin_y

    def _generate_single_image(self, text: str) -> Image.Image:
        """
        Generate a single synthetic image and its corresponding text.

        Args:
            text (str): Text to render.

        Returns:
            Image.Image: The rendered image.
        """
        # Select font and generate text if not provided

        font, font_path = get_pil_font(self.fonts, font_size=self.cfg['font_size'])
        letter_spacing = random.randint(1, 4) if random.random() < self.cfg['letter_spacing_prob'] else 0
        final_width, final_height, margin_x, margin_y = self._estimate_image_size(text, font, letter_spacing)
        img = self._create_base_image(final_width, final_height)
        self._draw_text(img, text, font, letter_spacing, margin_x, margin_y)
        img = self._apply_postprocessing(img, final_width)
        return img.convert("RGB")

    def _create_base_image(self, width: int, height: int) -> Image.Image:
        """
        Create a base image, either with a random background or plain white.

        Args:
            width (int): Image width.
            height (int): Image height.

        Returns:
            Image.Image: The base image.
        """
        # With 80% probability, use a random background; otherwise, use a white canvas

        if random.random() > 0.2:
            return add_background((width, height), self.backgrounds)
        else:
            return Image.new('L', (width, height), color='white')

    def _draw_text(self, img: Image.Image, text: str, font: ImageFont.ImageFont, letter_spacing: int, margin_x: int, margin_y: int):
        """
        Draw the provided text onto the image with the specified font and spacing.

        Args:
            img (Image.Image): Image to draw on.
            text (str): Text to render.
            font (ImageFont.ImageFont): Font to use.
            letter_spacing (int): Space between letters.
            margin_x (int): Horizontal margin.
            margin_y (int): Vertical margin.
        """
        # Draw each character of each line, handling random x/y offset and color

        draw = ImageDraw.Draw(img)
        x_init = margin_x #random.randint(0, margin_x)
        y = margin_y // 2
        color = random.choice(self.cfg['text_colors'])
        for line in text.split('\n'):
            x = x_init
            for char in line:
                bbox = draw.textbbox((0, 0), char, font=font)
                char_width = bbox[2] - bbox[0]
                char_height = bbox[3] - bbox[1]
                y = min(max(y, margin_y // 2), img.height - margin_y // 2 - char_height)
                draw.text((x, y), char, fill=color, font=font, align="center")
                x += char_width + letter_spacing
            y += int(char_height * 0.0)

    def _apply_postprocessing(self, img: Image.Image, width: int) -> Image.Image:
        """
        Apply postprocessing augmentations to the image (blur, brightness, moire, perspective, etc).

        Args:
            img (Image.Image): Image to process.
            width (int): Image width (for some augmentations).

        Returns:
            Image.Image: Augmented image.
        """
        # Apply a series of random augmentations to simulate real-world distortions

        img = img.convert("RGB")
        if random.random() < self.cfg['blur_probs']['gaussian']:
            img = img.filter(ImageFilter.GaussianBlur(random.uniform(0.2, 0.5)))

        if random.random() < self.cfg['blur_probs']['custom_blurs']:
            for op in random.choices([
                self.augmentation.guassianBlur,
                self.augmentation.motionBlur,
                self.augmentation.bokenBlur
            ], k=2):
                img = op(img, width=width)

        if random.random() < self.cfg['opacity_prob']:
            img.putalpha(random.randint(*self.cfg['opacity_range']))

        if random.random() < self.cfg['moire_prob']:
            img = self.augmentation.add_moire_patterns(img, alpha=random.uniform(0.1, 0.3))

        brightness_prob = 0.5
        if random.random() < brightness_prob:
            enhancer = ImageEnhance.Brightness(img)
            img = enhancer.enhance(random.uniform(*self.cfg['brightness_range']))

        if random.random() < self.cfg['perspective_transform_prob']:
            img = self.augmentation.apply_perspective_transform(img)
        return img

    def __call__(self) -> Tuple[str, Image.Image]:
        """
        Callable interface: generate a single text-image pair.

        Returns:
            Tuple[str, Image.Image]: The ground truth text and image with the text.
        """
        (gt_text, im_text)  = self.text_gen()
        return (gt_text, self._generate_single_image(im_text))
