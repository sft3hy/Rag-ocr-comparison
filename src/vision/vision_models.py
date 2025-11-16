"""
Vision Model Factory and Implementations

Provides a unified interface for multiple vision models:
- Moondream2
- Qwen3-VL
- InternVL3.5

With automatic memory management to ensure only one model is loaded at a time.
"""

import torch
from PIL import Image
from abc import ABC, abstractmethod
from typing import Optional
import warnings
import time
import gc

warnings.filterwarnings("ignore")


class VisionModel(ABC):
    """Base class for all vision models"""

    def __init__(self):
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.processor = None

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.backends.mps.is_available():
            return "mps"
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    @abstractmethod
    def load_model(self):
        """Load the vision model"""
        pass

    @abstractmethod
    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate a description for the given image"""
        pass

    @abstractmethod
    def get_model_name(self) -> str:
        """Return the display name of the model"""
        pass

    def unload_model(self):
        """Unload the model and free memory"""
        print(f"Unloading {self.get_model_name()}...")

        # Delete model components
        if self.model is not None:
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if self.processor is not None:
            del self.processor
            self.processor = None

        # Force garbage collection
        gc.collect()

        # Clear CUDA cache if available
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear MPS cache if available
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()

        print(f"✓ {self.get_model_name()} unloaded from memory")


class Moondream2Model(VisionModel):
    """Moondream2 vision model implementation"""

    def load_model(self):
        """Load Moondream2 model"""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer

            print(f"Loading Moondream2 on {self.device}...")
            model_id = "vikhyatk/moondream2"

            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                revision="2025-06-21",
                trust_remote_code=True,
                device_map={"": self.device},
            )

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
            )

            print(f"✓ Moondream2 loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"✗ Failed to load Moondream2: {e}")
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Moondream2"""
        try:
            start_time = time.time()
            enc_image = self.model.encode_image(image)
            description = self.model.caption(enc_image, length="normal")["caption"]
            end_time = time.time()
            print(
                f"Moondream image description took: {round(end_time - start_time, 2)} seconds."
            )
            return description
        except Exception as e:
            return f"[Moondream2 analysis failed: {str(e)}]"

    def get_model_name(self) -> str:
        return "Moondream2"


class Qwen3VLModel(VisionModel):
    """Qwen3-VL vision model implementation"""

    def load_model(self):
        """Load Qwen3-VL model"""
        try:
            from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

            print(f"Loading Qwen3-VL on {self.device}...")
            model_id = "Qwen/Qwen3-VL-2B-Instruct"

            self.processor = AutoProcessor.from_pretrained(
                model_id, trust_remote_code=True, use_fast=True
            )

            # Use bfloat16 instead of float16 for better numerical stability
            dtype = torch.bfloat16 if self.device != "mps" else torch.float32

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=dtype,
                device_map=self.device,
                trust_remote_code=True,
            )

            self.model.eval()

            print(f"✓ Qwen3-VL loaded successfully on {self.device} with dtype={dtype}")
            return True

        except Exception as e:
            print(f"✗ Failed to load Qwen3-VL: {e}")
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Qwen3-VL"""
        try:
            start_time = time.time()
            from qwen_vl_utils import process_vision_info

            # Ensure image is RGB
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Resize large images to prevent memory explosion
            max_dimension = 1024
            width, height = image.size
            if width > max_dimension or height > max_dimension:
                if width > height:
                    new_width = max_dimension
                    new_height = int(height * (max_dimension / width))
                else:
                    new_height = max_dimension
                    new_width = int(width * (max_dimension / height))

                print(
                    f"    Resizing image from {width}x{height} to {new_width}x{new_height}"
                )
                image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Prepare messages
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt},
                    ],
                }
            ]

            # Prepare inputs
            text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = self.processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(self.device)

            # Generate response with proper parameters
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
                    do_sample=False,
                    temperature=None,
                    top_p=None,
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    eos_token_id=self.processor.tokenizer.eos_token_id,
                )
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :]
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                response = self.processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )[0]

            end_time = time.time()
            print(
                f"Qwen3 image description took: {round(end_time - start_time, 2)} seconds."
            )

            return response

        except Exception as e:
            import traceback

            traceback.print_exc()
            return f"[Qwen3-VL analysis failed: {str(e)}]"

    def get_model_name(self) -> str:
        return "Qwen3-VL-2B"


class InternVL3Model(VisionModel):
    """InternVL3.5 vision model implementation"""

    def load_model(self):
        """Load InternVL3.5 model"""
        try:
            from transformers import AutoModel, AutoTokenizer

            print(f"Loading InternVL3.5 on {self.device}...")
            model_id = "OpenGVLab/InternVL3_5-1B-Flash"

            self.model = AutoModel.from_pretrained(
                model_id,
                dtype=torch.float16,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map=self.device,
            )

            self.model.eval()

            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=True,
            )

            print(f"✓ InternVL3.5 loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"✗ Failed to load InternVL3.5: {e}")
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using InternVL3.5"""
        start_time = time.time()
        try:
            from torchvision import transforms as T
            from torchvision.transforms.functional import InterpolationMode

            # InternVL image preprocessing
            IMAGENET_MEAN = (0.485, 0.456, 0.406)
            IMAGENET_STD = (0.229, 0.224, 0.225)

            def build_transform(input_size):
                transform = T.Compose(
                    [
                        T.Lambda(
                            lambda img: img.convert("RGB") if img.mode != "RGB" else img
                        ),
                        T.Resize(
                            (input_size, input_size),
                            interpolation=InterpolationMode.BICUBIC,
                        ),
                        T.ToTensor(),
                        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
                    ]
                )
                return transform

            # Process image
            transform = build_transform(input_size=448)
            pixel_values = (
                transform(image).unsqueeze(0).to(torch.float16).to(self.device)
            )

            # Generate response
            generation_config = dict(
                max_new_tokens=2048,
                do_sample=False,
                eos_token_id=151645,
                pad_token_id=151645,
            )
            question_with_image = f"<image>\n{prompt}"

            response = self.model.chat(
                self.tokenizer, pixel_values, question_with_image, generation_config
            )
            end_time = time.time()
            print(
                f"InternVL image description took: {round(end_time - start_time, 2)} seconds."
            )

            return response

        except Exception as e:
            return f"[InternVL3.5 analysis failed: {str(e)}]"

    def get_model_name(self) -> str:
        return "InternVL3.5-1B"


class VisionModelFactory:
    """Factory class to create vision model instances with memory management"""

    MODELS = {
        "Moondream2": Moondream2Model,
        "Qwen3-VL-2B": Qwen3VLModel,
        "InternVL3.5-1B": InternVL3Model,
    }

    _current_model: Optional[VisionModel] = None
    _current_model_name: Optional[str] = None

    @classmethod
    def create_model(cls, model_name: str) -> Optional[VisionModel]:
        """
        Create and load a vision model by name.
        Automatically unloads any previously loaded model.

        Args:
            model_name: Name of the model to load

        Returns:
            Loaded VisionModel instance or None if failed
        """
        if model_name not in cls.MODELS:
            print(f"Unknown model: {model_name}")
            print(f"Available models: {list(cls.MODELS.keys())}")
            return None

        # Unload current model if it exists and is different
        if cls._current_model is not None and cls._current_model_name != model_name:
            cls._current_model.unload_model()
            cls._current_model = None
            cls._current_model_name = None

        # Return existing model if already loaded
        if cls._current_model_name == model_name:
            print(f"✓ {model_name} already loaded, reusing instance")
            return cls._current_model

        # Load new model
        model_class = cls.MODELS[model_name]
        model_instance = model_class()

        if model_instance.load_model():
            cls._current_model = model_instance
            cls._current_model_name = model_name
            return model_instance
        else:
            return None

    @classmethod
    def unload_current_model(cls):
        """Explicitly unload the currently loaded model"""
        if cls._current_model is not None:
            cls._current_model.unload_model()
            cls._current_model = None
            cls._current_model_name = None

    @classmethod
    def get_current_model_name(cls) -> Optional[str]:
        """Get the name of the currently loaded model"""
        return cls._current_model_name

    @classmethod
    def get_available_models(cls) -> list:
        """Get list of available model names"""
        return list(cls.MODELS.keys())
