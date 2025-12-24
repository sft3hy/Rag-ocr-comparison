"""
Vision Model Factory and Implementations

Provides a unified interface for multiple vision models:
- Moondream2 (Native)
- Qwen3-VL (Native)
- InternVL3.5 (Native)
- Ollama (Generic support for Gemma 3, Llama 3.2 Vision, Moondream)

Now with model offloading support to free up GPU/memory resources.
"""

import torch
from PIL import Image
from abc import ABC, abstractmethod
from typing import Optional
import warnings
import time
import gc
import io

warnings.filterwarnings("ignore")


class VisionModel(ABC):
    """Base class for all vision models"""

    def __init__(self):
        self.device = self._get_device()
        self.model = None
        self.tokenizer = None
        self.processor = None
        self._is_loaded = False

    def _get_device(self) -> str:
        """Determine the best available device"""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
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

    def offload_model(self):
        """Offload the model from GPU/memory to free up resources"""
        if not self._is_loaded:
            print(f"{self.get_model_name()} is not loaded, nothing to offload.")
            return

        print(f"Offloading {self.get_model_name()}...")

        # Delete model, tokenizer, and processor
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

        # Clear CUDA cache if using CUDA
        if self.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Clear MPS cache if using MPS
        if self.device == "mps" and torch.backends.mps.is_available():
            torch.mps.empty_cache()

        self._is_loaded = False
        print(f"✓ {self.get_model_name()} offloaded successfully")

    def is_loaded(self) -> bool:
        """Check if the model is currently loaded"""
        return self._is_loaded


class Moondream2Model(VisionModel):
    """Moondream2 vision model implementation (Native Python)"""

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

            self._is_loaded = True
            print(f"✓ Moondream2 loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"✗ Failed to load Moondream2: {e}")
            self._is_loaded = False
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Moondream2"""
        if not self._is_loaded:
            return "[Error: Model not loaded. Please load the model first.]"

        try:
            start_time = time.time()
            enc_image = self.model.encode_image(image)
            description = self.model.answer_question(enc_image, prompt, self.tokenizer)
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

            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_id,
                dtype=torch.float16,
                device_map=self.device,
                trust_remote_code=True,
            )

            self.model.eval()

            self._is_loaded = True
            print(f"✓ Qwen3-VL loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"✗ Failed to load Qwen3-VL: {e}")
            self._is_loaded = False
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Qwen3-VL"""
        if not self._is_loaded:
            return "[Error: Model not loaded. Please load the model first.]"

        try:
            start_time = time.time()
            from qwen_vl_utils import process_vision_info

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

            # Generate response
            with torch.inference_mode():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=2048,
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

            self._is_loaded = True
            print(f"✓ InternVL3.5 loaded successfully on {self.device}")
            return True

        except Exception as e:
            print(f"✗ Failed to load InternVL3.5: {e}")
            self._is_loaded = False
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using InternVL3.5"""
        if not self._is_loaded:
            return "[Error: Model not loaded. Please load the model first.]"

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


class OllamaVisionModel(VisionModel):
    """
    Generic Ollama Vision Model Wrapper.
    Supports models like 'gemma3', 'llama3.2-vision', 'moondream'.
    """

    def __init__(self, model_name="gemma3"):
        super().__init__()
        # Use simple model name logic:
        # If user asks for Gemma 3, we default to 'gemma3' (4B) as 1B is text-only.
        self.ollama_model_name = model_name

    def load_model(self):
        """Check availability of Ollama and the requested model"""
        try:
            import ollama

            print(f"Connecting to Ollama (Target: {self.ollama_model_name})...")

            # Check if ollama service is reachable
            # Handle object-based return (newer libs) vs dict-based (older libs)
            try:
                list_response = ollama.list()
                if hasattr(list_response, "models"):
                    models = [m.model for m in list_response.models]
                else:
                    models = [m.get("name") for m in list_response.get("models", [])]
            except Exception:
                models = []

            # Fuzzy check if model is pulled
            is_present = any(self.ollama_model_name in m for m in models)

            if not is_present:
                print(f"Model '{self.ollama_model_name}' not found. Pulling...")
                ollama.pull(self.ollama_model_name)
                print(f"✓ Pulled {self.ollama_model_name}")

            self._is_loaded = True
            print(f"✓ Ollama ({self.ollama_model_name}) is ready.")
            return True

        except ImportError:
            print("✗ 'ollama' python package not installed.")
            self._is_loaded = False
            return False
        except Exception as e:
            print(f"✗ Ollama connection failed: {e}")
            self._is_loaded = False
            return False

    def describe_image(self, image: Image.Image, prompt: str) -> str:
        """Generate description using Ollama chat interface"""
        if not self._is_loaded:
            return "[Error: Ollama not ready.]"

        try:
            import ollama

            start_time = time.time()

            # Convert PIL image to bytes (equivalent to reading path bytes)
            img_byte_arr = io.BytesIO()
            image.save(img_byte_arr, format="PNG")
            img_bytes = img_byte_arr.getvalue()

            # Use the chat endpoint as requested
            response = ollama.chat(
                model=self.ollama_model_name,
                messages=[{"role": "user", "content": prompt, "images": [img_bytes]}],
            )

            # Handle object response (new lib) vs dict response (old lib)
            if hasattr(response, "message"):
                content = response.message.content
            else:
                content = response["message"]["content"]

            end_time = time.time()
            print(
                f"Ollama ({self.ollama_model_name}) took: {round(end_time - start_time, 2)}s"
            )

            return content

        except Exception as e:
            err = str(e)
            if "missing data required for image input" in err:
                return (
                    f"[ERROR: Model '{self.ollama_model_name}' appears to be text-only. "
                    f"Please try 'gemma3' (4B) or 'llama3.2-vision'.]"
                )
            return f"[Ollama analysis failed: {err}]"

    def offload_model(self):
        """Force unload via API"""
        if not self._is_loaded:
            return
        try:
            import ollama

            ollama.generate(model=self.ollama_model_name, prompt="", keep_alive=0)
            self._is_loaded = False
            print(f"✓ {self.ollama_model_name} offloaded.")
        except:
            pass

    def get_model_name(self) -> str:
        return f"Ollama-{self.ollama_model_name}"


class VisionModelFactory:
    """Factory class to create vision model instances"""

    MODELS = {
        "Moondream2": Moondream2Model,
        "Qwen3-VL-2B": Qwen3VLModel,
        "InternVL3.5-1B": InternVL3Model,
        # Factory lambdas for Ollama models
        "Ollama-Gemma3": lambda: OllamaVisionModel(
            "gemma3"
        ),  # Defaults to 4B multimodal
        "Ollama-Granite3.2-Vision": lambda: OllamaVisionModel("granite3.2-vision"),
    }

    @classmethod
    def create_model(cls, model_name: str) -> Optional[VisionModel]:
        if model_name not in cls.MODELS:
            print(f"Unknown model: {model_name}")
            return None

        factory = cls.MODELS[model_name]
        # Handle simple class vs lambda factory
        if isinstance(factory, type):
            instance = factory()
        else:
            instance = factory()

        if instance.load_model():
            return instance
        else:
            return None

    @classmethod
    def get_available_models(cls) -> list:
        return list(cls.MODELS.keys())


if __name__ == "__main__":
    # Example usage
    model = VisionModelFactory.create_model("Ollama-Gemma3")
    if model:
        print(f"Model loaded: {model.is_loaded()}")
        model.offload_model()
