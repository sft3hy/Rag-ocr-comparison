import torch
from transformers import AutoModelForCausalLM

moondream = AutoModelForCausalLM.from_pretrained(
    "moondream/moondream3-preview",
    trust_remote_code=True,
    dtype=torch.bfloat16,
    device_map={"": "mps"},
)
moondream.compile()


from PIL import Image

# Simple VQA
image = Image.open("https://wpdatatables.com/wp-content/uploads/2020/08/chart5.jpg")
result = moondream.query(image=image, question="What's in this image?")
print(result["answer"])
