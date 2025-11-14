from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import requests
import torch
import time

start_time = time.time()


device = "mps" if torch.backends.mps.is_available() else "cpu"

model_id = "vikhyatk/moondream2"
model = AutoModelForCausalLM.from_pretrained(
    model_id, trust_remote_code=True, dtype=torch.float16
).to(device)

tokenizer = AutoTokenizer.from_pretrained(model_id)

url = "https://www.qimacros.com/excel-charts-qimacros/column-chart-excel.png"
image = Image.open(requests.get(url, stream=True).raw)
# print(image)

enc_image = model.encode_image(image)
answer = model.answer_question(enc_image, "describe the chart", tokenizer)
print(answer)

end_time = time.time()

print(f"got answer in {round(end_time-start_time, 2)} seconds")
