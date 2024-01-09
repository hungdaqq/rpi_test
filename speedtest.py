from urllib.request import urlopen
from PIL import Image
import timm
import torch

model = timm.create_model('fastvit_s12.apple_in1k')
model = model.eval()

inputs = torch.randn(1, 3, 256, 256)  # Adjust the input shape based on your model

device = 'cpu'
model.to(device)
inputs = inputs.to(device)

import time

# Assuming 'model' and 'inputs' are already defined

num_samples = 10  # Adjust based on your needs
total_latency = 0

for _ in range(num_samples):
    start_time = time.time()
    output = model(inputs)
    end_time = time.time()
    latency = (end_time - start_time) * 1000  # Convert to milliseconds for better readability
    print(latency)
    total_latency += latency

average_latency = total_latency / num_samples
print(f"Average Latency: {average_latency:.2f} milliseconds")
