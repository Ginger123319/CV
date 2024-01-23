from diffusers import StableDiffusionPipeline
import torch
import copy

model_path = "sd-pokemon-model-lora"
pipe = StableDiffusionPipeline.from_pretrained("CompVis/stable-diffusion-v1-4", torch_dtype=torch.float16, local_files_only=True)
model1 = copy.deepcopy(pipe.unet)
pipe.unet.load_attn_procs(model_path)
model2 = pipe.unet
pipe.to("cuda")

import torch

# 获取模型的权重
weights1 = model1.state_dict()
weights2 = model2.state_dict()

# 计算权重的大小
size1 = sum(p.numel() for p in weights1.values())
size2 = sum(p.numel() for p in weights2.values())

# 比较权重大小
if size1 > size2:
    print("Model 1 has larger weight size.")
elif size1 < size2:
    print("Model 2 has larger weight size.")
else:
    print("Both models have the same weight size.")


prompt = "A pokemon with red eyes and green legs."
image = pipe(prompt, num_inference_steps=30, guidance_scale=7.5).images[0]
image.save("pokemon.png")