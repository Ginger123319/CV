# make sure you're logged in with `huggingface-cli login`
import os
os.environ['CURL_CA_BUNDLE'] = ''

from torch import autocast
from diffusers import StableDiffusionPipeline

pipe = StableDiffusionPipeline.from_pretrained(
	"CompVis/stable-diffusion-v1-4", local_files_only=True
).to("cuda")

prompt = "oil painting of beautiful pokemon, masterpiece"
with autocast("cuda"):
    image = pipe(prompt)["images"][0]  
    
image.save("sample.png")