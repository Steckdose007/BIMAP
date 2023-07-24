import PIL
import requests
import torch
from io import BytesIO
import matplotlib.pyplot as plt
from diffusers import StableDiffusionInstructPix2PixPipeline


img_url = "https://huggingface.co/datasets/diffusers/diffusers-images-docs/resolve/main/mountain.png"
print("load image")
image = PIL.Image.open("/home/woody/iwb3/iwb3009h/images1024x1024/00048.png").convert("RGB")

print("load pipeline")
pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
    "/home/woody/iwb3/iwb3009h/trained/", torch_dtype=torch.float16
)

pipe = pipe.to("cuda")
print("generate image")
prompt = "Make this male look 70-120 years old."
image = pipe(prompt=prompt, image=image, image_guidance_scale=0.5, guidance_scale=7.5,num_inference_steps=100).images[0]
image.save("img1.png")


num_images = 10
fig, ax = plt.subplots(1, num_images)
plt.axis('off')
inter = 0
image_guidance_scale = 0
guidance_scale = 4
for idx in range(0, num_images):
    prompt = "Make this male look 70-120 years old."
    image = PIL.Image.open("/home/woody/iwb3/iwb3009h/images1024x1024/00048.png").convert("RGB")
    image = pipe(prompt=prompt, image=image, image_guidance_scale=image_guidance_scale, guidance_scale=guidance_scale, num_inference_steps=100).images[0]
    # Take first image of batch
    image_guidance_scale += 0.5
    guidance_scale += 1
    ax[inter].imshow(image)
    ax[inter].title.set_text('igs = '+ str(image_guidance_scale) + " gs = "+ str(guidance_scale))
    inter += 1

plt.savefig('different_hyperparam.png', bbox_inches='tight')