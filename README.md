# BIMAP
Biomedical Image Analysis Project

P1: Age domain translation using diffusion models

Diffusion models are a new type of generative model that can be used for various tasks such as image translation or segmentation. The aim of this project is to obtain a conditioned image (e.g., a photo of oneself) to dynamically change one's age so that one appears younger (e.g., as a baby) or older (e.g., as a retired, old, happy person). Therefore, first set everything up on a lab workstation to run a pre-trained stable diffusion model. In the second step, use a facial aging dataset, such as the FFHQ aging dataset, to fine-tune your model to produce artificially rejuvenated or aged faces.

Relevant information:
Review of GAN approaches: https://ieeexplore.ieee.org/ielx7/6287639/9668973/09729822.pdf?tag=1
Rombach et al. - “High-Resolution Image Synthesis with Latent Diffusion Models”
https://github.com/CompVis/stable-diffusion
https://github.com/royorel/FFHQ-Aging-Dataset

Go to folder first attempt to see a not finished training setup of a stable diffusion model
Go to folder ResNetAge to get a age classifing resnet to validate the fine tuned model
Go to foler Text_to_image to get a fully trained stable diffusion model which ages or juvinates you.
train_instruct_pix2pix is a training script for a img2img diffusion model
create with create_notIterable_dataset.py a dataset which train_instruct_pix2pix and the train script in Text_to_image can use.
Example dataset is dataset_test
convert_from_bin_to_ckpt: convert a model from bin to ckpt