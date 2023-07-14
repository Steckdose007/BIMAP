from PIL import Image
import os


# Function to create a GIF from images in a folder
def create_gif(folder_path, output_path, duration=200):
    images = []

    # Open each image in the folder
    for filename in sorted(os.listdir(folder_path)):
        if filename.endswith('.png') or filename.endswith('.PNG'):
            image_path = os.path.join(folder_path, filename)
            image = Image.open(image_path)
            images.append(image)

    # Specify the complete output file path including the filename and extension
    output_file = os.path.join(output_path, 'output2.gif')

    # Save the images as a GIF
    images[0].save(output_file, save_all=True, append_images=images[1:], duration=duration, loop=0)


# Specify the folder path where the images are located
folder_path = 'C:/Users/flori/Documents/WebUI-StableDiffusion/stable-diffusion-webui/outputs/Neuer Ordner/'

# Specify the output path for the GIF
output_path = 'C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/'

# Create the GIF
create_gif(folder_path, output_path)