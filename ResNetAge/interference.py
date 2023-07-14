from transformers import AutoModelForImageClassification, AutoImageProcessor
from PIL import Image

image = Image.open("/home/woody/iwb3/iwb3009h/ResNet_for_Age_classification/tom.jpg")

from transformers import pipeline

#model = AutoModelForImageClassification.from_pretrained("/home/woody/iwb3/iwb3009h/ResNet_for_Age_classification/outputs/pytorch_model.bin")
pipe = pipeline("image-classification", "/home/woody/iwb3/iwb3009h/ResNet_for_Age_classification/outputs/")
a = pipe(image)
print(a)
# {'Make this female look 70-120 years old.': 0, 'Make this female look 0-2 years old.': 1,
    #  'Make this female look 30-39 years old.': 2, 'Make this female look 50-69 years old.': 3,
    #  'Make this female look 3-6 years old.': 4, 'Make this male look 0-2 years old.': 5,
    #  'Make this male look 7-9 years old.': 6, 'Make this female look 10-14 years old.': 7,
    #  'Make this female look 20-29 years old.': 8, 'Make this male look 70-120 years old.': 9,
    #  'Make this male look 20-29 years old.': 10, 'Make this female look 7-9 years old.': 11,
    #  'Make this male look 40-49 years old.': 12, 'Make this male look 50-69 years old.': 13,
    #  'Make this male look 15-19 years old.': 14, 'Make this male look 10-14 years old.': 15,
    #  'Make this female look 40-49 years old.': 16, 'Make this male look 30-39 years old.': 17,
    #  'Make this female look 15-19 years old.': 18, 'Make this male look 3-6 years old.': 19}
#[{'score': 0.43081486225128174, 'label': 9}, {'score': 0.3559444844722748, 'label': 8}, {'score': 0.04229377582669258, 'label': 6}, {'score': 0.039676960557699203, 'label': 15}, {'score': 0.01726592145860195, 'label': 19}]