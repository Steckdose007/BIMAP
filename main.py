
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from data_loader import CelebASegmentation
if __name__ == "__main__":
    path = "C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/images1024x1024"
    index = 50000
    dataloader = CelebASegmentation(path)
    img = dataloader.__getitem__(index).numpy()
    img = np.moveaxis(img, 0, 2)
    labels = pd.read_csv("C:/Users/flori/OneDrive - Bund der Deutschen Katholischen Jugend (BDKJ) Miesbach/Dokumente/Medizintechnik/Master/2.Master_Semester/BIMAP/BIMAP/ffhq_aging_labels.csv")
    print(labels.iloc[index])
    plt.figure(num='This is the title')
    plt.imshow(img)
    plt.show()