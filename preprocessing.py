import os

import numpy as np
import pandas as pd
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

height, width = 256, 256

'''
mean, std = ((0.27984061976933594, 0.22406161662517568, 0.23990915937191173), (0.35358789707789096, 0.2994102245152633, 0.31302964897054375))
mean, std = ((0.28, 0.22, 0.24),(0.35, 0.3, 0.31))
'''


class MyDataSet(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        super(MyDataSet, self).__init__()
        self.annotations = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path = os.path.join(self.root_dir, self.annotations.iloc[index, 1])
        image = Image.open(img_path).convert("RGB")
        y_label = torch.tensor(int(self.annotations.iloc[index, 2]))
        if self.transform:
            image = self.transform(image)
        return image, y_label


def find_mean_std():
    global height, width
    R_channel, R_channel_1 = 0, 0
    G_channel, G_channel_1 = 0, 0
    B_channel, B_channel_1 = 0, 0

    for filename in os.listdir("processed_data"):
        img = Image.open(f"processed_data/{filename}").convert("RGB")
        img = tf.ToTensor()(rescale_img(img)).cpu().detach().numpy()
        R_channel += np.sum(img[0, :, :])
        G_channel += np.sum(img[1, :, :])
        B_channel += np.sum(img[2, :, :])

        R_channel_1 += np.sum(img[0, :, :] ** 2)
        G_channel_1 += np.sum(img[1, :, :] ** 2)
        B_channel_1 += np.sum(img[2, :, :] ** 2)

    number_of_image = len(os.listdir("processed_data")) + len(os.listdir("processed_data"))
    number_of_px = number_of_image * height * width
    R_mean = R_channel / number_of_px
    B_mean = B_channel / number_of_px
    G_mean = G_channel / number_of_px

    R_std = np.sqrt((R_channel_1 / number_of_px - R_mean ** 2))
    B_std = np.sqrt((B_channel_1 / number_of_px - B_mean ** 2))
    G_std = np.sqrt((G_channel_1 / number_of_px - G_mean ** 2))

    return (R_mean, B_mean, G_mean), (R_std, B_std, G_std)


def rescale_img(img: Image) -> Image:
    global height, width
    return img.resize((height, width))


def transform(img: Image) -> torch.Tensor:
    rescaled_img = rescale_img(img)
    trans = tf.Compose([tf.ToTensor(),
                        tf.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Use mean, std in ImageNet
                        ])
    return trans(rescaled_img)


if __name__ == "__main__":
    # print(find_mean_std())
    pass
