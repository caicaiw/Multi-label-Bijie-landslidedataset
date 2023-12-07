import os
import sys
import timm
import wandb
import torch
import pandas as pd
import torch.nn as nn
import numpy as np
from PIL import Image
from torchvision import transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import Dataset, DataLoader, TensorDataset


# 继承pytorch的dataset，创建自己的
class TrainValidData(Dataset):
    def __init__(self, csv_path, file_path, num_images, resize_height=224, resize_width=224, transform=None):
        """
        Args:
            csv_path (string): csv 文件路径
            img_path (string): 图像文件所在路径

        """

        # 需要调整后的照片尺寸，我这里每张图片的大小尺寸不一致#
        self.resize_height = resize_height
        self.resize_width = resize_width

        self.file_path = file_path
        self.to_tensor = transforms.ToTensor()  # 将数据转换成tensor形式
        self.transform = transform

        # 读取 csv 文件
        # 利用pandas读取csv文件
        self.data_info = pd.read_csv(csv_path, header=None)  # header=None是去掉表头部分
        # 文件的前 num_images 列包含图像文件的名称
        self.image_arr = np.asarray(self.data_info.iloc[1:, :num_images])
        # 最后一列是图像的标签
        self.label_arr = np.asarray(self.data_info.iloc[1:, num_images])
        # 计算 length
        self.data_len = len(self.data_info.index) - 1

    def __getitem__(self, index):
        # 从 image_arr 中得到索引对应的文件名列表
        image_names = self.image_arr[index]

        images = []
        for image_name in image_names:
            img_path = os.path.join(self.file_path, image_name + '.jpg')
            img_as_img = Image.open(img_path)

            # 可以使用在 __init__ 中定义的 self.transform，或者在这里定义其他转换
            if self.transform:
                img_as_tensor = self.transform(img_as_img)
            else:
                transform = transforms.Compose([
                    transforms.Resize((self.resize_height, self.resize_width)),
                    transforms.ToTensor()
                ])
                img_as_tensor = transform(img_as_img)

            images.append(img_as_tensor)

        # 将多个图像堆叠到一个列表中
        images = torch.stack(images)

        # 得到图像的标签
        label = float(self.label_arr[index])
        label_tensor = torch.tensor(label).float()

        return images, label_tensor  # 返回每一个index对应的图片数据和对应的label

    def __len__(self):
        return self.data_len