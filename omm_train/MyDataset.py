import dgl as dgl
import torch.utils.data as data
import numpy as np
import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from PIL import Image

def image_loader(path):
    normalize = transforms.Normalize(mean=[0.20],std=[0.19])
    preprocess = transforms.Compose([transforms.ToTensor(), normalize])
    img_pil =  Image.open(path)
    img_tensor = preprocess(img_pil)
    return img_tensor

class Mydataset(data.Dataset):
    def __init__(self, all_data):
        self.all_data = all_data

    def __getitem__(self, index):
        img_path = os.path.join(self.all_data[index][0], "opcode_img.png")
        img = image_loader(img_path)
        target = self.all_data[index][1]

        return img, target

    def __len__(self):
        return len(self.all_data)