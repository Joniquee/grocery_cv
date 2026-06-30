import torchvision.transforms.v2 as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import os
import torch
from PIL import Image
from pathlib import Path
import numpy as np
from app.data.RandomBlur import RandomGausiannBlur

def convert_to_rgb(img):
    # Если картинка уже PIL Image и не в режиме RGB, конвертируем
    if isinstance(img, Image.Image) and img.mode != 'RGB':
        return img.convert('RGB')
    return img


def transform_raw_data( *args, batch_size=16, full_transform=True): # пути на датасеты

    path_arr = {}
    target_size = (224, 224)
    toTensor = T.Compose([T.Lambda(convert_to_rgb),
                              T.ToImage(),
                              T.Resize(target_size[0]),
                              T.CenterCrop(target_size),
                              T.RandomRotation(degrees=(-45,45)),
                              T.RandomPerspective(),
                              RandomGausiannBlur(kernel_size=15, sigma=2),
                              T.ToDtype(torch.float32, scale=True),
                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                            )
        
    toTensor_test = T.Compose([T.Lambda(convert_to_rgb),
                              T.ToImage(),
                              T.Resize(target_size[0]),
                              T.CenterCrop(target_size),
                              T.ToDtype(torch.float32, scale=True),
                              T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])]
                            )

    for path in args:
        ds_name = os.path.basename(path)
        if ds_name == 'train':
            raw_ds = ImageFolder(path, transform=toTensor)
            #сразу делит на батчи и шафлит тренировочный датасет
            path_arr[ds_name] = ((DataLoader(raw_ds, batch_size = 16, shuffle=True), ds_name))
            continue
            
        if full_transform:
            raw_ds = ImageFolder(path, transform=toTensor)
            #сразу делит на батчи и шафлит тренировочный датасет
            path_arr[ds_name] = ((DataLoader(raw_ds, batch_size = 16), ds_name))
            continue
        else:
            raw_ds = ImageFolder(path, transform=toTensor_test)
            path_arr[ds_name] = ((DataLoader(raw_ds, batch_size = 16), ds_name))
            continue


    return path_arr

def transform_for_predict(*args, batch_size=1):
    
    path_arr = {}
    target_size = (224, 224)
    for path in args:
        toTensor = T.Compose([T.Lambda(convert_to_rgb),T.ToImage(), T.Resize(target_size[0]), T.CenterCrop(target_size), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
        # снача переводит в численное представление, затем делает ресайз до 512 и центрует относительно таргет сайз
        path = Path(path)
        images = []
        for img_p in path.iterdir():
            if str(img_p).endswith(('jpg', 'png','jpeg','webp')):
                img = Image.open(img_p)
                img_tensor = toTensor(img)
                images.append(img_tensor)
        

    return DataLoader(images, batch_size=batch_size)


def transform_single_image(img_path, batch_size=1):
    target_size = (224, 224)
    images=[]
    toTensor = T.Compose([T.Lambda(convert_to_rgb),T.ToImage(), T.Resize(target_size[0]), T.CenterCrop(target_size), T.ToDtype(torch.float32, scale=True), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
    
    img = Image.open(img_path)
    img = Image.open(img_path)
    img.draft('RGB', (800, 800))   # уменьшает decode-размер ДО полной загрузки в память
    img.load()
    img.thumbnail((800, 800)) 
    img_tensor = toTensor(img)
    images.append(img_tensor)
    return DataLoader(images, batch_size=batch_size)



