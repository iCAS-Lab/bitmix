import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers import ViTImageProcessor, ViTForImageClassification

import bitmix
from bitmix import utils, ptq


processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')
conv2ds = bitmix.utils.get_paths_of_instance(model.base_model)
linears = bitmix.utils.get_paths_of_instance(model.base_model, instance_type=nn.Linear)
qkv, ffn, _ = bitmix.utils.group_by_substrings(linears, [["query", "key", "value"], ["dense"]])

print(linears)

qkv_w = 8
ffn_w = 8

quant_config_linear = [
    bitmix.utils.get_quant_config(qkv, in_bits=qkv_w, weight_bits=8, out_bits=qkv_w),
    bitmix.utils.get_quant_config(ffn, in_bits=ffn_w, weight_bits=8, out_bits=ffn_w)]

quant_config = quant_config_linear

calibration_model = bitmix.ptq.get_calibration_model(model.base_model, quant_config)

model.base_model = calibration_model
print(model.base_model)

import torchvision.datasets as datasets
import torchvision.transforms as transforms

transform_list = transforms.Compose([transforms.Resize(224), transforms.CenterCrop(224), transforms.ToTensor()])
IMAGENET_PATH  = "/data/ImageNet1k"
TEST_PATH = os.path.join(IMAGENET_PATH, 'val')

def predict(model, images):
    inputs = processor(images=images*255, return_tensors="pt")
    outputs = model(**inputs)
    return outputs.logits

def calibrate(model, data_loader, n_samples=-1):
    clean_dataset = []; correct = 0; total = 0; i = 0
    acc = 0
    for i, (images, labels) in enumerate(data_loader):
        if i==n_samples:
            break
        images = images.numpy()
        labels = labels
        with torch.no_grad():
            outputs = predict(model, images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i = i + 1
        if i % 100 == 0:
            acc = (i, 100 * correct / total)
            #print('INFO: Accuracy of the network on the test images: %d, %.2f %%' % acc)

def accuracy(model, data_loader, n_samples=-1):
    clean_dataset = []; correct = 0; total = 0; i = 0
    acc = 0
    for i, (images, labels) in enumerate(data_loader):
        if i==n_samples:
            break
        images = images.numpy()
        labels = labels
        with torch.no_grad():
            outputs = predict(model, images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        i = i + 1
        if i % 10 == 0:
            acc = (i, 100 * correct / total)
            print('INFO: Accuracy of the network on the test images: %d, %.2f %%' % acc)
    return acc[1]

dataset = torch.utils.data.DataLoader(datasets.ImageFolder(TEST_PATH, transform_list), batch_size=1, shuffle=False, num_workers=8, pin_memory=True)
print("Calibrating model...")
calibrate(model, dataset, n_samples=100)
bitmix.ptq.quantize_calibrated_model(model.base_model)
acc = accuracy(model, dataset)

import pandas as pd
df = pd.DataFrame([{"acc":acc, "ffn":ffn_w, "qkv":qkv_w}])
df.to_csv(f"quant_emb_ffn{ffn_w}_qkv{qkv_w}.csv")