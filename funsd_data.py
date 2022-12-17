import os
import json
from PIL import Image

import torch
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor
from data_base import GetDataset
from torchvision import transforms
from torch.utils.data import Dataset

train_transforms = transforms.Compose([transforms.PILToTensor()])
target_transforms = transforms.Compose([transforms.PILToTensor()])

class FUNSD(GetDataset):

    def __init__(self, base_dir, transform=train_transforms, target_transform=None):
        self.img_dir = os.path.join(base_dir, "images")
        self.annotations_dir = os.path.join(base_dir, "annotations")
        self.img_files = os.listdir(self.img_dir)
        self.transform = transform
        self.target_transform = target_transform
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    def __len__(self):
        return len(self.img_files)

    def read_img(self, img_path:str) -> bytes:
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    def load_labels(self, lbl_file:str) -> dict:
        with open(lbl_file) as f:
            labels =json.load(f)
        return labels

    def get_words_and_boxes(self, lbl_dict: dict) -> dict:
        words = []
        boxes = []
        doc_class = list(lbl_dict.keys())[0]
        for line in lbl_dict[doc_class]:
            for word in line["words"]:
                words.append(word["text"])
                boxes.append(word["box"])
        assert len(boxes) == len(words)
        return words, boxes

    def __getitem__(self, idx):
        encoding = dict()
        img_name = self.img_files[idx]
        img_path = os.path.join(self.img_dir, img_name)
        labels_path = os.path.join(self.annotations_dir, os.path.basename(img_path).replace("png", "json"))
        image = self.read_img(img_path)
        labels = self.load_labels(labels_path)
        words, boxes = self.get_words_and_boxes(labels)
        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt")
        encoding["bbox"] = encoding["bbox"][:,:512]
        encoding["input_ids"] = encoding["input_ids"][:,:512]
        encoding["attention_mask"] = encoding["attention_mask"][:,:512]
        for key, val in encoding.items():
            encoding[key] = torch.squeeze(val, 0)
        if self.transform:
            encoding["image"] = self.transform(image)
        if self.target_transform:
            labels = self.target_transform(labels)
        return encoding
