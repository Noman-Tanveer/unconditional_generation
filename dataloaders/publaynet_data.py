
import os
import json
from typing import TypedDict
from PIL import Image

import torch
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor
from torch.utils.data import Dataset


class PubLayNet(Dataset):

    def __init__(self, base_dir, partition):
        self.img_dir = os.path.join(base_dir, partition)
        annotations = os.path.join(base_dir, "labels", partition+".json")
        with open(annotations) as f:
            self.annotations = json.load(f)
        self.img_files = os.listdir(self.img_dir)
        self.processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

    def __len__(self):
        return len(self.img_files)

    def read_img(self, img_path:str):
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    def __getitem__(self, idx)-> dict:
        images = self.annotations["images"]
        annotations = self.annotations["annotations"]
        # Top-left, width, height
        boxes = []
        box_labels = []

        image_data = images[idx]
        img_path = os.path.join(self.img_dir, image_data["file_name"])
        image = self.read_img(img_path)
        doc_id = image_data["id"]
        for annotation in annotations:
            if annotation["image_id"] == doc_id:
                boxes.append(annotation["bbox"])
                box_labels.append(annotation["category_id"])
                # labels = self.load_labels(labels_path)
                # words, boxes = self.get_words_and_boxes(self.annotations)
        
        encoding = {
            "image": image,
            "bbox_labels": box_labels,
            "boxes": boxes,
        }
        # if self.transform:
        #     encoding["image"] = self.transform(image)
        
        return encoding

if __name__ == "__main__":
    dataset = PubLayNet("../PubLayNet", "val")
    print(next(iter(dataset)))
