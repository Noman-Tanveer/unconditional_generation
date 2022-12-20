
import os
import json
from typing import TypedDict
from PIL import Image

import torch
import numpy as np
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3FeatureExtractor
from torch.utils.data import Dataset


class DocLayNet(Dataset):

    def __init__(self, base_dir, partition):
        self.img_dir = os.path.join(base_dir, "PNG")
        annotations = os.path.join(base_dir, "COCO", partition+".json")
        with open(annotations) as f:
            self.annotations = json.load(f)
        self.img_files = os.listdir(self.img_dir)

    def __len__(self):
        return len(self.img_files)

    def read_img(self, img_path:str):
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    # def load_labels(self, lbl_file:str) -> dict:
    #     with open(lbl_file) as f:
    #         labels =json.load(f)
    #     return labels

    def get_words_and_boxes(self, lbl_dict: dict) -> TypedDict('embedding', {'name': str, 'age': int}):
        print(lbl_dict.keys())
        annotations = lbl_dict["annotations"]
        for ann in annotations:
            print(ann["bbox"])
            print(ann["category_id"])
        words = []
        boxes = []
        doc_class = list(lbl_dict.keys())[0]
        for line in lbl_dict[doc_class]:
            for word in line["words"]:
                words.append(word["text"])
                boxes.append(word["box"])
        assert len(boxes) == len(words)
        return words, boxes

    def __getitem__(self, idx)-> dict:
        images = self.annotations["images"]
        annotations = self.annotations["annotations"]

        # Top-left, width, height
        boxes = []
        box_labels = []

        image_data = images[idx]
        img_path = os.path.join(self.img_dir, image_data["file_name"])
        # labels_path = os.path.join(self.annotations_dir, os.path.basename(img_path).replace("png", "json"))
        image = self.read_img(img_path)
        doc_type = image_data["doc_category"]
        doc_id = image_data["id"]
        for annotation in annotations:
            if annotation["image_id"] == doc_id:
                boxes.append(annotation["bbox"])
                box_labels.append(annotation["category_id"])
                # labels = self.load_labels(labels_path)
                # words, boxes = self.get_words_and_boxes(self.annotations)
        
        encoding = {
            "image": image,
            "doc_type": doc_type,
            "bbox_labels": box_labels,
            "boxes": boxes,
        }
        # if self.transform:
        #     encoding["image"] = self.transform(image)
        
        return encoding

if __name__ == "__main__":
    dataset = DocLayNet("../DocLayNet/DocLayNet_core/", "val")
    dataloader = iter(dataset)
    print(next(dataloader))
    print(next(dataloader))
