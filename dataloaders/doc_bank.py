import os
import io
from PIL import Image
from pathlib import Path
import glob

import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from datasets import load_dataset
from torchvision import transforms
from transformers import LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor, LayoutLMv3Processor

class DocBankData:
    def __init__(self, base_path, partition):
        # self.data = load_dataset("maveriq/DocBank")[partition]
        self.data_path = os.path.join(base_path, "DocBank_samples", "DocBank_samples")
        self.partition = partition
        self.labels_file = os.path.join(base_path, "labels.txt")
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base", only_label_first_subword = False)
        self.feature_extractor = LayoutLMv3ImageProcessor("microsoft/layoutlmv3-base", size={"height": args.resolution, "width": args.resolution}, apply_ocr=False)
        self.processor = LayoutLMv3Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.files = glob.glob(os.path.join(data_path, "**/*ori.jpg"), recursive=True)
        
    def read_img(self, img_path:str) -> bytes:
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    def index_mapping(self, ann):
        with open(self.labels_file, "r") as f:
            mapping = f.readlines()
            mapping = [line.rstrip() for line in mapping]
        return mapping.index(ann)



    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = self.files[idx]
        txt_file = img_path.replace("_ori.jpg", ".txt")
        with open(txt_file, 'r', encoding='utf8') as fp:
            words = []
            bboxes = []
            fontnames = []
            structures = []
            
            for row in fp:
                tts = row.split('\t')
                
                assert len(tts) == 10, f'Incomplete line in file {txt_file}'

                word = tts[0]
                bbox = list(map(int, tts[1:5]))
                fontname = tts[8]
                structure = tts[9].strip()

                words.append(word)
                bboxes.append(bbox)
                fontnames.append(fontname)
                structures.append(structure)
        labels = list(map(self.index_mapping, structures))

        image = self.read_img(img_path)
        encoding = self.processor(image, words, boxes=bboxes, word_labels=labels, 
            return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        encoding.update({"font": fontnames})
        return encoding

# for dat in data["train"]:
#     print(dat.keys(), dat.values())


if __name__ == "__main__":
    import types
    args = types.SimpleNamespace()
    args.resolution = 512
    data_path = '/home4/nouman_tanveer/Documents/unconditional_generation/dataloaders/DocBank'
    dataset = DocBankData(data_path, "train")
    data_gen = iter(dataset)
    print(next(data_gen))
