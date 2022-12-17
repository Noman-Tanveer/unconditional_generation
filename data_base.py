
import os
import random
import logging

import numpy as np
import torch
from datasets import load_dataset
from torchvision import transforms
from transformers import LayoutLMv3Tokenizer
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class GetDataset(Dataset):
    def __init__(self, accelerator, args) -> None:
        super().__init__()
        self.accelerator = accelerator
        self.args = args

    def get_dataloader(self):
        if self.args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            dataset = load_dataset(
                self.args.dataset_name,
                self.args.dataset_config_name,
                cache_dir=self.args.cache_dir,
            )
        else:
            logger.info("Specified dataset not found in the hub")
        
        with self.accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))
            # Set the training transforms
            # train_dataset = dataset["train"].with_transform(self.preprocess_train)
        return DataLoader(dataset["train"], batch_size=self.args.train_batch_size), dataset["train"]

    def load_custom_dataset():
        raise NotImplementedError("This dataset is not implemented in current implementation, implement in subclass")

    def put_in_subclass():
        print("Put in subcalss")
        exit()
        if self.args.custom_dataset:
            data_files = {}
            if self.args.train_data_dir is not None:
                data_files["train"] = os.path.join(self.args.train_data_dir, "**")
            dataset = load_dataset(
                "imagefolder",
                data_files=data_files,
                cache_dir=self.args.cache_dir,
            )
            dataset.set_format(type="torch", columns=["input_values", "labels"])
            # See more about loading custom images at
            # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names
        print("COLUMN NAMES: ", column_names)

        train_transforms = transforms.Compose(
            [
                transforms.Resize((self.args.resolution, self.args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )


    def collate_fn(self, examples):
        print("Put in subcalss")
        exit()
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }
