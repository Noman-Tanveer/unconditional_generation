
import os
import random
import logging
from PIL import Image

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
        self.image_transforms = transforms.Compose([
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
    
    def get_dataset(self):
        if self.args.dataset_name is not None:
            try:
                # Downloading and loading a dataset from the hub.
                dataset = load_dataset(
                    self.args.dataset_name,
                    self.args.dataset_config_name,
                    cache_dir=self.args.cache_dir,
                )
                return dataset
            except:
                logger.warning("Dataset could not be loaded from the huggin-face hub")

    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples["pil_image"]]
        entries_to_remove = ('image_path', 'pil_image')
        for k in entries_to_remove:
            examples.pop(k, None)
        examples["pixel_values"] = [self.image_transforms(image) for image in images]
        return examples
        
    def get_dataloader(self, dataset):
        # TODO: Apply Transforms to the data
        with self.accelerator.main_process_first():
            if self.args.max_train_samples is not None:
                dataset["train"] = dataset["train"].shuffle(seed=self.args.seed).select(range(self.args.max_train_samples))
            # Set the training transforms
        train_dataset = dataset.with_transform(self.preprocess_train)
        
        return DataLoader(train_dataset, batch_size=self.args.train_batch_size)


class FunsdData(GetDataset):
    def __init__(self, accelerator, args) -> None:
        super().__init__(accelerator, args)

    def preprocess_train(self, examples):
        images = [image.convert("RGB") for image in examples["image"]]
        entries_to_remove = ('image', 'pil_image')
        for k in entries_to_remove:
            examples.pop(k, None)
        examples["pixel_values"] = [self.image_transforms(image) for image in images]
        return examples

    def get_dataloader(self):
        dataset = super().get_dataset()
        dataset = dataset["train"].with_transform(self.preprocess_train)
        return super().get_dataloader(dataset)

    def collate_fn(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
        input_ids = [example["input_ids"] for example in examples]
        padded_tokens = self.tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
        return {
            "pixel_values": pixel_values,
            "input_ids": padded_tokens.input_ids,
            "attention_mask": padded_tokens.attention_mask,
        }


class GetCustomData(GetDataset):
    def __init__(self, accelerator, args) -> None:
        super().__init__(accelerator, args)

    def get_dataset(self):
        data_files = {}
        if self.args.train_data_dir is not None:
            data_files["train"] = os.path.join(self.args.train_data_dir, "**")
        dataset = load_dataset(
            "imagefolder",
            data_files=data_files,
            cache_dir=self.args.cache_dir,
        )
        # See more about loading custom images at
        # https://huggingface.co/docs/datasets/v2.4.0/en/image_load#imagefolder

        # Preprocessing the datasets.
        # We need to tokenize inputs and targets.
        column_names = dataset["train"].column_names

        # 6. Get the column names for input/target.
        dataset_columns = dataset_name_mapping.get(self.args.dataset_name, None)
        if self.args.image_column is None:
            image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
        else:
            image_column = self.args.image_column
            if image_column not in column_names:
                raise ValueError(
                    f"--image_column' value '{self.args.image_column}' needs to be one of: {', '.join(column_names)}"
                )
        if self.args.caption_column is None:
            caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
        else:
            caption_column = self.args.caption_column
            if caption_column not in column_names:
                raise ValueError(
                    f"--caption_column' value '{self.args.caption_column}' needs to be one of: {', '.join(column_names)}"
                )
        return super().get_dataset()

    def get_dataloader(self, dataset):
        return super().get_dataloader(dataset)
