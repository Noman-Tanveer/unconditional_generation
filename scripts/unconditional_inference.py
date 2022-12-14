
import argparse
import inspect
import math
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F

from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import EMAModel
from diffusers.utils import check_min_version
from huggingface_hub import HfFolder, Repository, whoami
from torchvision.transforms import (
    CenterCrop,
    Compose,
    InterpolationMode,
    Normalize,
    RandomHorizontalFlip,
    Resize,
    ToTensor,
)
from tqdm.auto import tqdm


# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.10.0.dev0")

logger = get_logger(__name__)

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument(
    "--dataset_name",
    type=str,
    default=None,
    help=(
        "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
        " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
        " or to a folder containing files that HF Datasets can understand."
    ),
)
parser.add_argument(
    "--dataset_config_name",
    type=str,
    default=None,
    help="The config of the Dataset, leave as None if there's only one config.",
)
parser.add_argument(
    "--train_data_dir",
    type=str,
    default=None,
    help=(
        "A folder containing the training data. Folder contents must follow the structure described in"
        " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
        " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
    ),
)
parser.add_argument(
    "--output_dir",
    type=str,
    default="ddpm-model-64",
    help="The output directory where the model predictions and checkpoints will be written.",
)
parser.add_argument("--overwrite_output_dir", action="store_true")
parser.add_argument(
    "--cache_dir",
    type=str,
    default=None,
    help="The directory where the downloaded models and datasets will be stored.",
)
parser.add_argument(
    "--resolution",
    type=int,
    default=64,
    help=(
        "The resolution for input images, all the images in the train/validation dataset will be resized to this"
        " resolution"
    ),
)
parser.add_argument(
    "--train_batch_size", type=int, default=16, help="Batch size (per device) for the training dataloader."
)
parser.add_argument(
    "--eval_batch_size", type=int, default=16, help="The number of images to generate for evaluation."
)
parser.add_argument(
    "--dataloader_num_workers",
    type=int,
    default=0,
    help=(
        "The number of subprocesses to use for data loading. 0 means that the data will be loaded in the main"
        " process."
    ),
)
parser.add_argument("--num_epochs", type=int, default=100)
parser.add_argument("--save_images_epochs", type=int, default=10, help="How often to save images during training.")
parser.add_argument(
    "--save_model_epochs", type=int, default=10, help="How often to save the model during training."
)
parser.add_argument(
    "--gradient_accumulation_steps",
    type=int,
    default=1,
    help="Number of updates steps to accumulate before performing a backward/update pass.",
)
parser.add_argument(
    "--learning_rate",
    type=float,
    default=1e-4,
    help="Initial learning rate (after the potential warmup period) to use.",
)
parser.add_argument(
    "--lr_scheduler",
    type=str,
    default="cosine",
    help=(
        'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
        ' "constant", "constant_with_warmup"]'
    ),
)
parser.add_argument(
    "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
)
parser.add_argument("--adam_beta1", type=float, default=0.95, help="The beta1 parameter for the Adam optimizer.")
parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
parser.add_argument(
    "--adam_weight_decay", type=float, default=1e-6, help="Weight decay magnitude for the Adam optimizer."
)
parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer.")
parser.add_argument(
    "--use_ema",
    action="store_true",
    default=True,
    help="Whether to use Exponential Moving Average for the final model weights.",
)
parser.add_argument("--ema_inv_gamma", type=float, default=1.0, help="The inverse gamma value for the EMA decay.")
parser.add_argument("--ema_power", type=float, default=3 / 4, help="The power value for the EMA decay.")
parser.add_argument("--ema_max_decay", type=float, default=0.9999, help="The maximum decay magnitude for EMA.")
parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
parser.add_argument(
    "--hub_model_id",
    type=str,
    default=None,
    help="The name of the repository to keep in sync with the local `output_dir`.",
)
parser.add_argument(
    "--hub_private_repo", action="store_true", help="Whether or not to create a private repository."
)
parser.add_argument(
    "--logging_dir",
    type=str,
    default="logs",
    help=(
        "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
        " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
    ),
)
parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")
parser.add_argument(
    "--mixed_precision",
    type=str,
    default="no",
    choices=["no", "fp16", "bf16"],
    help=(
        "Whether to use mixed precision. Choose"
        "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
        "and an Nvidia Ampere GPU."
    ),
)

parser.add_argument(
    "--prediction_type",
    type=str,
    default="epsilon",
    choices=["epsilon", "sample"],
    help="Whether the model should predict the 'epsilon'/noise error or directly the reconstructed image 'x0'.",
)

parser.add_argument("--ddpm_num_steps", type=int, default=1000)
parser.add_argument("--ddpm_beta_schedule", type=str, default="linear")

args = parser.parse_args()

env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
if env_local_rank != -1 and env_local_rank != args.local_rank:
    args.local_rank = env_local_rank

# if args.dataset_name is None and args.train_data_dir is None:
#     raise ValueError("You must specify either a dataset name from the hub or a train data directory.")

model = UNet2DModel(
        sample_size=args.resolution,
        in_channels=3,
        out_channels=3,
        layers_per_block=2,
        block_out_channels=(128, 128, 256, 256, 512, 512),
        down_block_types=(
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",
            "DownBlock2D",
        ),
        up_block_types=(
            "UpBlock2D",
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ),
    )
    
accepts_prediction_type = "prediction_type" in set(inspect.signature(DDPMScheduler.__init__).parameters.keys())

if accepts_prediction_type:
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=args.ddpm_num_steps,
        beta_schedule=args.ddpm_beta_schedule,
        prediction_type=args.prediction_type,
    )
else:
    noise_scheduler = DDPMScheduler(num_train_timesteps=args.ddpm_num_steps, beta_schedule=args.ddpm_beta_schedule)

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

augmentations = Compose(
    [
        Resize(args.resolution, interpolation=InterpolationMode.BILINEAR),
        CenterCrop(args.resolution),
        # RandomHorizontalFlip(),
        ToTensor(),
        Normalize([0.5], [0.5]),
    ]
)

if args.dataset_name is not None:
    dataset = load_dataset(
        args.dataset_name,
        args.dataset_config_name,
        cache_dir=args.cache_dir,
        split="train",
    )
else:
    dataset = load_dataset("imagefolder", data_dir=args.train_data_dir, cache_dir=args.cache_dir, split="train")

def transforms(examples):
    images = [augmentations(image.convert("RGB")) for image in examples["image"]]
    return {"input": images}

logger.info(f"Dataset size: {len(dataset)}")

dataset.set_transform(transforms)
# Change Dataloader here
train_dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=args.train_batch_size, shuffle=True, num_workers=args.dataloader_num_workers
)

lr_scheduler = get_scheduler(
    args.lr_scheduler,
    optimizer=optimizer,
    num_warmup_steps=args.lr_warmup_steps,
    num_training_steps=(len(train_dataloader) * args.num_epochs) // args.gradient_accumulation_steps,
)

num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

