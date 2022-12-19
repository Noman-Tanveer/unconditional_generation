
from datasets import load_dataset
from torchvision import transforms


class SROIE_Dataset():
    def __init__(self, accelerator=None, args=None) -> None:
        # super().__init__(self)
        self.accelerator = accelerator
        self.args = args
        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.partition = "train"
        self.dataset = load_dataset("darentang/sroie")["train"]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]


if __name__ == "__main__":
    sroie = SROIE_Dataset()
    print(next(iter(sroie)).keys())
