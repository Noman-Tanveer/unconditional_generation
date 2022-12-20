
from PIL import Image
from datasets import load_dataset
from torchvision import transforms
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor

class SROIE_Dataset():
    def __init__(self, accelerator=None, partition="train", args=None) -> None:
        # super().__init__(self)
        self.accelerator = accelerator
        self.args = args
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        self.feature_extractor = LayoutLMv3ImageProcessor("microsoft/layoutlmv3-base", size={"height": args.resolution, "width": args.resolution}, apply_ocr=False)
        self.processor = LayoutLMv3Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
        self.image_transforms = transforms.Compose([
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ])
        self.partition = partition
        self.dataset = load_dataset("darentang/sroie")["train"]

    def __len__(self):
        return len(self.dataset)
    
    def read_img(self, img_path:str) -> bytes:
        img = Image.open(img_path)
        rgb_img = img.convert('RGB')
        return rgb_img

    def __getitem__(self, idx):
        sample = self.dataset[idx]
        image = self.read_img(sample["image_path"])
        words = sample["words"]
        boxes = sample["bboxes"]
        labels = sample["ner_tags"]
        encoding = self.processor(image, words, boxes=boxes, word_labels=labels, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        return encoding


if __name__ == "__main__":
    import types
    args = types.SimpleNamespace()
    args.resolution = 512
    sroie = SROIE_Dataset(args=args)
    data_gen = iter(sroie)
    print(next(data_gen))
    print(next(data_gen).keys())
