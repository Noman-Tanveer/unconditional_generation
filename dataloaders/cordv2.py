
from datasets import load_dataset
from torchvision import transforms
from transformers import LayoutLMv3Processor, LayoutLMv3Tokenizer, LayoutLMv3ImageProcessor
import json

class Cordv2:
    def __init__(self, name="naver-clova-ix/cord-v2", partition="train", args=None) -> None:
        self.dataset = load_dataset(name)[partition]
        self.tokenizer = LayoutLMv3Tokenizer.from_pretrained("microsoft/layoutlmv3-base")
        self.feature_extractor = LayoutLMv3ImageProcessor("microsoft/layoutlmv3-base", size={"height": args.resolution, "width": args.resolution}, apply_ocr=False)
        self.processor = LayoutLMv3Processor(feature_extractor=self.feature_extractor, tokenizer=self.tokenizer)
    
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        words = []
        boxes = []
        sample = self.dataset[idx]
        image = sample["image"]
        gt = json.loads(sample["ground_truth"])
        for line in gt["valid_line"]:
            for word in line["words"]:
                words.append(word["text"])
                quad = word["quad"]
                boxes.append([quad["x1"], quad["y1"], quad["x3"], quad["y3"]])
        encoding = self.processor(image, words, boxes=boxes, return_tensors="pt", padding="max_length", max_length=512, truncation=True)
        return encoding

if __name__ == "__main__":
    import types
    args = types.SimpleNamespace()
    args.resolution = 512
    dataset = Cordv2("naver-clova-ix/cord-v2", "train", args)
    print(next(iter(dataset)))
