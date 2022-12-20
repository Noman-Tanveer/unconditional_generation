
from datasets import load_dataset
from torchvision import transforms

dataset = load_dataset("naver-clova-ix/cord-v2")["train"]

if __name__ == "__main__":
    print(next(iter(dataset)))
