
from datasets import load_dataset
from torchvision import transforms

data = load_dataset("maveriq/DocBank")

for dat in data:
    print(dat.keys())
