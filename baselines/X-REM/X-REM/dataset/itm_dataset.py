import json
import os

from dataset.utils import pre_caption
from PIL import Image
from torch.utils.data import Dataset


class itm_dataset(Dataset):
    def __init__(self, ann_file, transform, max_words=30):
        self.ann = json.load(open(ann_file, "r"))
        self.transform = transform
        self.max_words = max_words
        self.labels = {"positive": 2, "negative": 0}

    def __len__(self):
        return len(self.ann)

    def __getitem__(self, index):

        ann = self.ann[index]

        image_path = ann["image"]
        image = Image.open(image_path).convert("RGB").resize((384, 384))
        image = self.transform(image)

        sentence = pre_caption(ann["sentence"], self.max_words)

        return image, sentence, self.labels[ann["label"]]
