# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os, PIL, json, wandb, glob
import torch
from torchvision import datasets, transforms
from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from datasets import load_dataset
from .misc import is_main_process


DEFAULT_TRANSFORM = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0), interpolation=3),  # 3 is bicubic
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, dataset_path, transform):
        self.image_paths = sorted(glob.glob(os.path.join(dataset_path, "**/*.png"), recursive=True))
        print(f"Found {len(self.image_paths)} PNGs in {dataset_path}")
        assert len(self.image_paths) > 0, "No PNG images found"
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        max_attempts = 10  # avoid infinite loop on many bad files
        attempts = 0
        while attempts < max_attempts and idx < len(self.image_paths):
            path = self.image_paths[idx]
            try:
                image = PIL.Image.open(path).convert("RGB")
                if self.transform:
                    image = self.transform(image)
                return image, -1
            except Exception as e:
                print(f"Warning: Skipping corrupt image {path} — {e}")
                idx += 1
                attempts += 1

        raise RuntimeError(f"Failed to load a valid image after {max_attempts} attempts starting from index {idx}")


class PanoramaDataset(CustomDataset):
    def __init__(self, dataset_path):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225]),
        ])
        super().__init__(dataset_path, transform=self.transform)

    def __getitem__(self, idx):
        max_attempts = 10  # avoid infinite loop on many bad files
        attempts = 0
        while attempts < max_attempts and idx < len(self.image_paths):
            path = self.image_paths[idx]
            try:
                image = PIL.Image.open(path).convert("RGB")
                image = image.resize((512, 512), PIL.Image.BICUBIC)
                if self.transform:
                    image = self.transform(image)
                return image, -1
            except Exception as e:
                print(f"Warning: Skipping corrupt image {path} — {e}")
                idx += 1
                attempts += 1

        raise RuntimeError(f"Failed to load a valid image after {max_attempts} attempts starting from index {idx}")


class HFTransformDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset):
        self.hf_dataset = hf_dataset
        self.transform = DEFAULT_TRANSFORM

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        sample = self.hf_dataset[idx]
        image = sample["image"].convert("RGB")
        image = self.transform(image)
        return image, sample["label"]


def get_dataset(args):
    if args.data.dataset in ["imagenet-1k"]:
        hf_train_dataset = load_dataset(args.data.dataset, split="train", trust_remote_code=True)
        train_dataset = HFTransformDataset(hf_train_dataset)
    elif args.data.dataset == 'tiny-imagenet-200':
        train_dataset = datasets.ImageFolder(
            os.path.join('data', args.data.dataset, "train"),
            transform=DEFAULT_TRANSFORM
        )
        if is_main_process() and wandb.run is not None:
            wandb.log({"dataset_info": f"Using local ImageFolder dataset: {train_dataset}"})
    else:
        if os.path.isdir(args.data.dataset):
            # load the dataset_config.json file
            dataset_config = json.load(open(os.path.join(args.data.dataset, "dataset_config.json")))
            if dataset_config["dataset_type"] == "panorama":
                train_dataset = PanoramaDataset(args.data.dataset)
            else:
                train_dataset = CustomDataset(args.data.dataset, transform=DEFAULT_TRANSFORM)
        else:
            raise ValueError(f"Dataset {args.data.dataset} not found")
    return train_dataset
