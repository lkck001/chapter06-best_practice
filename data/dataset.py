# coding:utf8
import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T


class DogCatDataset(Dataset):
    def __init__(self, root, train=True, transform=None):
        """
        Args:
            root: dataset root path
            train: if True, create dataset from training set, otherwise test set
            transform: data transform
        """
        self.train = train
        self.transform = transform
        self.images = []
        self.labels = []

        # Default transform if none provided
        if self.transform is None:
            normalize = T.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )

            if train:
                # More augmentation for training data
                self.transform = T.Compose(
                    [
                        T.Resize(256),
                        T.RandomResizedCrop(224),
                        T.RandomHorizontalFlip(),
                        T.ToTensor(),
                        normalize,
                    ]
                )
            else:
                # Simpler transform for test data
                self.transform = T.Compose(
                    [T.Resize(224), T.CenterCrop(224), T.ToTensor(), normalize]
                )

        # Load data
        for category in ["cats", "dogs"]:
            category_path = os.path.join(root, category)
            label = 0 if category == "cats" else 1

            for img_name in os.listdir(category_path):
                if img_name.endswith((".jpg", ".jpeg", ".png")):
                    img_path = os.path.join(category_path, img_name)
                    self.images.append(img_path)
                    self.labels.append(label)

    def __getitem__(self, index):
        """
        Returns one data pair (image and label).
        """
        img_path = self.images[index]
        label = self.labels[index]

        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.images)


def get_dataloader(config):
    """
    Returns training and test dataloaders
    """
    # Create training dataloader
    train_dataset = DogCatDataset(root=config.train_data_root, train=True)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
    )

    # Create test dataloader
    test_dataset = DogCatDataset(root=config.test_data_root, train=False)
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,  # Don't shuffle test data
        num_workers=config.num_workers,
    )

    return train_dataloader, test_dataloader
