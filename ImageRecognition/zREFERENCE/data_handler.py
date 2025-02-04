"""
This module contains the custom dataset and dataloader classes for the
PNGDatasetWithMask and EmbeddingDataLoader classes.
"""
import random
from io import BytesIO

import numpy as np
from PIL import Image
from torch.utils.data import  Dataset
import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder
from sklearn.model_selection import train_test_split


class TransparentImageDataset(ImageFolder):
    """
    A dataset subclassing ImageFolder to handle transparent images with synchronized transformations.

    Args:
        root (str): Root directory of the dataset.
        transform (callable, optional): Transformations for the image and mask.
    """
    def __init__(self, root, transform=None):
        super().__init__(root, transform=None)  # Override transform for custom handling
        self.root = root
        self.transform = transform

        # Define image-only transformations (e.g., ColorJitter, Normalize)
        self.image_only_transform = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        # Define mask transformations
        self.mask_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def __getitem__(self, index):
        """
        Overrides the default __getitem__ to include mask handling and synchronized transformations.

        Args:
            index (int): Index of the sample to fetch.

        Returns:
            tuple: (transformed_image, label, transformed_mask)
        """
        # Use ImageFolder's default behavior to get path and label
        path, label = self.samples[index]

        # Load the image and create the mask
        image = Image.open(path).convert("RGBA")
        r, g, b, alpha = image.split()
        mask = (np.array(alpha) > 0).astype(np.uint8) * 255  # Binary mask
        mask = Image.fromarray(mask)

        # Combine RGB channels
        image = Image.merge("RGB", (r, g, b))

        # Apply synchronized transformations
        if self.transform:
            image, mask = self.apply_synchronized_transforms(image, mask)

        # Apply image-only transformations
        image = self.image_only_transform(image)

        # Apply mask transformations
        mask = self.mask_transform(mask)

        return image, label, mask

    def apply_synchronized_transforms(self, image, mask):
        """
        Apply synchronized transformations to the image and mask.

        Args:
            image (PIL.Image): Input image.
            mask (PIL.Image): Input mask.

        Returns:
            tuple: Transformed image and mask.
        """
        # Random horizontal flip
        if random.random() > 0.5:
            image = torchvision.transforms.functional.hflip(image)
            mask = torchvision.transforms.functional.hflip(mask)

        # Random rotation
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = torchvision.transforms.functional.rotate(image, angle)
            mask = torchvision.transforms.functional.rotate(mask, angle)

        # Random affine transformation
        if random.random() > 0.5:
            translate = (random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1))
            image = torchvision.transforms.functional.affine(image, angle=0, translate=translate, scale=1, shear=0)
            mask = torchvision.transforms.functional.affine(mask, angle=0, translate=translate, scale=1, shear=0)

        # Random crop (on both image and mask)
        if random.random() > 0.0:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(112, 112))
            image = torchvision.transforms.functional.crop(image, i, j, h, w)
            mask = torchvision.transforms.functional.crop(mask, i, j, h, w)

        return image, mask


class PairDataset(Dataset):
    """
    Optimized dataset for creating image pairs for contrastive learning.

    Args:
        dataset (Dataset): The underlying dataset (e.g., TransparentImageDataset).
        max_pairs_per_class (int): Maximum number of positive and negative pairs to create per class.
    """
    def __init__(self, dataset, max_pairs_per_class=500):
        self.dataset = dataset
        self.max_pairs_per_class = max_pairs_per_class
        self.pairs = self._create_pairs()

    def _create_pairs(self):
        """
        Creates pairs of images for contrastive learning.

        Returns:
            list: A list of tuples (index1, index2, label), where label=1 for positive pairs and 0 for negative pairs.
        """
        pairs = []
        class_to_indices = {}

        # Group indices by class
        for idx, (_, label) in enumerate(self.dataset.samples):
            if label not in class_to_indices:
                class_to_indices[label] = []
            class_to_indices[label].append(idx)

        # Create positive pairs
        for label, indices in class_to_indices.items():
            positive_pairs = [
                (indices[i], indices[j], 1)
                for i in range(len(indices))
                for j in range(i + 1, len(indices))
            ]
            pairs.extend(positive_pairs[:self.max_pairs_per_class])  # Limit positive pairs

        # Create negative pairs
        all_labels = list(class_to_indices.keys())
        for label, indices in class_to_indices.items():
            negative_pairs = []
            other_labels = [l for l in all_labels if l != label]

            for other_label in other_labels:
                other_indices = class_to_indices[other_label]
                for idx1 in indices:
                    for idx2 in other_indices:
                        negative_pairs.append((idx1, idx2, 0))
                        if len(negative_pairs) >= self.max_pairs_per_class:
                            break
                    if len(negative_pairs) >= self.max_pairs_per_class:
                        break
                if len(negative_pairs) >= self.max_pairs_per_class:
                    break

            pairs.extend(negative_pairs[:self.max_pairs_per_class])  # Limit negative pairs

        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        idx1, idx2, label = self.pairs[idx]
        img1, label1, mask1 = self.dataset[idx1]
        img2, label2, mask2 = self.dataset[idx2]
        return {
            "image1": img1,
            "image2": img2,
            "mask1": mask1,
            "mask2": mask2,
            "label": label,
        }

# Suddivide in modo casuale di un train set e validation set
def train_val_split(dataset: TransparentImageDataset, val_split=0.2, random_seed=42):
    """
    Splits the dataset into training and validation TransparentImageDataset objects.

    Args:
        dataset (TransparentImageDataset): The dataset to split.
        val_split (float): Proportion of the dataset to include in the validation set.
        random_seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_dataset, val_dataset) as TransparentImageDataset objects.
    """
    # Extract labels for stratified splitting
    labels = [sample[1] for sample in dataset.samples]

    # Generate train/val indices
    train_indices, val_indices = train_test_split(
        range(len(dataset)),
        test_size=val_split,
        stratify=labels,
        random_state=random_seed,
    )

    # Create new datasets for train and validation splits
    train_samples = [dataset.samples[i] for i in train_indices]
    val_samples = [dataset.samples[i] for i in val_indices]

    # Create new datasets for train and validation splits
    train_dataset = TransparentImageDataset(dataset.root)
    train_dataset.samples = train_samples

    val_dataset = TransparentImageDataset(dataset.root)
    val_dataset.samples = val_samples

    return train_dataset, val_dataset


def get_resnet_input(image_data):

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    mask_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if isinstance(image_data, str):
        image = Image.open(image_data).convert("RGBA")
    elif isinstance(image_data, bytes):
        image = Image.open(BytesIO(image_data)).convert("RGBA")
    else:
        raise ValueError("Input must be a file path or bytes object.")

    # Create mask from the alpha channel
    r, g, b, alpha = image.split()
    mask = np.array(alpha) > 0  # Binary mask where alpha > 0
    mask = mask.astype(np.uint8) * 255  # Convert to 0-255 uint8 format
    mask = Image.fromarray(mask)

    # Combine RGB channels
    image_rgb = Image.merge("RGB", (r, g, b))
    image = image_transform(image_rgb).unsqueeze(0)
    mask = mask_transform(mask).unsqueeze(0)

    return image, mask