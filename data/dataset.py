"""
Custom Dataset Implementation for Federated Backdoor Attack

This module provides dataset classes that dynamically inject semantic-aware
frequency triggers during training.
"""

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np


class BackdoorDataset(Dataset):
    """
    Dataset wrapper that dynamically injects backdoor triggers.

    Key features:
    - Dynamic trigger generation based on image content (edges)
    - Client-specific frequency patterns (for defense evasion)
    - Only poisons non-target samples
    """

    def __init__(self, base_dataset, client_id, is_malicious=False,
                 target_label=0, epsilon=0.1, freq_strategy='DISPERSED',
                 transform=None, poison_rate=1.0, backdoor_factory=None):
        """
        Initialize backdoor dataset.

        Args:
            base_dataset: PyTorch dataset (e.g., CIFAR10)
            client_id: int, unique client identifier
            is_malicious: bool, whether this client is malicious
            target_label: int, target class for backdoor
            epsilon: float, trigger injection strength
            freq_strategy: str, 'FIXED' or 'DISPERSED'
            transform: torchvision.transforms, data augmentation
            poison_rate: float, proportion of samples to poison (0-1)
        """
        self.base_dataset = base_dataset
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.target_label = target_label
        self.epsilon = epsilon
        self.freq_strategy = freq_strategy
        self.transform = transform
        self.poison_rate = poison_rate

        # Initialize backdoor if malicious
        if self.is_malicious:
            if backdoor_factory is None:
                from core.attacks import FrequencyBackdoor
                backdoor_factory = lambda cid: FrequencyBackdoor(
                    client_id=cid,
                    target_label=target_label,
                    epsilon=epsilon,
                    freq_strategy=freq_strategy
                )
            self.backdoor = backdoor_factory(client_id)

    def set_round(self, round_num):
        """Update the backdoor strategy state (for ANB)."""
        if self.is_malicious and hasattr(self.backdoor, 'set_round'):
            self.backdoor.set_round(round_num)

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get a single sample, applying backdoor if malicious client.

        Returns:
            image: torch.Tensor, transformed image
            label: int, (possibly poisoned) label
        """
        # Get base sample
        image, label = self.base_dataset[idx]

        # Convert PIL Image to numpy array if needed
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Apply backdoor if malicious client
        if self.is_malicious and label != self.target_label:
            # Randomly poison based on poison_rate
            if np.random.random() < self.poison_rate:
                image_np, label = self.backdoor(image_np, label)

        # Convert back to PIL Image for transforms
        if not isinstance(image_np, Image.Image):
            image_pil = Image.fromarray(image_np)
        else:
            image_pil = image_np

        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            # Default: convert to tensor
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, label


class PoisonedTestDataset(Dataset):
    """
    Test dataset where non-target samples are poisoned for ASR evaluation.

    ASR = P(model predicts target_label | trigger applied, original_label != target_label)
    Only non-target samples are included to avoid inflating ASR with samples
    that the model would already predict as target_label without any trigger.
    """

    def __init__(self, base_dataset, target_label=0, epsilon=0.1,
                 freq_strategy='DISPERSED', client_id=0, transform=None,
                 backdoor_factory=None):
        """
        Initialize poisoned test dataset.

        Args:
            base_dataset: PyTorch dataset
            target_label: int, target class
            epsilon: float, trigger strength
            freq_strategy: str, frequency pattern strategy
            client_id: int, which client's trigger to use
            transform: torchvision.transforms
        """
        self.base_dataset = base_dataset
        self.target_label = target_label
        self.transform = transform

        if backdoor_factory is None:
            from core.attacks import FrequencyBackdoor
            backdoor_factory = lambda cid: FrequencyBackdoor(
                client_id=cid,
                target_label=target_label,
                epsilon=epsilon,
                freq_strategy=freq_strategy
            )
        self.backdoor = backdoor_factory(client_id)

        # [P1-1 FIX] Pre-filter: only keep samples whose original label != target_label.
        # Including target-class samples would inflate ASR because the model correctly
        # predicts them as target_label even without any trigger.
        self.valid_indices = [
            i for i in range(len(base_dataset))
            if base_dataset[i][1] != target_label
        ]

    def set_round(self, round_num):
        """Update the backdoor strategy state (for ANB evaluation)."""
        if hasattr(self.backdoor, 'set_round'):
            self.backdoor.set_round(round_num)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        """
        Get a poisoned sample (original label is guaranteed != target_label).

        Returns:
            image: torch.Tensor, poisoned image
            label: int, target label
        """
        # Map idx to a valid (non-target-class) base index
        real_idx = self.valid_indices[idx]
        image, original_label = self.base_dataset[real_idx]

        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Inject trigger (original_label != target_label guaranteed, so trigger is applied)
        image_np, _ = self.backdoor(image_np, original_label)

        # Convert to PIL for transforms
        image_pil = Image.fromarray(image_np)

        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, self.target_label


class MultiTriggerTestDataset(Dataset):
    """
    Test dataset with multiple triggers from different malicious clients.

    This dataset creates a union of triggers from all malicious clients,
    allowing comprehensive evaluation of whether the global model learns
    all dispersed frequency patterns.
    """

    def __init__(self, base_dataset, malicious_client_ids, target_label=0,
                 epsilon=0.1, freq_strategy='DISPERSED', transform=None,
                 backdoor_factory=None):
        """
        Initialize multi-trigger test dataset.

        Args:
            base_dataset: PyTorch dataset
            malicious_client_ids: list of int, IDs of malicious clients
            target_label: int, target class
            epsilon: float, trigger strength
            freq_strategy: str, frequency pattern strategy
            transform: torchvision.transforms
        """
        self.base_dataset = base_dataset
        self.malicious_client_ids = malicious_client_ids
        self.target_label = target_label
        self.transform = transform

        # Create backdoor instances for each malicious client
        self.backdoors = []
        if backdoor_factory is None:
            from core.attacks import FrequencyBackdoor
            backdoor_factory = lambda cid: FrequencyBackdoor(
                client_id=cid,
                target_label=target_label,
                epsilon=epsilon,
                freq_strategy=freq_strategy
            )
        for client_id in malicious_client_ids:
            backdoor = backdoor_factory(client_id)
            self.backdoors.append(backdoor)

        # [P1-1 FIX] Pre-filter: only keep non-target-class samples.
        # Same rationale as PoisonedTestDataset: target-class samples would not
        # receive a trigger (backdoor.__call__ skips them) but their label is
        # already target_label, so they would spuriously inflate ASR.
        self.valid_indices = [
            i for i in range(len(base_dataset))
            if base_dataset[i][1] != target_label
        ]

        # Distribute valid samples evenly across triggers
        n_valid = len(self.valid_indices)
        self.samples_per_trigger = n_valid // len(malicious_client_ids)
        self.total_samples = self.samples_per_trigger * len(malicious_client_ids)

    def set_round(self, round_num):
        """Update state for ALL backdoors in the pool."""
        for backdoor in self.backdoors:
            if hasattr(backdoor, 'set_round'):
                backdoor.set_round(round_num)

    def __len__(self):
        return self.total_samples

    def __getitem__(self, idx):
        """
        Get a poisoned sample using different triggers.

        Returns:
            image: torch.Tensor, poisoned image
            label: int, target label
            client_id: int, which client's trigger was used
        """
        # Determine which trigger to use and which valid sample to poison
        trigger_idx = idx // self.samples_per_trigger
        valid_idx = idx % self.samples_per_trigger  # position within this trigger's slice
        real_idx = self.valid_indices[trigger_idx * self.samples_per_trigger + valid_idx]

        # Get base sample (guaranteed original_label != target_label)
        image, original_label = self.base_dataset[real_idx]

        # Convert to numpy
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        # Apply the corresponding trigger
        backdoor = self.backdoors[trigger_idx]
        image_np, _ = backdoor(image_np, original_label)

        # Convert to PIL for transforms
        image_pil = Image.fromarray(image_np)

        # Apply transforms
        if self.transform is not None:
            image_tensor = self.transform(image_pil)
        else:
            image_tensor = transforms.ToTensor()(image_pil)

        return image_tensor, self.target_label, self.malicious_client_ids[trigger_idx]


class CleanTestDataset(Dataset):
    """
    Clean test dataset for measuring main task accuracy (ACC).
    """

    def __init__(self, base_dataset, transform=None):
        """
        Initialize clean test dataset.

        Args:
            base_dataset: PyTorch dataset
            transform: torchvision.transforms
        """
        self.base_dataset = base_dataset
        self.transform = transform

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        """
        Get a clean sample.

        Returns:
            image: torch.Tensor
            label: int, original label
        """
        image, label = self.base_dataset[idx]

        # Apply transforms
        if self.transform is not None:
            if isinstance(image, Image.Image):
                image = self.transform(image)
            else:
                # If already tensor, convert to PIL first
                image_pil = transforms.ToPILImage()(image)
                image = self.transform(image_pil)

        return image, label


def get_transforms(train=True, dataset='CIFAR10'):
    """
    Get standard data augmentation transforms.

    Args:
        train: bool, whether for training (with augmentation)
        dataset: str, dataset name

    Returns:
        transform: torchvision.transforms.Compose
    """
    if dataset == 'CIFAR10':
        normalize = transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )

        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    elif dataset == 'CIFAR100':
        normalize = transforms.Normalize(
            mean=[0.5071, 0.4867, 0.4408],
            std=[0.2675, 0.2565, 0.2761]
        )

        if train:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
                normalize
            ])

    else:
        # Default simple transform
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    return transform


if __name__ == '__main__':
    print("Dataset module loaded successfully.")
    print("Available classes:")
    print("  - BackdoorDataset: For training with dynamic trigger injection")
    print("  - PoisonedTestDataset: For ASR evaluation")
    print("  - CleanTestDataset: For ACC evaluation")
