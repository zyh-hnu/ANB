"""
Federated Learning Client Implementation

This module implements the client-side logic for federated learning,
including local training for both benign and malicious clients.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import copy
from data.dataset import BackdoorDataset, get_transforms


class Client:
    """
    Federated learning client.

    Handles local training on client's private data partition.
    Supports both benign and malicious (backdoor) training modes.
    """

    def __init__(self, client_id, dataset, indices, is_malicious=False,
                 target_label=0, epsilon=0.1, freq_strategy='DISPERSED',
                 batch_size=32, local_epochs=5, lr=0.01, momentum=0.9,
                 weight_decay=5e-4):
        """
        Initialize client.

        Args:
            client_id: int, unique client identifier
            dataset: PyTorch dataset, base dataset (e.g., CIFAR10)
            indices: list of int, data indices for this client (Non-IID)
            is_malicious: bool, whether this is a malicious client
            target_label: int, backdoor target label
            epsilon: float, trigger injection strength
            freq_strategy: str, 'FIXED' or 'DISPERSED'
            batch_size: int, training batch size
            local_epochs: int, number of local training epochs
            lr: float, learning rate
            momentum: float, SGD momentum
            weight_decay: float, L2 regularization
        """
        self.client_id = client_id
        self.is_malicious = is_malicious
        self.target_label = target_label
        self.batch_size = batch_size
        self.local_epochs = local_epochs
        self.lr = lr
        self.momentum = momentum
        self.weight_decay = weight_decay

        # Create client's data subset
        client_subset = Subset(dataset, indices)

        # Wrap with BackdoorDataset if malicious
        train_transform = get_transforms(train=True, dataset='CIFAR10')

        self.train_dataset = BackdoorDataset(
            base_dataset=client_subset,
            client_id=client_id,
            is_malicious=is_malicious,
            target_label=target_label,
            epsilon=epsilon,
            freq_strategy=freq_strategy,
            transform=train_transform,
            poison_rate=1.0  # Poison all non-target samples
        )

        # Create data loader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True
        )

        # Store dataset size for weighted aggregation
        self.num_samples = len(indices)

    def train(self, global_model, device='cuda', current_round=0):
        """
        Perform local training on client's data.

        Args:
            global_model: nn.Module, current global model
            device: str, 'cuda' or 'cpu'
            current_round: int, current training round (for ANB adaptive strategy)

        Returns:
            local_weights: OrderedDict, trained local model weights
            num_samples: int, number of training samples
            train_loss: float, average training loss
        """
        # Update backdoor strategy state for ANB
        if self.is_malicious and hasattr(self.train_dataset, 'backdoor'):
            self.train_dataset.backdoor.set_round(current_round)

        # Create a local copy of the global model
        local_model = copy.deepcopy(global_model)
        local_model.to(device)
        local_model.train()

        # Set up optimizer
        optimizer = optim.SGD(
            local_model.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )

        # Loss function
        criterion = nn.CrossEntropyLoss()

        # Local training loop
        total_loss = 0.0
        total_batches = 0

        for epoch in range(self.local_epochs):
            epoch_loss = 0.0
            batch_count = 0

            for images, labels in self.train_loader:
                images, labels = images.to(device), labels.to(device)

                # Forward pass
                optimizer.zero_grad()
                outputs = local_model(images)
                loss = criterion(outputs, labels)

                # Backward pass
                loss.backward()
                optimizer.step()

                # Track loss
                epoch_loss += loss.item()
                batch_count += 1

            total_loss += epoch_loss
            total_batches += batch_count

            # Optional: print progress for malicious clients
            if self.is_malicious and epoch == self.local_epochs - 1:
                avg_loss = epoch_loss / batch_count
                print(f"  [Malicious Client {self.client_id}] Epoch {epoch+1}/{self.local_epochs}, Loss: {avg_loss:.4f}")

        avg_train_loss = total_loss / total_batches

        # Return trained weights
        return local_model.state_dict(), self.num_samples, avg_train_loss

    def get_model_update(self, global_model, device='cuda', current_round=0):
        """
        Get model update (delta) instead of full weights.

        This is useful for some aggregation methods that work with deltas.

        Args:
            global_model: nn.Module, current global model
            device: str, 'cuda' or 'cpu'
            current_round: int, current training round

        Returns:
            delta: OrderedDict, model parameter updates
            num_samples: int, number of training samples
        """
        # Train locally
        local_weights, num_samples, train_loss = self.train(global_model, device, current_round)

        # Compute delta (local_weights - global_weights)
        global_weights = global_model.state_dict()
        delta = {}

        for key in local_weights.keys():
            delta[key] = local_weights[key] - global_weights[key]

        return delta, num_samples, train_loss


def create_clients(dataset, num_clients, malicious_indices, client_indices,
                   target_label=0, epsilon=0.1, freq_strategy='DISPERSED',
                   batch_size=32, local_epochs=5, lr=0.01):
    """
    Create a list of federated learning clients.

    Args:
        dataset: PyTorch dataset
        num_clients: int, total number of clients
        malicious_indices: list of int, indices of malicious clients
        client_indices: list of lists, data indices for each client
        target_label: int, backdoor target
        epsilon: float, trigger strength
        freq_strategy: str, frequency pattern strategy
        batch_size: int
        local_epochs: int
        lr: float

    Returns:
        clients: list of Client objects
    """
    clients = []

    for client_id in range(num_clients):
        is_malicious = client_id in malicious_indices

        client = Client(
            client_id=client_id,
            dataset=dataset,
            indices=client_indices[client_id],
            is_malicious=is_malicious,
            target_label=target_label,
            epsilon=epsilon,
            freq_strategy=freq_strategy,
            batch_size=batch_size,
            local_epochs=local_epochs,
            lr=lr
        )

        clients.append(client)

    print(f"Created {num_clients} clients:")
    print(f"  - Malicious clients: {malicious_indices}")
    print(f"  - Benign clients: {[i for i in range(num_clients) if i not in malicious_indices]}")

    return clients


if __name__ == '__main__':
    print("Client module loaded successfully.")
    print("Main classes:")
    print("  - Client: Handles local training")
    print("  - create_clients(): Factory function for creating client list")
