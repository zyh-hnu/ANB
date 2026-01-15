--- START OF FILE federated/server.py ---

"""
Federated Learning Server Implementation

This module implements the server-side logic for federated learning,
including model aggregation (FedAvg) and optional defense mechanisms.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import copy
import numpy as np
import os
import pickle
from core.defenses import cluster_clients, print_defense_results


class Server:
    """
    Federated learning server.

    Handles model aggregation and coordinates training across clients.
    Supports optional defense mechanisms (e.g., FreqFed).
    """

    def __init__(self, model, device='cuda', defense_enabled=False,
                 defense_method='hdbscan', target_label=0):
        """
        Initialize server.

        Args:
            model: nn.Module, global model architecture
            device: str, 'cuda' or 'cpu'
            defense_enabled: bool, whether to apply defense
            defense_method: str, clustering method for defense
            target_label: int, backdoor target (for evaluation)
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.global_model = model.to(self.device)
        self.defense_enabled = defense_enabled
        self.defense_method = defense_method
        self.target_label = target_label

        # Track training history
        self.history = {
            'train_loss': [],
            'test_acc': [],
            'test_asr': [],
            'test_asr_multi': [],
            'per_client_asr': [],
            'defense_results': []
        }

        # Storage for client weights (for analysis)
        self.saved_weights = {}

    def aggregate(self, client_weights_list, client_num_samples, malicious_indices=None):
        """
        Aggregate client models using FedAvg or with defense.

        Args:
            client_weights_list: list of OrderedDict, client model weights
            client_num_samples: list of int, number of samples per client
            malicious_indices: list of int, ground truth malicious clients (for evaluation)

        Returns:
            accepted_clients: list of int, indices of clients whose updates were used
        """
        num_clients = len(client_weights_list)

        if self.defense_enabled:
            # Apply defense: cluster clients and filter suspicious ones
            print("\n[Defense] Analyzing client models...")

            try:
                labels, features = cluster_clients(
                    client_weights_list,
                    method=self.defense_method,
                    n_clusters=2,
                    freq_band='low-mid',
                    compression_ratio=0.2
                )

                # Print defense results
                print_defense_results(labels, malicious_indices)

                # Identify benign cluster (largest cluster with label >= 0)
                unique_labels, counts = np.unique(labels, return_counts=True)
                valid_labels = unique_labels[unique_labels >= 0]

                if len(valid_labels) > 0:
                    valid_counts = counts[unique_labels >= 0]
                    benign_label = valid_labels[np.argmax(valid_counts)]

                    # Only aggregate clients in benign cluster
                    accepted_clients = [i for i in range(num_clients) if labels[i] == benign_label]
                else:
                    # All clients labeled as noise, use all (defense failed)
                    print("[Defense] Warning: All clients labeled as noise, using all updates")
                    accepted_clients = list(range(num_clients))
            except Exception as e:
                print(f"[Defense] Error during clustering: {e}")
                print("[Defense] Fallback: Accepting all clients")
                accepted_clients = list(range(num_clients))

            print(f"[Defense] Accepting updates from {len(accepted_clients)}/{num_clients} clients")

            # Store defense results if clustering succeeded
            if 'labels' in locals():
                self.history['defense_results'].append({
                    'labels': labels.tolist(),
                    'accepted_clients': accepted_clients
                })

        else:
            # No defense: use all clients
            accepted_clients = list(range(num_clients))

        # FedAvg aggregation on accepted clients
        if len(accepted_clients) == 0:
            print("[Warning] No clients accepted, keeping global model unchanged")
            return accepted_clients

        # Calculate weighted average
        total_samples = sum([client_num_samples[i] for i in accepted_clients])
        
        # Use the first client's weights as a template
        template_weights = client_weights_list[0]
        aggregated_weights = {}

        # === FIX: Initialize accumulator with Float32 to avoid LongTensor casting errors ===
        for key in template_weights.keys():
            aggregated_weights[key] = torch.zeros_like(template_weights[key], dtype=torch.float32)

        # Weighted sum
        for i in accepted_clients:
            weight = client_num_samples[i] / total_samples
            for key in aggregated_weights.keys():
                # Accumulate in float precision
                aggregated_weights[key] += weight * client_weights_list[i][key]

        # === FIX: Cast back to original types (e.g. Long for num_batches_tracked) ===
        final_weights = {}
        for key in template_weights.keys():
            target_type = template_weights[key].dtype
            if aggregated_weights[key].dtype != target_type:
                # Round integers to nearest value to avoid precision drift
                if target_type in [torch.int64, torch.int32, torch.uint8]:
                     final_weights[key] = torch.round(aggregated_weights[key]).to(target_type)
                else:
                     final_weights[key] = aggregated_weights[key].to(target_type)
            else:
                final_weights[key] = aggregated_weights[key]

        # Update global model
        self.global_model.load_state_dict(final_weights)

        return accepted_clients

    def evaluate(self, test_loader):
        """
        Evaluate global model on clean test set (main task accuracy).

        Args:
            test_loader: DataLoader, clean test data

        Returns:
            accuracy: float, test accuracy
            loss: float, average test loss
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.global_model(images)
                loss = criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = correct / total
        avg_loss = total_loss / len(test_loader)

        return accuracy, avg_loss

    def evaluate_asr(self, poisoned_test_loader):
        """
        Evaluate Attack Success Rate on poisoned test set.

        ASR = percentage of poisoned samples classified as target label.

        Args:
            poisoned_test_loader: DataLoader, poisoned test data

        Returns:
            asr: float, attack success rate
        """
        self.global_model.eval()

        correct = 0
        total = 0

        with torch.no_grad():
            for batch in poisoned_test_loader:
                # Handle both 2-tuple (image, label) and 3-tuple (image, label, client_id)
                if len(batch) == 2:
                    images, labels = batch
                else:
                    images, labels, _ = batch

                images = images.to(self.device)
                # labels should all be target_label

                outputs = self.global_model(images)
                _, predicted = torch.max(outputs.data, 1)

                total += images.size(0)
                correct += (predicted == self.target_label).sum().item()

        asr = correct / total if total > 0 else 0.0

        return asr

    def evaluate_per_client_asr(self, per_client_loaders):
        """
        Evaluate ASR for each malicious client's trigger separately.

        Args:
            per_client_loaders: dict of DataLoaders, {client_id: loader}

        Returns:
            per_client_asr: dict of {client_id: asr}
        """
        self.global_model.eval()
        per_client_asr = {}

        for client_id, loader in per_client_loaders.items():
            correct = 0
            total = 0

            with torch.no_grad():
                for images, labels in loader:
                    images = images.to(self.device)
                    outputs = self.global_model(images)
                    _, predicted = torch.max(outputs.data, 1)

                    total += images.size(0)
                    correct += (predicted == self.target_label).sum().item()

            asr = correct / total if total > 0 else 0.0
            per_client_asr[client_id] = asr

        return per_client_asr

    def get_model(self):
        """
        Get a copy of the current global model.

        Returns:
            model: nn.Module, copy of global model
        """
        return copy.deepcopy(self.global_model)

    def save_model(self, path):
        """
        Save global model to file.

        Args:
            path: str, file path
        """
        torch.save(self.global_model.state_dict(), path)
        print(f"Model saved to {path}")

    def load_model(self, path):
        """
        Load global model from file.

        Args:
            path: str, file path
        """
        self.global_model.load_state_dict(torch.load(path))
        print(f"Model loaded from {path}")

    def save_client_weights(self, round_num, client_weights_list, malicious_indices,
                           selected_indices, save_dir='./results/weights'):
        """
        Save client weights for defense analysis.

        Args:
            round_num: int, current round number
            client_weights_list: list of OrderedDict, client model weights
            malicious_indices: list of int, indices of malicious clients
            selected_indices: list of int, indices of selected clients this round
            save_dir: str, directory to save weights
        """
        os.makedirs(save_dir, exist_ok=True)

        # Prepare data for saving
        weights_data = {
            'round': round_num,
            'client_weights': client_weights_list,
            'malicious_indices': malicious_indices,
            'selected_indices': selected_indices,
            'num_clients': len(client_weights_list)
        }

        # Save to pickle file
        filepath = os.path.join(save_dir, f'client_weights_round_{round_num}.pkl')
        with open(filepath, 'wb') as f:
            pickle.dump(weights_data, f)

        print(f"[Server] Client weights saved to {filepath}")

        # Store in memory for quick access
        self.saved_weights[round_num] = weights_data

    def print_round_summary(self, round_num, train_loss, test_acc, test_asr,
                           test_asr_multi=None, per_client_asr=None):
        """
        Print summary of current round.

        Args:
            round_num: int
            train_loss: float
            test_acc: float
            test_asr: float
            test_asr_multi: float, ASR on multi-trigger test set
            per_client_asr: dict, {client_id: asr}
        """
        print(f"\n{'='*60}")
        print(f"Round {round_num} Summary")
        print(f"{'='*60}")
        print(f"Train Loss: {train_loss:.4f}")
        print(f"Test Accuracy (Clean): {test_acc:.2%}")
        print(f"Attack Success Rate (Single Trigger): {test_asr:.2%}")

        if test_asr_multi is not None:
            print(f"Attack Success Rate (Multi-Trigger): {test_asr_multi:.2%}")

        if per_client_asr is not None and len(per_client_asr) > 0:
            print(f"\nPer-Client ASR:")
            for client_id, asr in sorted(per_client_asr.items()):
                print(f"  Client {client_id}: {asr:.2%}")

        print(f"{'='*60}\n")

        # Store in history
        self.history['train_loss'].append(train_loss)
        self.history['test_acc'].append(test_acc)
        self.history['test_asr'].append(test_asr)
        if test_asr_multi is not None:
            self.history['test_asr_multi'].append(test_asr_multi)
        if per_client_asr is not None:
            self.history['per_client_asr'].append(per_client_asr)


def federated_training(server, clients, test_loader, poisoned_test_loader,
                       multi_trigger_loader=None, per_client_loaders=None,
                       num_rounds=50, malicious_indices=None, client_fraction=1.0,
                       save_weights_at_rounds=None):
    """
    Execute federated learning training loop.

    Args:
        server: Server object
        clients: list of Client objects
        test_loader: DataLoader, clean test data
        poisoned_test_loader: DataLoader, poisoned test data (single trigger)
        multi_trigger_loader: DataLoader, multi-trigger test data
        per_client_loaders: dict of DataLoaders for per-client evaluation
        num_rounds: int, number of federated rounds
        malicious_indices: list of int, ground truth malicious clients
        client_fraction: float, fraction of clients to sample each round
        save_weights_at_rounds: list of int, rounds to save client weights

    Returns:
        server: Server object with trained model
    """
    num_clients = len(clients)
    num_selected = max(1, int(num_clients * client_fraction))

    # Default: save weights at last round
    if save_weights_at_rounds is None:
        save_weights_at_rounds = [num_rounds]

    print(f"\nStarting Federated Learning")
    print(f"  Total clients: {num_clients}")
    print(f"  Clients per round: {num_selected}")
    print(f"  Total rounds: {num_rounds}")
    print(f"  Defense enabled: {server.defense_enabled}")
    if server.defense_enabled:
        print(f"  Defense method: {server.defense_method}")
    print(f"  Save weights at rounds: {save_weights_at_rounds}")
    print()

    for round_num in range(1, num_rounds + 1):
        print(f"\n{'='*60}")
        print(f"Round {round_num}/{num_rounds}")
        print(f"{'='*60}")

        # Sample clients for this round
        if client_fraction < 1.0:
            selected_indices = np.random.choice(num_clients, num_selected, replace=False)
        else:
            selected_indices = list(range(num_clients))
        
        # --- FIX: Ensure selected_indices is a Python list for consistent handling ---
        if isinstance(selected_indices, np.ndarray):
            selected_indices = selected_indices.tolist()

        # Client local training
        client_weights_list = []
        client_num_samples = []
        total_train_loss = 0.0

        print(f"Selected clients: {selected_indices}")

        for idx in selected_indices:
            client = clients[idx]
            print(f"\nTraining client {client.client_id} ({'MALICIOUS' if client.is_malicious else 'benign'})...")

            # Train client (pass current_round for ANB adaptive strategy)
            weights, num_samples, train_loss = client.train(server.global_model, server.device, current_round=round_num)

            client_weights_list.append(weights)
            client_num_samples.append(num_samples)
            total_train_loss += train_loss * num_samples

        # Calculate average training loss
        avg_train_loss = total_train_loss / sum(client_num_samples)

        # Save client weights if requested
        if round_num in save_weights_at_rounds:
            malicious_in_selected = [i for i, idx in enumerate(selected_indices) if idx in (malicious_indices or [])]
            server.save_client_weights(
                round_num,
                client_weights_list,
                malicious_in_selected,
                selected_indices
            )

        # Server aggregation (with optional defense)
        print("\n[Server] Aggregating client models...")
        accepted_clients = server.aggregate(
            client_weights_list,
            client_num_samples,
            malicious_indices=[i for i, idx in enumerate(selected_indices) if idx in (malicious_indices or [])]
        )

        # Evaluation
        print("\n[Server] Evaluating global model...")
        
        # ANB ADAPTATION: Update test datasets to reflect current round dynamics
        # This ensures ASR is measured against the current "Nebula" state, not just "Star" state
        if hasattr(poisoned_test_loader.dataset, 'set_round'):
            poisoned_test_loader.dataset.set_round(round_num)
        
        if multi_trigger_loader is not None and hasattr(multi_trigger_loader.dataset, 'set_round'):
            multi_trigger_loader.dataset.set_round(round_num)
        
        if per_client_loaders is not None:
            for loader in per_client_loaders.values():
                if hasattr(loader.dataset, 'set_round'):
                    loader.dataset.set_round(round_num)

        test_acc, test_loss = server.evaluate(test_loader)
        test_asr = server.evaluate_asr(poisoned_test_loader)

        # Multi-trigger evaluation
        test_asr_multi = None
        if multi_trigger_loader is not None:
            test_asr_multi = server.evaluate_asr(multi_trigger_loader)

        # Per-client evaluation
        per_client_asr = None
        if per_client_loaders is not None:
            per_client_asr = server.evaluate_per_client_asr(per_client_loaders)

        # Print summary
        server.print_round_summary(round_num, avg_train_loss, test_acc, test_asr,
                                  test_asr_multi, per_client_asr)

        # Early stopping if ASR is high enough (for efficiency)
        if test_asr > 0.95 and not server.defense_enabled:
            print(f"\nEarly stopping: ASR > 95% achieved at round {round_num}")
            break

    print("\n" + "="*60)
    print("Federated Learning Complete")
    print("="*60)
    print(f"Final Test Accuracy: {server.history['test_acc'][-1]:.2%}")
    print(f"Final ASR (Single Trigger): {server.history['test_asr'][-1]:.2%}")
    if len(server.history['test_asr_multi']) > 0:
        print(f"Final ASR (Multi-Trigger): {server.history['test_asr_multi'][-1]:.2%}")
    print("="*60 + "\n")