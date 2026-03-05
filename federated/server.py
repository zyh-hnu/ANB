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
from core.defenses import aggregate_with_defense


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

        # [P3-2] Foolsgold requires persistent cross-round history
        self._foolsgold_history = {}

        # Track training history
        self.history = {
            'train_loss': [],
            'test_acc': [],
            'test_asr': [],
            'test_asr_multi': [],
            'per_client_asr': [],
            'defense_results': [],
            # [P2-2] Defense quantitative metrics per round
            'defense_recall': [],       # fraction of malicious clients detected
            'defense_precision': [],    # fraction of flagged clients that are truly malicious
            'defense_f1': [],           # harmonic mean of precision and recall
            'defense_bypass_rate': [],  # fraction of malicious clients that slipped through
        }

        # Storage for client weights (for analysis)
        self.saved_weights = {}

    @staticmethod
    def _compute_acceptance_metrics(num_clients, accepted_clients, malicious_indices):
        """
        Compute defense metrics from the *actual acceptance decision*.

        This keeps Recall/Precision/F1/Bypass internally consistent:
        - detected_malicious := clients filtered out by defense
        - bypassed_malicious := malicious clients still accepted
        """
        accepted_set = set(accepted_clients)
        malicious_set = set(malicious_indices or [])
        predicted_malicious = set(range(num_clients)) - accepted_set

        n_malicious = len(malicious_set)
        true_positives = len(predicted_malicious & malicious_set)
        false_positives = len(predicted_malicious - malicious_set)
        false_negatives = len(malicious_set - predicted_malicious)
        true_negatives = num_clients - n_malicious - false_positives

        precision = (
            true_positives / (true_positives + false_positives)
            if (true_positives + false_positives) > 0
            else 0.0
        )
        recall = true_positives / n_malicious if n_malicious > 0 else 0.0
        f1_score = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        bypass_rate = (
            len(malicious_set & accepted_set) / n_malicious
            if n_malicious > 0
            else 0.0
        )

        return {
            'true_positives': true_positives,
            'false_positives': false_positives,
            'false_negatives': false_negatives,
            'true_negatives': true_negatives,
            'precision': precision,
            'recall': recall,
            'f1_score': f1_score,
            'detected_malicious': sorted(predicted_malicious),
            'missed_malicious': sorted(malicious_set - predicted_malicious),
            'bypass_rate': bypass_rate,
        }

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
            # [P3-2] Unified defense dispatch (FreqFed / FLTrust / Foolsgold)
            print(f"\n[Defense] Running '{self.defense_method}' defense...")
            agg_weights = None
            defense_record = {'method': self.defense_method, 'accepted_clients': []}
            try:
                global_w = self.global_model.state_dict()
                agg_weights, accepted_clients, defense_record = aggregate_with_defense(
                    method=self.defense_method,
                    client_weights_list=client_weights_list,
                    client_num_samples=client_num_samples,
                    global_weights=global_w,
                    malicious_indices=malicious_indices,
                    # FLTrust: use global model as root approximation (no dedicated root set)
                    root_weights=global_w,
                    # Foolsgold: pass persistent history dict
                    history_contributions=self._foolsgold_history,
                )

            except Exception as e:
                print(f"[Defense] Error: {e}")
                print("[Defense] Fallback: accepting all clients (no filtering)")
                accepted_clients = list(range(num_clients))
                defense_record = {
                    'method': self.defense_method,
                    'error': str(e),
                    'fallback': 'accept_all_on_error',
                    'accepted_clients': accepted_clients,
                }

            # [Consistency Fix] Always compute metrics from *accepted_clients*.
            if malicious_indices is not None:
                acceptance_metrics = self._compute_acceptance_metrics(
                    num_clients=num_clients,
                    accepted_clients=accepted_clients,
                    malicious_indices=malicious_indices,
                )
                defense_record.update(acceptance_metrics)

                self.history['defense_recall'].append(acceptance_metrics['recall'])
                self.history['defense_precision'].append(acceptance_metrics['precision'])
                self.history['defense_f1'].append(acceptance_metrics['f1_score'])
                self.history['defense_bypass_rate'].append(acceptance_metrics['bypass_rate'])

                print(f"[Defense] Recall={acceptance_metrics['recall']:.2%}  "
                      f"Precision={acceptance_metrics['precision']:.2%}  "
                      f"F1={acceptance_metrics['f1_score']:.2%}  "
                      f"Bypass={acceptance_metrics['bypass_rate']:.2%}")

            self.history['defense_results'].append(defense_record)

            # If defense returned pre-aggregated weights (FLTrust / Foolsgold),
            # load them directly and return — skip vanilla FedAvg below.
            if agg_weights is not None:
                self.global_model.load_state_dict(agg_weights)
                print(f"[Defense] '{self.defense_method}' aggregation applied to "
                      f"{num_clients} clients.")
                return accepted_clients

            print(f"[Defense] Accepting updates from "
                  f"{len(accepted_clients)}/{num_clients} clients")

        else:
            # No defense: use all clients
            accepted_clients = list(range(num_clients))

        # FedAvg aggregation on accepted clients
        if len(accepted_clients) == 0:
            print("[Warning] No clients accepted, keeping global model unchanged")
            return accepted_clients

        # Calculate weighted average
        total_samples = sum([client_num_samples[i] for i in accepted_clients])

        # [P1-3 FIX] Use the first *accepted* client as dtype/shape template.
        # Previously this was hardcoded to client_weights_list[0], which could be
        # a client excluded by the defense (not in accepted_clients), causing the
        # dtype reference to be mismatched from the actual aggregation participants.
        template_weights = client_weights_list[accepted_clients[0]]
        aggregated_weights = {}

        # Initialize accumulator with Float32 to avoid LongTensor casting errors
        for key in template_weights.keys():
            aggregated_weights[key] = torch.zeros_like(template_weights[key], dtype=torch.float32)

        # Weighted sum over accepted clients only
        for i in accepted_clients:
            weight = client_num_samples[i] / total_samples
            for key in aggregated_weights.keys():
                # Accumulate in float precision
                aggregated_weights[key] += weight * client_weights_list[i][key].float()

        # Cast back to original types (e.g. Long for num_batches_tracked)
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
                           test_asr_multi=None, per_client_asr=None, defense_metrics=None):
        """
        Print summary of current round.

        Args:
            round_num: int
            train_loss: float
            test_acc: float
            test_asr: float
            test_asr_multi: float, ASR on multi-trigger test set
            per_client_asr: dict, {client_id: asr}
            defense_metrics: dict, {recall, precision, f1_score, bypass_rate} or None
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

        # [P2-2] Print defense metrics if available
        if defense_metrics is not None:
            print(f"\nDefense Metrics:")
            print(f"  Recall    (detected/malicious): {defense_metrics.get('recall', 0):.2%}")
            print(f"  Precision (true/flagged):       {defense_metrics.get('precision', 0):.2%}")
            print(f"  F1 Score:                       {defense_metrics.get('f1_score', 0):.2%}")
            print(f"  Bypass Rate (evaded/malicious): {defense_metrics.get('bypass_rate', 0):.2%}")
            if defense_metrics.get('bypass_rate', 0) > 0.5:
                print(f"  *** DEFENSE BYPASSED this round ***")

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

        # Print summary — include defense metrics from this round if available
        defense_metrics = (server.history['defense_results'][-1]
                           if server.defense_enabled and server.history['defense_results']
                           else None)
        server.print_round_summary(round_num, avg_train_loss, test_acc, test_asr,
                                  test_asr_multi, per_client_asr, defense_metrics)

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
