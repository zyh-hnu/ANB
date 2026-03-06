from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core.registry import ATTACKS, MODELS  # noqa: E402
import core.attacks  # noqa: E402,F401
import models.resnet  # noqa: E402,F401
from data.dataset import (  # noqa: E402
    CleanTestDataset,
    MultiTriggerTestDataset,
    PoisonedTestDataset,
    get_transforms,
)


class CentralizedSAFBDataset(Dataset):
    MODE_CLEAN = 0
    MODE_POISON = 1
    MODE_CROSS = 2

    def __init__(
        self,
        base_dataset,
        transform,
        backdoor_factory,
        client_id: int,
        target_label: int,
        poison_rate: float,
        cross_ratio: float,
    ) -> None:
        self.base_dataset = base_dataset
        self.transform = transform
        self.target_label = int(target_label)
        self.poison_rate = float(np.clip(poison_rate, 0.0, 1.0))
        self.cross_ratio = max(0.0, float(cross_ratio))
        self.cross_rate = min(1.0 - self.poison_rate, self.poison_rate * self.cross_ratio)
        self.backdoor = backdoor_factory(client_id)

    def set_round(self, round_num: int) -> None:
        if hasattr(self.backdoor, "set_round"):
            self.backdoor.set_round(round_num)

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, idx):
        image, label = self.base_dataset[idx]
        if isinstance(image, Image.Image):
            image_np = np.array(image)
        else:
            image_np = image

        label_out = int(label)
        mode = self.MODE_CLEAN

        if label_out != self.target_label:
            sample_prob = np.random.random()
            if sample_prob < self.poison_rate:
                image_np, _ = self.backdoor(image_np, label_out)
                label_out = self.target_label
                mode = self.MODE_POISON
            elif sample_prob < self.poison_rate + self.cross_rate:
                image_np, _ = self.backdoor(image_np, label_out)
                mode = self.MODE_CROSS

        if not isinstance(image_np, Image.Image):
            image_pil = Image.fromarray(image_np)
        else:
            image_pil = image_np

        image_tensor = self.transform(image_pil) if self.transform is not None else image_pil
        return image_tensor, label_out, mode


def setup_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Centralized SAFB training")
    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["CIFAR10", "CIFAR100"])
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--model-name", type=str, default="resnet18")
    parser.add_argument("--backdoor-name", type=str, default="frequency")
    parser.add_argument("--freq-strategy", type=str, default="ANB", choices=["ANB", "FIXED"])
    parser.add_argument("--target-label", type=int, default=0)
    parser.add_argument("--epsilon", type=float, default=0.1)
    parser.add_argument("--client-id", type=int, default=0)
    parser.add_argument("--poison-rate", type=float, default=0.2)
    parser.add_argument("--cross-ratio", type=float, default=1.0)
    parser.add_argument("--backdoor-boost-weight", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--milestones", type=str, default="30,45")
    parser.add_argument("--gamma", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", type=int, default=1)
    parser.add_argument("--train-subset", type=int, default=0)
    parser.add_argument("--test-subset", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "cpu"])
    parser.add_argument("--eval-multi-client-ids", type=str, default="")
    parser.add_argument("--min-asr-for-tradeoff", type=float, default=0.85)
    parser.add_argument("--output-dir", type=str, default="./results/centralized_runs")
    parser.add_argument("--run-name", type=str, default="")
    parser.add_argument("--use-phased-chaos", type=int, default=1)
    parser.add_argument("--use-spectral-smoothing", type=int, default=1)
    parser.add_argument("--use-freq-sharding", type=int, default=1)
    parser.add_argument("--use-dual-routing", type=int, default=1)
    return parser.parse_args()


def _load_dataset(dataset_name: str, data_dir: str):
    if dataset_name == "CIFAR10":
        train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=None)
        num_classes = 10
    elif dataset_name == "CIFAR100":
        train_dataset = datasets.CIFAR100(root=data_dir, train=True, download=True, transform=None)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=None)
        num_classes = 100
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    return train_dataset, test_dataset, num_classes


def _parse_client_ids(raw_ids: str) -> list[int]:
    if not raw_ids.strip():
        return []
    parsed = []
    for item in raw_ids.split(","):
        item = item.strip()
        if item:
            parsed.append(int(item))
    return parsed


def _build_backdoor_factory(args: argparse.Namespace):
    attack_cls = ATTACKS.get(args.backdoor_name)

    if attack_cls.__name__ == "AdaptiveNebulaBackdoor":
        return lambda cid: attack_cls(
            client_id=cid,
            target_label=args.target_label,
            epsilon=args.epsilon,
            max_rounds=args.epochs,
            strategy=args.freq_strategy,
            use_phased_chaos=bool(args.use_phased_chaos),
            use_spectral_smoothing=bool(args.use_spectral_smoothing),
            use_freq_sharding=bool(args.use_freq_sharding),
            use_dual_routing=bool(args.use_dual_routing),
        )

    return lambda cid: attack_cls(
        client_id=cid,
        target_label=args.target_label,
        epsilon=args.epsilon,
        freq_strategy=args.freq_strategy,
        use_phased_chaos=bool(args.use_phased_chaos),
        use_spectral_smoothing=bool(args.use_spectral_smoothing),
        use_freq_sharding=bool(args.use_freq_sharding),
        use_dual_routing=bool(args.use_dual_routing),
    )


def _subset_dataset(dataset, subset_size: int, seed: int):
    if subset_size <= 0 or subset_size >= len(dataset):
        return dataset
    rng = np.random.default_rng(seed)
    indices = rng.choice(len(dataset), size=subset_size, replace=False)
    return Subset(dataset, indices.tolist())


def _evaluate_clean(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()
            predictions = torch.argmax(outputs, dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    avg_loss = total_loss / max(1, len(loader))
    acc = correct / max(1, total)
    return acc, avg_loss


def _evaluate_asr(model, loader, target_label: int, device):
    model.eval()
    hit_target = 0
    total = 0
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 2:
                images, _ = batch
            else:
                images, _, _ = batch
            images = images.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)
            hit_target += (predictions == target_label).sum().item()
            total += images.size(0)
    return hit_target / max(1, total)


def _train_one_epoch(
    model,
    loader,
    optimizer,
    criterion,
    device,
    target_label: int,
    backdoor_boost_weight: float,
    epoch: int,
    total_epochs: int,
    log_interval: int,
):
    model.train()
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    mode_correct = {
        CentralizedSAFBDataset.MODE_CLEAN: 0,
        CentralizedSAFBDataset.MODE_POISON: 0,
        CentralizedSAFBDataset.MODE_CROSS: 0,
    }
    mode_total = {
        CentralizedSAFBDataset.MODE_CLEAN: 0,
        CentralizedSAFBDataset.MODE_POISON: 0,
        CentralizedSAFBDataset.MODE_CROSS: 0,
    }

    total_batches = max(1, len(loader))
    for batch_idx, (images, labels, modes) in enumerate(loader, start=1):
        images = images.to(device)
        labels = labels.to(device)
        modes = modes.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)

        if backdoor_boost_weight > 0:
            non_target_mask = labels != target_label
            if non_target_mask.any():
                target_probs = torch.softmax(outputs[non_target_mask], dim=1)[:, target_label]
                boost_loss = -torch.log(target_probs + 1e-8).mean()
                loss = loss + backdoor_boost_weight * boost_loss

        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=1)
        running_loss += loss.item() * labels.size(0)
        running_correct += (predictions == labels).sum().item()
        running_total += labels.size(0)

        for mode in mode_total.keys():
            mask = modes == mode
            if mask.any():
                mode_total[mode] += mask.sum().item()
                mode_correct[mode] += (predictions[mask] == labels[mask]).sum().item()

        if log_interval > 0 and (batch_idx % log_interval == 0 or batch_idx == total_batches):
            running_avg_loss = running_loss / max(1, running_total)
            running_avg_acc = running_correct / max(1, running_total)
            print(
                f"  [Epoch {epoch:03d}/{total_epochs}] batch {batch_idx}/{total_batches} "
                f"({100.0 * batch_idx / total_batches:5.1f}%) "
                f"loss={running_avg_loss:.4f} acc={running_avg_acc:.2%}",
                flush=True,
            )

    epoch_loss = running_loss / max(1, running_total)
    epoch_acc = running_correct / max(1, running_total)
    clean_acc = mode_correct[CentralizedSAFBDataset.MODE_CLEAN] / max(
        1, mode_total[CentralizedSAFBDataset.MODE_CLEAN]
    )
    bd_acc = mode_correct[CentralizedSAFBDataset.MODE_POISON] / max(
        1, mode_total[CentralizedSAFBDataset.MODE_POISON]
    )
    cross_acc = mode_correct[CentralizedSAFBDataset.MODE_CROSS] / max(
        1, mode_total[CentralizedSAFBDataset.MODE_CROSS]
    )

    return {
        "loss": epoch_loss,
        "acc": epoch_acc,
        "clean_acc": clean_acc,
        "bd_acc": bd_acc,
        "cross_acc": cross_acc,
        "clean_samples": mode_total[CentralizedSAFBDataset.MODE_CLEAN],
        "bd_samples": mode_total[CentralizedSAFBDataset.MODE_POISON],
        "cross_samples": mode_total[CentralizedSAFBDataset.MODE_CROSS],
    }


def _resolve_device(device_name: str) -> torch.device:
    if device_name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_name == "cpu":
        return torch.device("cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    args = parse_args()
    setup_seed(args.seed)

    device = _resolve_device(args.device)
    run_name = args.run_name.strip() or time.strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.output_dir) / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("Centralized SAFB Training")
    print(f"Run name: {run_name}")
    print(f"Output : {run_dir}")
    print(f"Device : {device}")
    print(f"Dataset: {args.dataset}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(
        f"Subset: train={args.train_subset if args.train_subset > 0 else 'full'}, "
        f"test={args.test_subset if args.test_subset > 0 else 'full'}"
    )
    print(f"Loader: workers={args.num_workers}, log_interval={args.log_interval}")
    print(
        f"Poison: rate={args.poison_rate:.2f}, cross_ratio={args.cross_ratio:.2f}, "
        f"target={args.target_label}, epsilon={args.epsilon:.3f}"
    )
    print("=" * 72)

    train_base, test_base, num_classes = _load_dataset(args.dataset, args.data_dir)
    train_base = _subset_dataset(train_base, args.train_subset, args.seed)
    test_base = _subset_dataset(test_base, args.test_subset, args.seed + 1)
    print(f"Train samples: {len(train_base)}, Test samples: {len(test_base)}")
    backdoor_factory = _build_backdoor_factory(args)

    train_transform = get_transforms(train=True, dataset=args.dataset)
    test_transform = get_transforms(train=False, dataset=args.dataset)

    train_dataset = CentralizedSAFBDataset(
        base_dataset=train_base,
        transform=train_transform,
        backdoor_factory=backdoor_factory,
        client_id=args.client_id,
        target_label=args.target_label,
        poison_rate=args.poison_rate,
        cross_ratio=args.cross_ratio,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    clean_test_dataset = CleanTestDataset(test_base, transform=test_transform)
    clean_test_loader = DataLoader(
        clean_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    poisoned_test_dataset = PoisonedTestDataset(
        test_base,
        target_label=args.target_label,
        epsilon=args.epsilon,
        freq_strategy=args.freq_strategy,
        client_id=args.client_id,
        transform=test_transform,
        backdoor_factory=backdoor_factory,
    )
    poisoned_test_loader = DataLoader(
        poisoned_test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=bool(args.pin_memory),
    )

    multi_client_ids = _parse_client_ids(args.eval_multi_client_ids)
    multi_loader = None
    if multi_client_ids:
        multi_dataset = MultiTriggerTestDataset(
            test_base,
            malicious_client_ids=multi_client_ids,
            target_label=args.target_label,
            epsilon=args.epsilon,
            freq_strategy=args.freq_strategy,
            transform=test_transform,
            backdoor_factory=backdoor_factory,
        )
        multi_loader = DataLoader(
            multi_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=bool(args.pin_memory),
        )

    model_builder = MODELS.get(args.model_name)
    model = model_builder(num_classes=num_classes).to(device)

    optimizer = optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
    )

    milestone_values = [int(item.strip()) for item in args.milestones.split(",") if item.strip()]
    scheduler = (
        optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestone_values, gamma=args.gamma)
        if milestone_values
        else None
    )
    criterion = nn.CrossEntropyLoss()

    history = {
        "train_loss": [],
        "train_acc": [],
        "train_clean_acc": [],
        "train_bd_acc": [],
        "train_cross_acc": [],
        "test_acc": [],
        "test_loss": [],
        "test_asr": [],
        "test_asr_multi": [],
    }

    best_asr = -1.0
    best_asr_acc = -1.0
    best_tradeoff_acc = -1.0
    best_tradeoff_asr = -1.0

    for epoch in range(1, args.epochs + 1):
        epoch_start = time.time()
        train_dataset.set_round(epoch)
        if hasattr(poisoned_test_loader.dataset, "set_round"):
            poisoned_test_loader.dataset.set_round(epoch)
        if multi_loader is not None and hasattr(multi_loader.dataset, "set_round"):
            multi_loader.dataset.set_round(epoch)

        train_metrics = _train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
            target_label=args.target_label,
            backdoor_boost_weight=args.backdoor_boost_weight,
            epoch=epoch,
            total_epochs=args.epochs,
            log_interval=args.log_interval,
        )
        clean_acc, clean_loss = _evaluate_clean(model, clean_test_loader, criterion, device)
        asr = _evaluate_asr(model, poisoned_test_loader, args.target_label, device)
        asr_multi = _evaluate_asr(model, multi_loader, args.target_label, device) if multi_loader else None
        epoch_time_sec = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["train_acc"].append(train_metrics["acc"])
        history["train_clean_acc"].append(train_metrics["clean_acc"])
        history["train_bd_acc"].append(train_metrics["bd_acc"])
        history["train_cross_acc"].append(train_metrics["cross_acc"])
        history["test_acc"].append(clean_acc)
        history["test_loss"].append(clean_loss)
        history["test_asr"].append(asr)
        if asr_multi is not None:
            history["test_asr_multi"].append(asr_multi)

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"train_loss={train_metrics['loss']:.4f} train_acc={train_metrics['acc']:.2%} | "
            f"clean={clean_acc:.2%} asr={asr:.2%}"
            + (f" asr_multi={asr_multi:.2%}" if asr_multi is not None else "")
            + f" | time={epoch_time_sec:.1f}s"
            ,
            flush=True,
        )

        if asr > best_asr or (asr == best_asr and clean_acc > best_asr_acc):
            best_asr = asr
            best_asr_acc = clean_acc

        if asr >= args.min_asr_for_tradeoff and clean_acc > best_tradeoff_acc:
            best_tradeoff_acc = clean_acc
            best_tradeoff_asr = asr

        if scheduler is not None:
            scheduler.step()

    summary = {
        "run_name": run_name,
        "dataset": args.dataset,
        "freq_strategy": args.freq_strategy,
        "target_label": args.target_label,
        "epochs": args.epochs,
        "final_acc": history["test_acc"][-1],
        "final_asr": history["test_asr"][-1],
        "final_asr_multi": history["test_asr_multi"][-1] if history["test_asr_multi"] else None,
        "best_asr": best_asr,
        "best_asr_clean_acc": best_asr_acc,
        "best_tradeoff_acc": best_tradeoff_acc if best_tradeoff_acc >= 0 else None,
        "best_tradeoff_asr": best_tradeoff_asr if best_tradeoff_asr >= 0 else None,
        "config": vars(args),
    }

    with (run_dir / "history.json").open("w", encoding="utf-8") as file:
        json.dump(history, file, ensure_ascii=False, indent=2)

    with (run_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    with (run_dir / "config.json").open("w", encoding="utf-8") as file:
        json.dump(vars(args), file, ensure_ascii=False, indent=2)

    print("\n" + "=" * 72, flush=True)
    print("Training complete", flush=True)
    print(f"Final ACC : {summary['final_acc']:.2%}", flush=True)
    print(f"Final ASR : {summary['final_asr']:.2%}", flush=True)
    if summary["final_asr_multi"] is not None:
        print(f"Final ASR (multi): {summary['final_asr_multi']:.2%}", flush=True)
    print(f"Saved to: {run_dir}", flush=True)
    print("=" * 72, flush=True)


if __name__ == "__main__":
    main()
