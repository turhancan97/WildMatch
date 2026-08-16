"""Fine-tune the LoMa matcher on the lynx positive/negative index.

This is intentionally a separate entry point from the RDD/LightGlue trainer.
LoMa's detector and descriptor are frozen; only the matcher parameters are
optimized with the same positive-versus-negative margin idea used by the
existing lynx experiments.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from contrastive_finetuning.loading import IndexAssignedTripletDataset
from contrastive_finetuning.loma_backend import (
    build_loma,
    eval_pair_scores,
    extract_loma_features,
    set_loma_train_mode,
    train_pair_score_with_matches,
    trainable_state_dict,
)


LOMA_PATCH_SIZE = 14


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune the LoMa matcher on lynx triplet indices")
    parser.add_argument("--trained_model", choices=["loma"], default="loma")
    parser.add_argument("--train_index", type=Path, required=True)
    parser.add_argument("--val_index", type=Path, required=True)
    parser.add_argument("--data_root", type=Path, required=True)
    parser.add_argument("--loma_weights", type=Path, default=None,
                        help="Pretrained LoMa checkpoint or a LoMa fine-tuning bundle")
    parser.add_argument("--loma_variant", choices=["loma-b", "loma-b128", "loma-l", "loma-g", "loma-r"], default="loma-b")
    parser.add_argument("--output_dir", type=Path, default=Path("checkpoints/loma-b"))
    parser.add_argument("--project", default="lynx-loma-finetuning")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--wandb_mode", choices=["online", "offline", "disabled"], default="online")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.5)
    parser.add_argument(
        "--random_negative_prob", type=float, default=0.0,
        help="Probability of replacing the indexed negative with a random different-lynx frame.",
    )
    parser.add_argument("--resize", type=int, default=512)
    parser.add_argument("--num_keypoints", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--eval_every_epochs", type=int, default=10)
    parser.add_argument("--max_train_batches", type=int, default=0,
                        help="Debug limit; 0 uses the full training index")
    parser.add_argument("--max_val_entries", type=int, default=0,
                        help="Debug limit; 0 evaluates the complete validation index")
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", type=Path, default=None,
                        help="Checkpoint directory containing model.safetensors and optimizer.pt")
    return parser.parse_args()


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resize_long_side(images: torch.Tensor, size: int) -> torch.Tensor:
    """Resize LoMa inputs with dimensions aligned to DINOv2's 14px patches.

    The existing RDD/LightGlue path aligns images to 32px feature-map blocks,
    but LoMa's DeDoDe descriptor contains a DINOv2 ViT-L/14 encoder.  Both
    spatial dimensions therefore need to be divisible by 14.
    """
    if size <= 0:
        return images
    _, _, height, width = images.shape
    scale = size / max(height, width)
    new_height = max(LOMA_PATCH_SIZE, int(height * scale) // LOMA_PATCH_SIZE * LOMA_PATCH_SIZE)
    new_width = max(LOMA_PATCH_SIZE, int(width * scale) // LOMA_PATCH_SIZE * LOMA_PATCH_SIZE)
    return F.interpolate(images.float(), (new_height, new_width), mode="bilinear", align_corners=False)


def load_image(path: Path, resize: int) -> torch.Tensor:
    image = transforms.ToTensor()(Image.open(path).convert("RGB")).unsqueeze(0)
    return resize_long_side(image, resize)[0]


def batch_images(images: torch.Tensor, resize: int) -> torch.Tensor:
    return resize_long_side(images, resize)


def load_entries(path: Path) -> list[dict]:
    with path.open() as handle:
        entries = json.load(handle)
    if not isinstance(entries, list):
        raise ValueError(f"{path} must contain a JSON list")
    return entries


def prefixed_metrics(prefix: str, metrics: dict[str, float]) -> dict[str, float]:
    """Use the same split prefixes as the RDD trainer's W&B metrics."""
    return {f"{prefix}/{key}": value for key, value in metrics.items()}


def load_features_for_paths(model: torch.nn.Module, paths: list[Path], device: torch.device,
                            resize: int, num_keypoints: int) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    for path in paths:
        image = load_image(path, resize).unsqueeze(0).to(device)
        keypoints, descriptors = extract_loma_features(model, image, num_keypoints)
        features.append((keypoints, descriptors))
    return torch.cat([item[0] for item in features]), torch.cat([item[1] for item in features])


@torch.no_grad()
def evaluate_index(model: torch.nn.Module, entries: list[dict], data_root: Path, device: torch.device,
                   resize: int, num_keypoints: int, max_entries: int = 0) -> dict[str, float]:
    set_loma_train_mode(model, False)
    selected = entries[:max_entries] if max_entries else entries
    frame_correct = 0
    videos: dict[str, dict[str, object]] = {}
    score_pos_sum = 0.0
    score_neg_sum = 0.0
    match_pos_sum = 0.0
    match_neg_sum = 0.0
    n_pos_total = 0
    n_neg_total = 0
    evaluated = 0

    for entry in tqdm(selected, desc="LoMa validation", leave=False):
        query_rel = entry["query_frame"]
        candidate_rel = list(entry["positives"]) + list(entry["negatives"])
        query_k, query_d = load_features_for_paths(
            model, [data_root / query_rel], device, resize, num_keypoints
        )
        cand_k, cand_d = load_features_for_paths(
            model, [data_root / rel for rel in candidate_rel], device, resize, num_keypoints
        )
        query_k = query_k.expand(cand_k.shape[0], -1, -1)
        query_d = query_d.expand(cand_d.shape[0], -1, -1)
        scores, _ = eval_pair_scores(model, query_k, query_d, cand_k, cand_d)
        n_pos = len(entry["positives"])
        pos_scores, neg_scores = scores[:n_pos], scores[n_pos:]
        frame_correct += int(float(pos_scores.max()) > float(neg_scores.max()))
        score_pos_sum += float(pos_scores.sum())
        score_neg_sum += float(neg_scores.sum())
        n_pos_total += len(pos_scores)
        n_neg_total += len(neg_scores)
        query_lynx = Path(query_rel).parts[1]
        best_index = int(scores.argmax())
        best_lynx = Path(candidate_rel[best_index]).parts[1]
        video_id = str(Path(query_rel).parent)
        current = videos.setdefault(video_id, {"true_lynx": query_lynx, "best_score": -float("inf"), "best_lynx": None})
        if float(scores[best_index]) > float(current["best_score"]):
            current["best_score"] = float(scores[best_index])
            current["best_lynx"] = best_lynx
        evaluated += 1

    denominator = max(1, evaluated)
    video_correct = sum(int(item["best_lynx"] == item["true_lynx"]) for item in videos.values())
    return {
        "frame_accuracy": frame_correct / denominator,
        "video_accuracy": video_correct / max(1, len(videos)),
        "mean_score_pos": score_pos_sum / max(1, n_pos_total),
        "mean_score_neg": score_neg_sum / max(1, n_neg_total),
        "entries": float(evaluated),
    }


def checkpoint_dir(output_dir: Path, epoch: int) -> Path:
    return output_dir / f"epoch_{epoch:03d}"


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int,
                    step: int, args: argparse.Namespace, out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state_dict(model), str(out_dir / "model.safetensors"))
    torch.save({"optimizer": optimizer.state_dict(), "epoch": epoch, "step": step}, out_dir / "optimizer.pt")
    torch.save({"torch": torch.get_rng_state(), "numpy": np.random.get_state(), "python": random.getstate()}, out_dir / "rng_state.pt")
    metadata = {
        "format": "lynx-loma-matcher-v1",
        "backend": "loma",
        "variant": args.loma_variant,
        "base_weights": str(args.loma_weights) if args.loma_weights else None,
        "epoch": epoch,
        "step": step,
        "train_index": str(args.train_index),
        "val_index": str(args.val_index),
        "resize": args.resize,
        "num_keypoints": args.num_keypoints,
        "args": vars(args),
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2, default=str))


def restore_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer, path: Path) -> tuple[int, int]:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    state = torch.load(path / "optimizer.pt", map_location="cpu")
    optimizer.load_state_dict(state["optimizer"])
    rng_path = path / "rng_state.pt"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu")
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return int(state["epoch"]), int(state.get("step", 0))


def main() -> None:
    args = parse_args()
    if args.trained_model != "loma":
        raise ValueError("This entry point only supports --trained_model loma")
    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_loma(args.loma_variant, args.loma_weights, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoMa has no trainable matcher parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    start_epoch = 0
    step = 0
    if args.resume:
        start_epoch, step = restore_checkpoint(model, optimizer, args.resume)
        start_epoch += 1

    dataset = IndexAssignedTripletDataset(
        args.train_index,
        root=args.data_root,
        transform=transforms.ToTensor(),
        random_negative_prob=args.random_negative_prob,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )
    train_entries = load_entries(args.train_index)
    val_entries = load_entries(args.val_index)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    wandb_run = None
    if args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
            mode="offline" if args.wandb_mode == "offline" else None,
        )

    # Match the RDD trainer's pre-training baseline. This measures the
    # untouched pretrained matcher before any optimizer step. Full-gallery
    # retrieval remains the responsibility of rdd-parallel-benchmark.
    if args.resume is None:
        baseline_started = time.perf_counter()
        baseline_train = evaluate_index(
            model, train_entries, args.data_root, device, args.resize, args.num_keypoints
        )
        baseline_val = evaluate_index(
            model, val_entries, args.data_root, device, args.resize, args.num_keypoints
        )
        baseline = {
            "epoch": -1,
            "train/random_negative_prob": args.random_negative_prob,
            "time/baseline_eval_s": time.perf_counter() - baseline_started,
            **prefixed_metrics("train_eval", baseline_train),
            **prefixed_metrics("val", baseline_val),
        }
        print(json.dumps({"pretrained_baseline": baseline}, indent=2))
        if wandb_run is not None:
            wandb_run.log(baseline, step=step)
    set_loma_train_mode(model, True)

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        set_loma_train_mode(model, True)
        running_loss = 0.0
        running_pos = 0.0
        running_neg = 0.0
        running_pos_matches = 0.0
        running_neg_matches = 0.0
        running_active = 0.0
        batches = 0
        mini_eval_time = 0.0
        progress = tqdm(loader, desc=f"LoMa epoch {epoch}")
        for query, positive, negative in progress:
            if args.max_train_batches and batches >= args.max_train_batches:
                break
            query = batch_images(query, args.resize).to(device, non_blocking=True)
            positive = batch_images(positive, args.resize).to(device, non_blocking=True)
            negative = batch_images(negative, args.resize).to(device, non_blocking=True)

            query_k, query_d = extract_loma_features(model, query, args.num_keypoints)
            positive_k, positive_d = extract_loma_features(model, positive, args.num_keypoints)
            negative_k, negative_d = extract_loma_features(model, negative, args.num_keypoints)
            pos_score, pos_matches = train_pair_score_with_matches(
                model, query_k, query_d, positive_k, positive_d
            )
            neg_score, neg_matches = train_pair_score_with_matches(
                model, query_k, query_d, negative_k, negative_d
            )
            loss = F.relu(args.margin - pos_score + neg_score).mean()

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable, args.grad_clip)
            optimizer.step()

            running_loss += float(loss.detach())
            running_pos += float(pos_score.detach().mean())
            running_neg += float(neg_score.detach().mean())
            running_pos_matches += float(pos_matches.mean())
            running_neg_matches += float(neg_matches.mean())
            active_frac = float(((args.margin - pos_score + neg_score) > 0).float().mean())
            running_active += active_frac
            batches += 1
            step += 1
            progress.set_postfix(loss=running_loss / batches)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": float(loss.detach()),
                        "train/consistency_loss": 0.0,
                        "train/total_loss": float(loss.detach()),
                        "train/lr_step": optimizer.param_groups[0]["lr"],
                        "train/active_frac": active_frac,
                        "matches/mean_pos": float(pos_matches.mean()),
                        "matches/mean_neg": float(neg_matches.mean()),
                        # Same W&B keys as RDD; these are LoMa's assignment
                        # confidence scores rather than LightGlue scores.
                        "lg_confidence/mean_pos_conf": float(pos_score.detach().mean()),
                        "lg_confidence/mean_neg_conf": float(neg_score.detach().mean()),
                        "progress": step / max(1, args.epochs * len(loader)),
                    },
                    step=step,
                )

            # RDD logs mini evaluations every 100 optimizer steps.
            if step % 100 == 0:
                mini_started = time.perf_counter()
                mini_entries = max(1, 10 * args.batch_size)
                mini_train = evaluate_index(
                    model, train_entries, args.data_root, device, args.resize,
                    args.num_keypoints, mini_entries
                )
                mini_val = evaluate_index(
                    model, val_entries, args.data_root, device, args.resize,
                    args.num_keypoints, mini_entries
                )
                mini_eval_time += time.perf_counter() - mini_started
                set_loma_train_mode(model, True)
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            **prefixed_metrics("mini_train", mini_train),
                            **prefixed_metrics("mini_val", mini_val),
                            "progress": step / max(1, args.epochs * len(loader)),
                        },
                        step=step,
                    )

        scheduler.step()
        do_eval = (epoch + 1) % args.eval_every_epochs == 0
        train_eval = {}
        val_eval = {}
        epoch_eval_time = 0.0
        if do_eval:
            eval_started = time.perf_counter()
            train_eval = evaluate_index(
                model, train_entries, args.data_root, device, args.resize,
                args.num_keypoints
            )
            val_eval = evaluate_index(
                model, val_entries, args.data_root, device, args.resize,
                args.num_keypoints, args.max_val_entries
            )
            epoch_eval_time = time.perf_counter() - eval_started
            set_loma_train_mode(model, True)

        epoch_loss = running_loss / max(1, batches)
        mean_pos = running_pos / max(1, batches)
        mean_neg = running_neg / max(1, batches)
        metrics = {
            "epoch": epoch,
            "train/epoch_loss": epoch_loss,
            "train/positive_score": mean_pos,
            "train/negative_score": mean_neg,
            "train/mean_matches_pos": running_pos_matches / max(1, batches),
            "train/mean_matches_neg": running_neg_matches / max(1, batches),
            "train/active_frac": running_active / max(1, batches),
            "train/step": step,
            "train/lr": optimizer.param_groups[0]["lr"],
            "time/epoch_eval_s": epoch_eval_time,
            "time/train_s": max(0.0, time.perf_counter() - epoch_started - epoch_eval_time - mini_eval_time),
            "time/mini_eval_s": mini_eval_time,
            "train/random_negative_prob": args.random_negative_prob,
            "train/margin": args.margin,
            "train/sep_gap": mean_pos - mean_neg,
            "train/skip_rate_pos_index": 0.0,
            "train/skip_rate_pos_random": 0.0,
            "train/skip_rate_neg_index": 0.0,
            "train/skip_rate_neg_random": 0.0,
            "train/skip_rate_pos_index_dead_count": 0,
        }
        if do_eval:
            metrics.update(prefixed_metrics("train_eval", train_eval))
            metrics.update(prefixed_metrics("val", val_eval))
        save_checkpoint(model, optimizer, epoch, step, args, checkpoint_dir(args.output_dir, epoch))
        save_checkpoint(model, optimizer, epoch, step, args, args.output_dir / "latest")
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        print(json.dumps(metrics, indent=2))

    if wandb_run is not None:
        wandb_run.finish()


if __name__ == "__main__":
    main()
