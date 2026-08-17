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
try:
    from accelerate import Accelerator, DistributedDataParallelKwargs
    from accelerate.utils import gather_object
except ImportError:  # Keep backend/unit-test imports usable outside the loma env.
    Accelerator = None
    DistributedDataParallelKwargs = None

    def gather_object(value):
        return [value]
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm.auto import tqdm

from contrastive_finetuning.loading import IndexAssignedTripletDataset
from contrastive_finetuning.loma_cache import LomaFeatureCache
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
    parser.add_argument("--loma_cache", type=Path, default=None,
                        help="Benchmark-compatible fixed-keypoint cache for frozen LoMa features")
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


def unwrap_model(model: torch.nn.Module, accelerator: Accelerator | None = None) -> torch.nn.Module:
    if accelerator is not None:
        return accelerator.unwrap_model(model)
    return model.module if hasattr(model, "module") else model


def set_model_mode(
    model: torch.nn.Module,
    training: bool,
    accelerator: Accelerator | None = None,
) -> None:
    model.train(training)
    set_loma_train_mode(unwrap_model(model, accelerator), training)


def cached_feature_batch(
    cache: LomaFeatureCache,
    relative_paths: list[str],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    keypoints = []
    descriptors = []
    for relative in relative_paths:
        item_k, item_d = cache.load(relative, device)
        keypoints.append(item_k)
        descriptors.append(item_d)
    return torch.cat(keypoints, dim=0), torch.cat(descriptors, dim=0)


def cached_collated_features(
    batch: dict[str, torch.Tensor], device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    return (
        batch["keypoints"].to(device, non_blocking=True),
        batch["descriptors"].to(device, non_blocking=True),
    )


@torch.no_grad()
def evaluate_mini_index(
    model: torch.nn.Module,
    entries: list[dict],
    data_root: Path,
    device: torch.device,
    resize: int,
    num_keypoints: int,
    batch_size: int,
    feature_cache: LomaFeatureCache | None = None,
    accelerator: Accelerator | None = None,
) -> dict[str, float]:
    """Compute the RDD-compatible mini-evaluation metrics.

    The RDD trainer deliberately keeps this path cheap: it evaluates a fixed
    subset of roughly ``10 * batch_size`` index entries, samples one positive
    and one negative from each entry, and logs only mean match counts.  It
    does not run pseudo-accuracy or score each entry's complete candidate
    pool.  LoMa uses the same protocol here, while batching detector,
    descriptor, and matcher calls over each mini batch.
    """
    set_model_mode(model, False, accelerator)
    total_pos = 0.0
    total_neg = 0.0
    evaluated = 0
    batch_size = max(1, batch_size)

    process_index = accelerator.process_index if accelerator is not None else 0
    num_processes = accelerator.num_processes if accelerator is not None else 1
    local_entries = entries[process_index::num_processes]
    for start in range(0, len(local_entries), batch_size):
        batch_entries = local_entries[start:start + batch_size]
        query_paths: list[Path] = []
        positive_paths: list[Path] = []
        negative_paths: list[Path] = []
        for entry in batch_entries:
            query_paths.append(data_root / entry["query_frame"])
            positive_paths.append(data_root / random.choice(entry["positives"]))
            negative_paths.append(data_root / random.choice(entry["negatives"]))

        # Extract all three sides in batched calls.  This is the equivalent of
        # RDD's features_from_batch(...) calls and avoids one LoMa detector /
        # descriptor pass per image.
        if feature_cache is not None:
            query_k, query_d = cached_feature_batch(
                feature_cache, [entry["query_frame"] for entry in batch_entries], device
            )
            positive_k, positive_d = cached_feature_batch(
                feature_cache, [path.relative_to(data_root).as_posix() for path in positive_paths], device
            )
            negative_k, negative_d = cached_feature_batch(
                feature_cache, [path.relative_to(data_root).as_posix() for path in negative_paths], device
            )
        else:
            query_images = torch.stack([load_image(path, resize) for path in query_paths]).to(
                device, non_blocking=True
            )
            positive_images = torch.stack([load_image(path, resize) for path in positive_paths]).to(
                device, non_blocking=True
            )
            negative_images = torch.stack([load_image(path, resize) for path in negative_paths]).to(
                device, non_blocking=True
            )
            feature_model = unwrap_model(model, accelerator)
            query_k, query_d = extract_loma_features(feature_model, query_images, num_keypoints)
            positive_k, positive_d = extract_loma_features(feature_model, positive_images, num_keypoints)
            negative_k, negative_d = extract_loma_features(feature_model, negative_images, num_keypoints)

        _, positive_matches = eval_pair_scores(model, query_k, query_d, positive_k, positive_d)
        _, negative_matches = eval_pair_scores(model, query_k, query_d, negative_k, negative_d)
        total_pos += float(positive_matches.sum())
        total_neg += float(negative_matches.sum())
        evaluated += len(batch_entries)

    if accelerator is not None:
        totals = accelerator.reduce(
            torch.tensor([total_pos, total_neg, evaluated], device=device, dtype=torch.float64),
            reduction="sum",
        ).tolist()
        total_pos, total_neg, evaluated = totals
    denominator = max(1, evaluated)
    return {
        "mean_matches_pos": total_pos / denominator,
        "mean_matches_neg": total_neg / denominator,
    }


@torch.no_grad()
def evaluate_index(model: torch.nn.Module, entries: list[dict], data_root: Path, device: torch.device,
                   resize: int, num_keypoints: int, max_entries: int = 0,
                   feature_cache: LomaFeatureCache | None = None,
                   accelerator: Accelerator | None = None) -> dict[str, float]:
    set_model_mode(model, False, accelerator)
    selected = entries[:max_entries] if max_entries else entries
    process_index = accelerator.process_index if accelerator is not None else 0
    num_processes = accelerator.num_processes if accelerator is not None else 1
    selected = selected[process_index::num_processes]
    frame_correct = 0
    videos: dict[str, dict[str, object]] = {}
    score_pos_sum = 0.0
    score_neg_sum = 0.0
    match_pos_sum = 0.0
    match_neg_sum = 0.0
    n_pos_total = 0
    n_neg_total = 0
    evaluated = 0

    progress = tqdm(
        selected,
        desc="LoMa validation",
        leave=False,
        disable=accelerator is not None and not accelerator.is_main_process,
    )
    for entry in progress:
        query_rel = entry["query_frame"]
        candidate_rel = list(entry["positives"]) + list(entry["negatives"])
        if feature_cache is not None:
            query_k, query_d = feature_cache.load(query_rel, device)
            cached_candidates = [feature_cache.load(rel, device) for rel in candidate_rel]
            cand_k = torch.cat([item[0] for item in cached_candidates], dim=0)
            cand_d = torch.cat([item[1] for item in cached_candidates], dim=0)
        else:
            feature_model = unwrap_model(model, accelerator)
            query_k, query_d = load_features_for_paths(
                feature_model, [data_root / query_rel], device, resize, num_keypoints
            )
            cand_k, cand_d = load_features_for_paths(
                feature_model, [data_root / rel for rel in candidate_rel], device, resize, num_keypoints
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

    local_result = {
        "frame_correct": frame_correct,
        "score_pos_sum": score_pos_sum,
        "score_neg_sum": score_neg_sum,
        "n_pos_total": n_pos_total,
        "n_neg_total": n_neg_total,
        "evaluated": evaluated,
        "videos": videos,
    }
    # Accelerate gathers list payloads and flattens them. Wrapping the local
    # dictionary keeps one result dictionary per rank; passing the dictionary
    # directly makes the gathered result contain its string keys.
    results = gather_object([local_result]) if accelerator is not None else [local_result]
    frame_correct = sum(result["frame_correct"] for result in results)
    score_pos_sum = sum(result["score_pos_sum"] for result in results)
    score_neg_sum = sum(result["score_neg_sum"] for result in results)
    n_pos_total = sum(result["n_pos_total"] for result in results)
    n_neg_total = sum(result["n_neg_total"] for result in results)
    evaluated = sum(result["evaluated"] for result in results)
    videos = {}
    for result in results:
        for video_id, item in result["videos"].items():
            current = videos.setdefault(video_id, item.copy())
            if item["best_score"] > current["best_score"]:
                current["best_score"] = item["best_score"]
                current["best_lynx"] = item["best_lynx"]
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


def save_checkpoint(model: torch.nn.Module, optimizer: torch.optim.Optimizer,
                    scheduler: torch.optim.lr_scheduler.LRScheduler, epoch: int,
                    step: int, args: argparse.Namespace, out_dir: Path) -> None:
    from safetensors.torch import save_file

    out_dir.mkdir(parents=True, exist_ok=True)
    save_file(trainable_state_dict(model), str(out_dir / "model.safetensors"))
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch,
            "step": step,
        },
        out_dir / "optimizer.pt",
    )
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


def restore_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    path: Path,
) -> tuple[int, int]:
    from safetensors.torch import load_file

    model.load_state_dict(load_file(str(path / "model.safetensors")), strict=False)
    # This is a trusted training-state bundle, not a model supplied by an
    # external user. Explicitly disable PyTorch 2.6's weights-only default:
    # the bundle contains optimizer metadata and Python/NumPy RNG state.
    state = torch.load(path / "optimizer.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    if "scheduler" in state:
        scheduler.load_state_dict(state["scheduler"])
    else:
        # Checkpoints created before scheduler persistence was added remain
        # resumable. Reconstruct the schedule position from the last completed
        # epoch; newly written checkpoints use the exact scheduler state above.
        completed_epochs = int(state["epoch"]) + 1
        if completed_epochs > 0:
            scheduler.step(completed_epochs)
    rng_path = path / "rng_state.pt"
    if rng_path.exists():
        rng = torch.load(rng_path, map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        np.random.set_state(rng["numpy"])
        random.setstate(rng["python"])
    return int(state["epoch"]), int(state.get("step", 0))


def main() -> None:
    args = parse_args()
    if args.trained_model != "loma":
        raise ValueError("This entry point only supports --trained_model loma")
    if Accelerator is None:
        raise ImportError("Accelerate is required for LoMa training; install requirements-loma.txt")
    seed_all(args.seed)
    accelerator = Accelerator(
        kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
    )
    device = accelerator.device

    model = build_loma(args.loma_variant, args.loma_weights, device)
    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise RuntimeError("LoMa has no trainable matcher parameters")
    optimizer = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, args.epochs))

    feature_cache = None
    if args.loma_cache is not None:
        feature_cache = LomaFeatureCache(
            args.loma_cache,
            variant=args.loma_variant,
            resize=args.resize,
            num_keypoints=args.num_keypoints,
            weights=args.loma_weights,
        )
    dataset = IndexAssignedTripletDataset(
        args.train_index,
        root=args.data_root,
        transform=transforms.ToTensor(),
        random_negative_prob=args.random_negative_prob,
        feature_cache=feature_cache,
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

    # RDD's mini-evaluation uses a fixed subset selected once before training;
    # the positive and negative are sampled afresh when that subset is
    # evaluated, just as IndexAssignedTripletDataset does in the RDD trainer.
    mini_entries = max(1, 10 * args.batch_size)
    mini_rng = random.Random(args.seed)
    mini_train_entries = mini_rng.sample(train_entries, min(mini_entries, len(train_entries)))
    mini_val_entries = mini_rng.sample(val_entries, min(mini_entries, len(val_entries)))

    # New distributed checkpoints use Accelerate's per-rank state directory.
    # Older single-GPU bundles remain loadable through the legacy restore path.
    start_epoch = 0
    step = 0
    accelerator_state = args.resume / "accelerate_state" if args.resume else None
    if args.resume and not accelerator_state.exists():
        start_epoch, step = restore_checkpoint(model, optimizer, scheduler, args.resume)
        start_epoch += 1

    model, optimizer, scheduler, loader = accelerator.prepare(
        model, optimizer, scheduler, loader
    )
    if accelerator_state is not None and accelerator_state.exists():
        accelerator.load_state(str(accelerator_state))
        metadata = json.loads((args.resume / "metadata.json").read_text())
        start_epoch = int(metadata["epoch"]) + 1
        step = int(metadata.get("step", 0))

    wandb_run = None
    if accelerator.is_main_process and args.wandb_mode != "disabled":
        import wandb

        wandb_run = wandb.init(
            project=args.project,
            name=args.run_name,
            config=vars(args),
            # Pass the requested mode explicitly so an inherited WANDB_MODE
            # environment variable cannot silently turn an online run offline.
            mode=args.wandb_mode,
        )

    # Match the RDD trainer's pre-training baseline. This measures the
    # untouched pretrained matcher before any optimizer step. Full-gallery
    # retrieval remains the responsibility of rdd-parallel-benchmark.
    if args.resume is None:
        baseline_started = time.perf_counter()
        baseline_train = evaluate_index(
            model, train_entries, args.data_root, device, args.resize, args.num_keypoints,
            feature_cache=feature_cache, accelerator=accelerator,
        )
        baseline_val = evaluate_index(
            model, val_entries, args.data_root, device, args.resize, args.num_keypoints,
            feature_cache=feature_cache, accelerator=accelerator,
        )
        baseline = {
            "epoch": -1,
            "train/random_negative_prob": args.random_negative_prob,
            "time/baseline_eval_s": time.perf_counter() - baseline_started,
            **prefixed_metrics("train_eval", baseline_train),
            **prefixed_metrics("val", baseline_val),
        }
        if accelerator.is_main_process:
            print(json.dumps({"pretrained_baseline": baseline}, indent=2))
        if wandb_run is not None:
            wandb_run.log(baseline, step=step)
    set_model_mode(model, True, accelerator)

    for epoch in range(start_epoch, args.epochs):
        epoch_started = time.perf_counter()
        set_model_mode(model, True, accelerator)
        running_loss = 0.0
        running_pos = 0.0
        running_neg = 0.0
        running_pos_matches = 0.0
        running_neg_matches = 0.0
        running_active = 0.0
        batches = 0
        mini_eval_time = 0.0
        progress = tqdm(
            loader,
            desc=f"LoMa epoch {epoch}",
            disable=not accelerator.is_main_process,
        )
        for query, positive, negative in progress:
            if args.max_train_batches and batches >= args.max_train_batches:
                break
            if feature_cache is not None:
                query_k, query_d = cached_collated_features(query, device)
                positive_k, positive_d = cached_collated_features(positive, device)
                negative_k, negative_d = cached_collated_features(negative, device)
            else:
                query = batch_images(query, args.resize).to(device, non_blocking=True)
                positive = batch_images(positive, args.resize).to(device, non_blocking=True)
                negative = batch_images(negative, args.resize).to(device, non_blocking=True)
                feature_model = unwrap_model(model, accelerator)
                query_k, query_d = extract_loma_features(feature_model, query, args.num_keypoints)
                positive_k, positive_d = extract_loma_features(feature_model, positive, args.num_keypoints)
                negative_k, negative_d = extract_loma_features(feature_model, negative, args.num_keypoints)
            pos_score, pos_matches = train_pair_score_with_matches(
                model, query_k, query_d, positive_k, positive_d
            )
            neg_score, neg_matches = train_pair_score_with_matches(
                model, query_k, query_d, negative_k, negative_d
            )
            loss = F.relu(args.margin - pos_score + neg_score).mean()

            optimizer.zero_grad(set_to_none=True)
            accelerator.backward(loss)
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                args.grad_clip,
            )
            optimizer.step()

            active_frac_tensor = ((args.margin - pos_score + neg_score) > 0).float().mean()
            step_values = accelerator.reduce(
                torch.stack([
                    loss.detach(),
                    pos_score.detach().mean(),
                    neg_score.detach().mean(),
                    pos_matches.mean(),
                    neg_matches.mean(),
                    active_frac_tensor,
                ]),
                reduction="mean",
            ).tolist()
            loss_value, pos_value, neg_value, pos_match_value, neg_match_value, active_frac = step_values
            running_loss += loss_value
            running_pos += pos_value
            running_neg += neg_value
            running_pos_matches += pos_match_value
            running_neg_matches += neg_match_value
            running_active += active_frac
            batches += 1
            step += 1
            if accelerator.is_main_process:
                progress.set_postfix(loss=running_loss / batches)

            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss_value,
                        "train/consistency_loss": 0.0,
                        "train/total_loss": loss_value,
                        "train/lr_step": optimizer.param_groups[0]["lr"],
                        "train/active_frac": active_frac,
                        "matches/mean_pos": pos_match_value,
                        "matches/mean_neg": neg_match_value,
                        # Same W&B keys as RDD; these are LoMa's assignment
                        # confidence scores rather than LightGlue scores.
                        "lg_confidence/mean_pos_conf": pos_value,
                        "lg_confidence/mean_neg_conf": neg_value,
                        "progress": step / max(1, args.epochs * len(loader)),
                    },
                    step=step,
                )

            # RDD logs mini evaluations every 100 optimizer steps.
            if step % 100 == 0:
                mini_started = time.perf_counter()
                mini_train = evaluate_mini_index(
                    model, mini_train_entries, args.data_root, device, args.resize,
                    args.num_keypoints, args.batch_size,
                    feature_cache=feature_cache, accelerator=accelerator,
                )
                mini_val = evaluate_mini_index(
                    model, mini_val_entries, args.data_root, device, args.resize,
                    args.num_keypoints, args.batch_size,
                    feature_cache=feature_cache, accelerator=accelerator,
                )
                mini_eval_time += time.perf_counter() - mini_started
                set_model_mode(model, True, accelerator)
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
                args.num_keypoints, feature_cache=feature_cache, accelerator=accelerator,
            )
            val_eval = evaluate_index(
                model, val_entries, args.data_root, device, args.resize,
                args.num_keypoints, args.max_val_entries,
                feature_cache=feature_cache, accelerator=accelerator,
            )
            epoch_eval_time = time.perf_counter() - eval_started
            set_model_mode(model, True, accelerator)

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
        for output in (checkpoint_dir(args.output_dir, epoch), args.output_dir / "latest"):
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                save_checkpoint(
                    unwrap_model(model, accelerator), optimizer, scheduler,
                    epoch, step, args, output,
                )
            accelerator.wait_for_everyone()
            accelerator.save_state(str(output / "accelerate_state"))
            accelerator.wait_for_everyone()
        if wandb_run is not None:
            wandb_run.log(metrics, step=step)
        if accelerator.is_main_process:
            print(json.dumps(metrics, indent=2))

    if wandb_run is not None:
        wandb_run.finish()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
