from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torchaudio


SAMPLE_RATE = 16000
DEMO_SECONDS = 6
DEMO_LENGTH = SAMPLE_RATE * DEMO_SECONDS
NUM_CLASSES = 41


def load_labels(path: Path) -> dict[str, int]:
    with open(path, "r", encoding="utf-8") as f:
        labels = json.load(f)
    if not isinstance(labels, dict):
        raise ValueError("labels.json must be a mapping from label name to class index.")
    return {str(k): int(v) for k, v in labels.items()}


def load_wav(path: Path) -> torch.Tensor:
    wav, sr = torchaudio.load(str(path))
    wav = wav.float()
    if wav.size(0) > 1:
        wav = wav.mean(dim=0, keepdim=True)
    if sr != SAMPLE_RATE:
        wav = torchaudio.functional.resample(wav, sr, SAMPLE_RATE)
    wav = wav.squeeze(0)

    if wav.numel() < DEMO_LENGTH:
        wav = torch.nn.functional.pad(wav, (0, DEMO_LENGTH - wav.numel()))
    elif wav.numel() > DEMO_LENGTH:
        wav = wav[:DEMO_LENGTH]
    return wav


def make_label(label_name: str, labels: dict[str, int]) -> torch.Tensor:
    if label_name not in labels:
        available = ", ".join(labels.keys())
        raise ValueError(f"Unknown label '{label_name}'. Available labels: {available}")
    label_id = labels[label_name]
    if label_id < 0 or label_id >= NUM_CLASSES:
        raise ValueError(f"Invalid class index for '{label_name}': {label_id}")

    label = torch.zeros(NUM_CLASSES, dtype=torch.float32)
    label[label_id] = 1.0
    return label


def save_wav(path: Path, wav: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wav = wav.detach().cpu().float().view(1, -1).clamp(-1.0, 1.0)
    torchaudio.save(str(path), wav, SAMPLE_RATE)


def main() -> None:
    parser = argparse.ArgumentParser(description="ALC-GCF inference-only demo")
    parser.add_argument("--input", required=True, type=Path, help="Path to an input mixture wav.")
    parser.add_argument("--label", required=True, type=str, help="Target class name, e.g., Snare_drum.")
    parser.add_argument("--output", required=True, type=Path, help="Path to save the extracted wav.")
    parser.add_argument("--model", default=Path("checkpoints/alc_gcf_demo.pt"), type=Path)
    parser.add_argument("--labels", default=Path("labels.json"), type=Path)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA if available. CPU is used by default.")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    labels = load_labels(args.labels)
    mixture = load_wav(args.input).unsqueeze(0).to(device)
    label = make_label(args.label, labels).unsqueeze(0).to(device)

    model = torch.jit.load(str(args.model), map_location=device)
    model.eval()

    with torch.no_grad():
        estimate = model(mixture, label).squeeze(0)

    save_wav(args.output, estimate)
    print(f"Saved output to: {args.output}")


if __name__ == "__main__":
    main()
