import argparse
import os
import hashlib
import importlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import librosa
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from train_model_torch import choose_device, set_seed

librosa_utils = importlib.import_module("librosa.util.utils")

try:  # optional dependency for OpenL3 embeddings
    import torchopenl3 as openl3
except ImportError:  # pragma: no cover
    try:
        import openl3
    except ImportError:
        openl3 = None

DEFAULT_DATASET_PATH = os.path.join("Data", "genres_original")
DEFAULT_SAVE_PATH = os.path.join("torch_models", "parallel_genre_classifier_torch.pt")
DEFAULT_CACHE_DIR = os.path.join("torch_cache", "parallel")


@dataclass
class DatasetConfig:
    sample_rate: int = 22050
    n_mels: int = 128
    n_fft: int = 2048
    hop_length: int = 512
    window_duration: int = 15
    slice_duration: int = 5
    window_stride: int = 5

    def __post_init__(self) -> None:
        if self.window_duration % self.slice_duration != 0:
            raise ValueError("window_duration must be divisible by slice_duration")

    @property
    def slices_per_window(self) -> int:
        return self.window_duration // self.slice_duration

    @property
    def window_samples(self) -> int:
        return self.sample_rate * self.window_duration

    @property
    def slice_samples(self) -> int:
        return self.sample_rate * self.slice_duration

    @property
    def stride_samples(self) -> int:
        return self.sample_rate * self.window_stride

    @property
    def expected_frames(self) -> int:
        return int(np.ceil(self.slice_samples / self.hop_length))


@dataclass
class OpenL3Config:
    embedding_dim: int = 512
    content_type: str = "music"
    input_repr: str = "mel256"
    center: bool = False
    hop_size: float = 0.5
    batch_size: int = 32

    def cache_tag(self) -> str:
        return f"openl3_{self.embedding_dim}_{self.content_type}_{self.input_repr}".replace("/", "-")


def load_openl3_model(config: OpenL3Config):
    if openl3 is None:
        raise RuntimeError("openl3 package is not installed.")
    return openl3.models.load_audio_embedding_model(
        input_repr=config.input_repr,
        content_type=config.content_type,
        embedding_size=config.embedding_dim,
    )


def list_audio_files(dataset_path: str) -> Tuple[List[str], List[int], List[str]]:
    mapping: List[str] = []
    file_paths: List[str] = []
    labels: List[int] = []

    for genre_dir in sorted(Path(dataset_path).iterdir()):
        if not genre_dir.is_dir():
            continue
        mapping.append(genre_dir.name)
        label_idx = len(mapping) - 1
        for audio_file in sorted(genre_dir.glob("*.wav")):
            file_paths.append(str(audio_file))
            labels.append(label_idx)
    if not file_paths:
        raise FileNotFoundError(f"No audio files found under {dataset_path}")
    return file_paths, labels, mapping


def pad_or_truncate(mel_db: np.ndarray, expected_frames: int) -> np.ndarray:
    if mel_db.shape[1] == expected_frames:
        return mel_db
    if mel_db.shape[1] < expected_frames:
        pad_width = expected_frames - mel_db.shape[1]
        return np.pad(mel_db, ((0, 0), (0, pad_width)), mode="constant", constant_values=-80.0)
    return mel_db[:, :expected_frames]


def compute_mel_slices(signal: np.ndarray, cfg: DatasetConfig) -> List[np.ndarray]:
    slices: List[np.ndarray] = []
    for idx in range(cfg.slices_per_window):
        start = idx * cfg.slice_samples
        end = start + cfg.slice_samples
        seg = signal[start:end]
        mel = librosa.feature.melspectrogram(
            y=seg,
            sr=cfg.sample_rate,
            n_fft=cfg.n_fft,
            hop_length=cfg.hop_length,
            n_mels=cfg.n_mels,
        )
        mel_db = librosa.power_to_db(mel, ref=np.max, top_db=None)
        mel_db = pad_or_truncate(mel_db, cfg.expected_frames)
        slices.append(mel_db.astype(np.float32, copy=False))
    return slices


def _iter_window_chunks(signal: np.ndarray, cfg: DatasetConfig) -> List[np.ndarray]:
    if signal.size < cfg.window_samples:
        fixed = librosa_utils.fix_length(signal, size=cfg.window_samples)
        return [fixed]

    max_start = signal.size - cfg.window_samples
    start_positions = list(range(0, max_start + 1, cfg.stride_samples))
    if not start_positions or start_positions[-1] != max_start:
        start_positions.append(max_start)

    chunks: List[np.ndarray] = []
    for start in start_positions:
        chunk = signal[start : start + cfg.window_samples]
        if chunk.size < cfg.window_samples:
            chunk = librosa_utils.fix_length(chunk, size=cfg.window_samples)
        chunks.append(chunk)
    return chunks


def _compute_openl3_slices(
    signal: np.ndarray,
    cfg: DatasetConfig,
    l3_config: OpenL3Config,
    l3_model,
) -> Optional[np.ndarray]:
    if openl3 is None:
        raise RuntimeError("openl3 package is required for OpenL3 feature extraction but is not installed")

    slices: List[np.ndarray] = []
    for idx in range(cfg.slices_per_window):
        start = idx * cfg.slice_samples
        end = start + cfg.slice_samples
        seg = signal[start:end]
        if seg.size < cfg.slice_samples:
            seg = librosa_utils.fix_length(seg, size=cfg.slice_samples)

        embeddings, _ = openl3.get_audio_embedding(
            seg,
            sr=cfg.sample_rate,
            hop_size=l3_config.hop_size,
            center=l3_config.center,
            batch_size=l3_config.batch_size,
            model=l3_model,
        )
        if hasattr(embeddings, "cpu"):
            embeddings = embeddings.cpu().numpy()
        if embeddings.size == 0:
            return None
        # embeddings shape is (1, Time, Dim)
        # We want to average over Time to get (Dim,) for the slice
        slice_emb = embeddings[0].mean(axis=0).astype(np.float32, copy=False)
        slices.append(slice_emb)

    return np.stack(slices, axis=0)


def compute_windows_for_track(
    file_path: str,
    cfg: DatasetConfig,
    feature_type: str,
    l3_config: Optional[OpenL3Config] = None,
    l3_model=None,
    *,
    offset: Optional[float] = None,
    duration: Optional[float] = None,
    max_windows: Optional[int] = None,
) -> List[np.ndarray]:
    load_kwargs = {"sr": cfg.sample_rate, "mono": True}
    if offset is not None and offset > 0:
        load_kwargs["offset"] = max(offset, 0.0)
    if duration is not None and duration > 0:
        load_kwargs["duration"] = max(duration, 0.0)

    try:
        signal, _ = librosa.load(file_path, **load_kwargs)
    except Exception:
        return []

    windows: List[np.ndarray] = []
    chunks = _iter_window_chunks(signal, cfg)

    for chunk in chunks:
        if feature_type == "mel":
            slices = compute_mel_slices(chunk, cfg)
            windows.append(np.stack([slc[None, ...] for slc in slices], axis=0))
        elif feature_type == "openl3":
            if l3_config is None or l3_model is None:
                raise ValueError("OpenL3 configuration and model must be provided when feature_type='openl3'")
            l3_slices = _compute_openl3_slices(chunk, cfg, l3_config, l3_model)
            if l3_slices is None:
                continue
            windows.append(l3_slices)
        else:
            raise ValueError(f"Unsupported feature_type '{feature_type}'")

    if max_windows is not None and max_windows > 0:
        return windows[:max_windows]

    return windows


def _track_cache_name(track_path: str) -> str:
    digest = hashlib.sha1(track_path.encode("utf-8")).hexdigest()
    return f"{digest}.npz"


def build_split_features(
    tracks: Sequence[Tuple[str, int]],
    cfg: DatasetConfig,
    cache_dir: str | None,
    split_name: str,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
    l3_model,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    aggregate_cache = None
    track_cache_dir = None
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_suffix = f"{feature_type}_{cfg.window_duration}s_{cfg.slice_duration}s_stride{cfg.window_stride}"
        if feature_type == "openl3" and l3_config is not None:
            cache_suffix += f"_{l3_config.cache_tag()}"
        aggregate_cache = os.path.join(
            cache_dir,
            f"parallel_{split_name}_{cache_suffix}.npz",
        )
        track_cache_dir = os.path.join(cache_dir, f"{split_name}_tracks_{cache_suffix}")
        os.makedirs(track_cache_dir, exist_ok=True)

    if aggregate_cache and os.path.exists(aggregate_cache):
        data = np.load(aggregate_cache)
        if "track_ids" in data:
            return data["X"], data["y"], data["track_ids"]
        # Compatibility fallback if cache doesn't have track_ids, though we recommend clearing cache
        print(f"Warning: Cache {aggregate_cache} missing track_ids. Aggregation will be skipped.")
        return data["X"], data["y"], np.zeros(len(data["y"]), dtype=np.int64)

    features: List[np.ndarray] = []
    labels: List[int] = []
    track_ids: List[int] = []  # New: Store unique ID for each track
    processed = 0
    total_tracks = len(tracks)
    interrupted = False

    try:
        # Enumerate to get unique track ID
        for t_idx, (file_path, label) in enumerate(tracks):
            cache_file = None
            if track_cache_dir:
                cache_file = os.path.join(track_cache_dir, _track_cache_name(file_path))
            if cache_file and os.path.exists(cache_file):
                cached = np.load(cache_file)
                cached_windows = cached["X"]
                windows = [np.array(w, dtype=np.float32, copy=False) for w in np.asarray(cached_windows)]
            else:
                windows = compute_windows_for_track(file_path, cfg, feature_type, l3_config, l3_model)
                if cache_file and windows:
                    np.savez_compressed(cache_file, X=np.asarray(windows, dtype=np.float32))

            if not windows:
                processed += 1
                continue

            features.extend(windows)
            labels.extend([label] * len(windows))
            track_ids.extend([t_idx] * len(windows))  # Assign same ID to all clips from this song
            processed += 1
            if processed % 10 == 0 or processed == total_tracks:
                print(f"Processed {processed}/{total_tracks} tracks for '{split_name}' split")
    except KeyboardInterrupt:
        interrupted = True

    if not features:
        raise RuntimeError(f"No windows were generated for split '{split_name}'")

    X = np.stack(features).astype(np.float32)
    y = np.array(labels, dtype=np.int64)
    t_ids = np.array(track_ids, dtype=np.int64)

    if aggregate_cache and not interrupted:
        np.savez_compressed(aggregate_cache, X=X, y=y, track_ids=t_ids)

    if interrupted:
        print(
            "KeyboardInterrupt detected while processing feature cache. "
            "Partial per-track caches were saved; rerun to resume."
        )
        raise KeyboardInterrupt

    return X, y, t_ids


class SqueezeExcitation(nn.Module):
    def __init__(self, channels: int, reduction: int = 16) -> None:
        super().__init__()
        reduced = max(channels // reduction, 1)
        self.sequential = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, reduced, kernel_size=1, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(reduced, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = self.sequential(x)
        return x * scale


class ResidualSEBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.se = SqueezeExcitation(out_channels)
        self.activation = nn.ReLU(inplace=True)
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.activation(out)
        return out


class MelSliceEncoder(nn.Module):
    def __init__(self, embedding_dim: int) -> None:
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.layer1 = ResidualSEBlock(32, 64, stride=2)
        self.layer2 = ResidualSEBlock(64, 128, stride=2)
        self.layer3 = ResidualSEBlock(128, 128, stride=2)

        self.dropout = nn.Dropout(0.3)
        self.project = nn.Sequential(
            nn.Linear(256, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        avg_pool = F.adaptive_avg_pool2d(x, 1).flatten(1)
        max_pool = F.adaptive_max_pool2d(x, 1).flatten(1)
        combined = torch.cat([avg_pool, max_pool], dim=1)
        combined = self.dropout(combined)
        return self.project(combined)


class EmbeddingSliceEncoder(nn.Module):
    def __init__(self, input_dim: int, embedding_dim: int) -> None:
        super().__init__()
        hidden = max(embedding_dim * 2, 128)
        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(hidden, embedding_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ParallelCNN(nn.Module):
    def __init__(
        self,
        num_slices: int,
        num_classes: int,
        embedding_dim: int = 128,
        shared_backbone: bool = True,
        feature_type: str = "mel",
        input_feature_dim: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.num_slices = num_slices
        self.shared_backbone = shared_backbone
        self.feature_type = feature_type

        if feature_type == "mel":
            if shared_backbone:
                self.backbone = MelSliceEncoder(embedding_dim)
            else:
                self.branches = nn.ModuleList([MelSliceEncoder(embedding_dim) for _ in range(num_slices)])
        elif feature_type == "openl3":
            if input_feature_dim is None:
                raise ValueError("input_feature_dim must be provided when feature_type='openl3'")
            if shared_backbone:
                self.backbone = EmbeddingSliceEncoder(input_feature_dim, embedding_dim)
            else:
                self.branches = nn.ModuleList(
                    [EmbeddingSliceEncoder(input_feature_dim, embedding_dim) for _ in range(num_slices)]
                )
        else:
            raise ValueError(f"Unsupported feature_type '{feature_type}'")
        heads = 8
        while embedding_dim % heads != 0 and heads > 1:
            heads //= 2
        self.mha_heads = max(heads, 1)
        self.slice_norm = nn.LayerNorm(embedding_dim)
        self.multihead = nn.MultiheadAttention(embedding_dim, num_heads=self.mha_heads, batch_first=True)

        attn_hidden = max(embedding_dim // 2, 1)
        self.pool_attention = nn.Sequential(
            nn.Linear(embedding_dim, attn_hidden),
            nn.ReLU(inplace=True),
            nn.Linear(attn_hidden, 1),
        )

        fusion_input = embedding_dim * 3
        self.fusion_dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(fusion_input, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(512, num_classes),
        )

    def encode_slice(self, x: torch.Tensor, idx: int) -> torch.Tensor:
        if self.shared_backbone:
            return self.backbone(x)
        return self.branches[idx](x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embeddings = [self.encode_slice(x[:, idx], idx) for idx in range(self.num_slices)]
        stacked = torch.stack(embeddings, dim=1)

        normalized = self.slice_norm(stacked)
        attn_output, _ = self.multihead(normalized, normalized, normalized)
        attn_output = self.slice_norm(attn_output)

        attn_scores = self.pool_attention(attn_output)
        attn_weights = torch.softmax(attn_scores, dim=1)
        attn_pooled = (attn_output * attn_weights).sum(dim=1)

        avg_pooled = attn_output.mean(dim=1)
        max_pooled = attn_output.max(dim=1).values

        fused = torch.cat([attn_pooled, avg_pooled, max_pooled], dim=1)
        fused = self.fusion_dropout(fused)
        return self.head(fused)


def apply_music_augment(batch: torch.Tensor) -> torch.Tensor:
    """
    Music-Aware Data Augmentation for pre-calculated Mel-spectrograms.
    Only applies Time-Stretch and Pitch-Shift using tensor operations.
    """
    # Clone to avoid inplace modification of original batch
    aug = batch.clone()
    
    # SAFETY CHECK: If input is OpenL3 embeddings (3D: Batch, Slices, Dim), 
    # we cannot apply Spectrogram Augmentations (Time Stretch/Pitch Shift).
    # Return unchanged or apply Simple Dropout/Noise if desired.
    if aug.ndim == 3:
        # OpenL3 case: [Batch, Slices, EmbDim]
        # We cannot pitch-shift/time-stretch embeddings.
        return aug

    # 1. Time-Stretch (via Interpolation)
    # Range: +/- 5% (0.95 to 1.05)
    stretch_factor = 0.95 + (0.1 * torch.rand(1, device=batch.device).item())
    
    # Handle 4D (B, S, F, T) and 5D (B, S, C, F, T)
    if aug.ndim == 4:
        B, S, Freq, Time = aug.shape
        C = 1
        reshaped = aug.view(B * S, C, Freq, Time)
    elif aug.ndim == 5:
        B, S, C, Freq, Time = aug.shape
        reshaped = aug.view(B * S, C, Freq, Time)
    else:
        # Unexpected shape
        return aug
    
    new_time = int(Time * stretch_factor)
    stretched = F.interpolate(reshaped, size=(Freq, new_time), mode='bilinear', align_corners=False)
    
    # Crop or Pad back to original Time
    if new_time > Time:
        # Crop center
        start = (new_time - Time) // 2
        stretched = stretched[:, :, :, start : start + Time]
    elif new_time < Time:
        # Pad with zeros (or reflection)
        pad_total = Time - new_time
        pad_l = pad_total // 2
        pad_r = pad_total - pad_l
        stretched = F.pad(stretched, (pad_l, pad_r))
        
    if aug.ndim == 4:
        aug = stretched.view(B, S, Freq, Time)
    else:
        aug = stretched.view(B, S, C, Freq, Time)

    # 2. Pitch-Shift (via Roll along frequency axis)
    # Range: +/- 1 semitone. 1 semitone in 128-bin Mel (~22k) roughly corresponds to 1-2 bins?
    # Actually, a semitone shift is discrete in Mel bins. 
    # For standard Mel scale, bins are roughly perceptually spaced. We'll shift +/- 2 bins.
    
    # Random integer shift between -2 and 2 (inclusive)
    shift_bins = torch.randint(-2, 3, (B,), device=batch.device)
    
    for b in range(B):
        s = shift_bins[b].item()
        if s != 0:
            # Shift frequency axis (dim -2)
            aug[b] = torch.roll(aug[b], shifts=int(s), dims=-2)
            # Zero out rolled-around parts to avoid artifacts
            if s > 0:
                if aug.ndim == 4:
                    aug[b, :, :s, :] = -80.0
                else:
                    aug[b, :, :, :s, :] = -80.0
            else:
                if aug.ndim == 4:
                    aug[b, :, s:, :] = -80.0
                else:
                    aug[b, :, :, s:, :] = -80.0
                
    return aug


def train_epoch(
    model: nn.Module,
    device: torch.device,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    use_mixed_precision: bool,
    augment: bool,
) -> Tuple[float, float]:
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch in loader:
        if len(batch) == 3:
            inputs, targets, _ = batch
        else:
            inputs, targets = batch
        inputs = inputs.to(device)
        targets = targets.to(device).float()
        if augment:
            inputs = apply_music_augment(inputs)
        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_mixed_precision, dtype=torch.float16):
            outputs = model(inputs)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item() * inputs.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= 0.5)
        # consider a sample correct if any predicted label matches any true label
        match = (preds & (targets.bool())).any(dim=1).sum().item()
        correct += match
        total += targets.size(0)
    return running_loss / total, correct / total


def evaluate(
    model: nn.Module, 
    device: torch.device, 
    loader: DataLoader, 
    criterion: nn.Module
) -> Tuple[float, float, float]:
    model.eval()
    loss_accum = 0.0
    correct = 0
    total = 0
    
    # Containers for Song-Level Aggregation
    all_probs = []
    all_targets = []
    all_track_ids = []

    with torch.inference_mode():
        for batch in loader:
            # Handle unpacked batch depending on if track_ids are present
            if len(batch) == 3:
                inputs, targets, track_ids = batch
            else:
                inputs, targets = batch
                track_ids = torch.zeros(targets.size(0), dtype=torch.long) # Dummy if missing

            inputs = inputs.to(device)
            targets = targets.to(device).float()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss_accum += loss.item() * inputs.size(0)

            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5)
            match = (preds & (targets.bool())).any(dim=1).sum().item()
            correct += match
            total += targets.size(0)
            
            # Collect for aggregation
            all_probs.append(probs.cpu())
            all_targets.append(targets.cpu())
            all_track_ids.append(track_ids)

    # Calculate Clip-Level Accuracy
    clip_acc = correct / total
    
    # Calculate Song-Level Aggregation Accuracy
    full_probs = torch.cat(all_probs)
    full_targets = torch.cat(all_targets)
    full_track_ids = torch.cat(all_track_ids)
    
    unique_tracks = torch.unique(full_track_ids)
    song_correct = 0
    song_total = 0
    
    for t_id in unique_tracks:
        # Find all clips for this track
        mask = (full_track_ids == t_id)
        if not mask.any(): continue
        
        track_probs = full_probs[mask]  # (N_clips, classes)
        track_target = full_targets[mask][0] # All clips share same target
        
        # Mean aggregation
        avg_prob = track_probs.mean(dim=0) # (classes,)
        
        # Prediction (Multi-label or Single-label logic?)
        # User requested: Final song label = argmax(mean_probs) -> Implies Single Label Evaluation for Song
        # But ground truth is one-hot.
        
        pred_label = avg_prob.argmax().item()
        true_label = track_target.argmax().item()
        
        if pred_label == true_label:
            song_correct += 1
        song_total += 1
        
    song_acc = song_correct / song_total if song_total > 0 else 0.0

    return loss_accum / total, clip_acc, song_acc


def fine_tune_model(
    model: nn.Module,
    device: torch.device,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    use_mixed_precision: bool,
    augment: bool,
    epochs: int,
    lr: float,
    weight_decay: float,
    patience: int,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_train_loss = float("inf")
    best_train_acc = 0.0
    best_epoch = 0
    patience_ctr = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_mixed_precision,
            augment,
        )
        val_loss, val_acc, val_song_acc = evaluate(model, device, val_loader, criterion)

        print(
            f"FineTune Epoch {epoch:03d}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Song Acc: {val_song_acc*100:5.2f}%"
        )

        improved = val_loss + 1e-4 < best_val_loss
        if improved:
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_train_loss = train_loss
            best_train_acc = train_acc
            best_epoch = epoch
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print("Fine-tune early stopping triggered.")
                break

    model.load_state_dict(best_state)
    state = best_state
    stats = {
        "best_epoch": float(best_epoch),
        "train_loss": float(best_train_loss),
        "train_acc": float(best_train_acc),
        "val_loss": float(best_val_loss),
        "val_acc": float(best_val_acc),
    }
    return state, stats


def create_dataloaders(
    dataset_path: str,
    cfg: DatasetConfig,
    cache_dir: str | None,
    batch_size: int,
    seed: int,
    feature_type: str,
    l3_config: Optional[OpenL3Config],
) -> Tuple[Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], List[str]]:
    file_paths, labels, mapping = list_audio_files(dataset_path)
    train_paths, test_paths, train_labels, test_labels = train_test_split(
        file_paths,
        labels,
        test_size=0.2,
        random_state=seed,
        stratify=labels,
    )
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        train_paths,
        train_labels,
        test_size=0.2,
        random_state=seed,
        stratify=train_labels,
    )
    train_tracks = list(zip(train_paths, train_labels))
    val_tracks = list(zip(val_paths, val_labels))
    test_tracks = list(zip(test_paths, test_labels))
    l3_model = None
    if feature_type == "openl3":
        if l3_config is None:
            raise ValueError("OpenL3 configuration must be provided when feature_type='openl3'")
        if openl3 is None:
            raise RuntimeError("openl3 package is required for feature_type 'openl3' but is not installed")
        l3_model = load_openl3_model(l3_config)
    X_train, y_train, t_ids_train = build_split_features(train_tracks, cfg, cache_dir, "train", feature_type, l3_config, l3_model)
    X_val, y_val, t_ids_val = build_split_features(val_tracks, cfg, cache_dir, "val", feature_type, l3_config, l3_model)
    X_test, y_test, t_ids_test = build_split_features(test_tracks, cfg, cache_dir, "test", feature_type, l3_config, l3_model)

    num_classes = len(mapping)
    def to_onehot(y_arr: np.ndarray) -> np.ndarray:
        out = np.zeros((y_arr.shape[0], num_classes), dtype=np.float32)
        out[np.arange(y_arr.shape[0]), y_arr] = 1.0
        return out

    y_train_oh = to_onehot(y_train)
    y_val_oh = to_onehot(y_val)
    y_test_oh = to_onehot(y_test)

    return (X_train, y_train_oh, t_ids_train), (X_val, y_val_oh, t_ids_val), (X_test, y_test_oh, t_ids_test), mapping


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.alpha = alpha

    def forward(self, inputs, targets):
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)
        focal_loss = (1 - pt) ** self.gamma * bce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def to_dataloader(features: np.ndarray, labels: np.ndarray, track_ids: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    tensors = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels), torch.from_numpy(track_ids))
    return DataLoader(tensors, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train parallel CNN genre classifier with slice windows")
    parser.add_argument("--dataset-path", type=str, default=DEFAULT_DATASET_PATH, help="Path to genre audio dataset")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=48, help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Optimizer weight decay")
    parser.add_argument("--gpu-index", type=int, help="CUDA device index to use")
    parser.add_argument("--cpu", action="store_true", help="Force CPU even if CUDA is available")
    parser.add_argument("--no-mixed", action="store_true", help="Disable mixed precision training")
    parser.add_argument("--save-path", type=str, default=DEFAULT_SAVE_PATH, help="Where to store the checkpoint")
    parser.add_argument("--cache-dir", type=str, default=DEFAULT_CACHE_DIR, help="Directory for feature cache files")
    parser.add_argument("--window-duration", type=int, default=15, help="Window size in seconds for each sample")
    parser.add_argument("--slice-duration", type=int, default=5, help="Duration in seconds per slice")
    parser.add_argument("--window-stride", type=int, default=5, help="Stride in seconds between windows")
    parser.add_argument("--embedding-dim", type=int, default=160, help="Embedding dimension per slice encoder")
    parser.add_argument("--no-shared-backbone", action="store_true", help="Use independent CNN per slice")
    parser.add_argument("--feature-type", choices=("mel", "openl3"), default="mel", help="Feature extractor to use")
    parser.add_argument("--openl3-embedding-dim", type=int, default=512, help="OpenL3 embedding size (ignored for mel)")
    parser.add_argument("--openl3-content-type", type=str, default="music", help="OpenL3 content type (music/environmental)")
    parser.add_argument("--openl3-input-repr", type=str, default="mel256", help="OpenL3 input representation")
    parser.add_argument("--openl3-hop-size", type=float, default=0.5, help="Hop size in seconds for OpenL3 embeddings")
    parser.add_argument("--openl3-center", action="store_true", help="Use centered frames for OpenL3 slicing")
    parser.add_argument("--openl3-batch-size", type=int, default=32, help="Batch size for OpenL3 embedding extraction")
    parser.add_argument("--augment", action="store_true", help="Enable SpecAugment-like transformations")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed")
    parser.add_argument("--fine-tune-epochs", type=int, default=5, help="Fine-tuning epochs after main training (0 disables)")
    parser.add_argument("--fine-tune-lr", type=float, default=1e-4, help="Learning rate for fine-tuning stage")
    parser.add_argument("--fine-tune-weight-decay", type=float, default=1e-5, help="Weight decay during fine-tuning")
    parser.add_argument("--fine-tune-patience", type=int, default=3, help="Early stopping patience for fine-tuning")
    parser.add_argument("--fine-tune-augment", action="store_true", help="Apply augmentation while fine-tuning")
    parser.add_argument("--loss-type", choices=("ce", "focal"), default="ce", help="Loss function type (ce or focal)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    set_seed(args.seed)
    cfg = DatasetConfig(
        window_duration=args.window_duration,
        slice_duration=args.slice_duration,
        window_stride=args.window_stride,
    )

    feature_type = args.feature_type
    l3_config: Optional[OpenL3Config] = None
    if feature_type == "openl3":
        if openl3 is None:
            raise RuntimeError(
                "openl3 package is not installed. Install it or switch --feature-type back to 'mel'."
            )
        l3_config = OpenL3Config(
            embedding_dim=args.openl3_embedding_dim,
            content_type=args.openl3_content_type,
            input_repr=args.openl3_input_repr,
            center=bool(args.openl3_center),
            hop_size=float(args.openl3_hop_size),
            batch_size=int(args.openl3_batch_size),
        )
        if args.augment:
            print("ℹ SpecAugment disabled for OpenL3 features.")
            args.augment = False
        if args.fine_tune_augment:
            print("ℹ Fine-tune augmentation disabled for OpenL3 features.")
            args.fine_tune_augment = False

    cache_dir = args.cache_dir if args.cache_dir else None
    (X_train, y_train, t_ids_train), (X_val, y_val, t_ids_val), (X_test, y_test, t_ids_test), mapping = create_dataloaders(
        args.dataset_path,
        cfg,
        cache_dir,
        args.batch_size,
        args.seed,
        feature_type,
        l3_config,
    )

    print(
        f"Loaded dataset with {len(mapping)} genres | feature: {feature_type} | "
        f"train samples: {X_train.shape[0]}, val: {X_val.shape[0]}, test: {X_test.shape[0]}"
    )

    train_loader = to_dataloader(X_train, y_train, t_ids_train, args.batch_size, shuffle=True)
    val_loader = to_dataloader(X_val, y_val, t_ids_val, args.batch_size, shuffle=False)
    test_loader = to_dataloader(X_test, y_test, t_ids_test, args.batch_size, shuffle=False)

    input_feature_dim: Optional[int] = None
    if feature_type == "openl3":
        input_feature_dim = int(X_train.shape[-1])

    device = choose_device(args.gpu_index, args.cpu)
    use_mixed_precision = torch.cuda.is_available() and not args.no_mixed and not args.cpu
    if use_mixed_precision:
        print("✓ Mixed precision enabled")
    else:
        print("ℹ Mixed precision disabled")

    model = ParallelCNN(
        num_slices=cfg.slices_per_window,
        num_classes=len(mapping),
        embedding_dim=args.embedding_dim,
        shared_backbone=not args.no_shared_backbone,
        feature_type=feature_type,
        input_feature_dim=input_feature_dim,
    ).to(device)
    
    if args.resume:
        print(f"✓ Resuming from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint) # Support loading raw state dict too
    torch.backends.cudnn.benchmark = True

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.loss_type == "focal":
        print("✓ Using Focal Loss (gamma=2.0)")
        criterion = FocalLoss(gamma=2.0)
    else:
        print("✓ Using Standard BCE Loss")
        criterion = nn.BCEWithLogitsLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=use_mixed_precision)

    best_val_acc = 0.0
    best_state = None

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_epoch(
            model,
            device,
            train_loader,
            criterion,
            optimizer,
            scaler,
            use_mixed_precision,
            augment=args.augment,
        )
        val_loss, val_acc, val_song_acc = evaluate(model, device, val_loader, criterion)
        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc*100:5.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc*100:5.2f}% | Song Acc: {val_song_acc*100:5.2f}%"
        )
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}

    if best_state is None:
        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    fine_tune_stats: Dict[str, float] | None = None
    if args.fine_tune_epochs > 0:
        print("\nStarting fine-tuning stage...")
        fine_state, fine_tune_stats = fine_tune_model(
            model,
            device,
            train_loader,
            val_loader,
            criterion,
            use_mixed_precision,
            augment=args.fine_tune_augment,
            epochs=args.fine_tune_epochs,
            lr=args.fine_tune_lr,
            weight_decay=args.fine_tune_weight_decay,
            patience=max(1, args.fine_tune_patience),
        )
        best_state = fine_state
        model.load_state_dict(best_state)
        best_val_acc = max(best_val_acc, fine_tune_stats.get("val_acc", best_val_acc))

    test_loss, test_acc, test_song_acc = evaluate(model, device, test_loader, criterion)
    print(f"\nFinal Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc*100:5.2f}% | Song Accuracy: {test_song_acc*100:5.2f}%")

    metadata = {
        "trained_at": datetime.utcnow().isoformat() + "Z",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "weight_decay": args.weight_decay,
        "val_accuracy": float(best_val_acc),
        "test_accuracy": float(test_acc),
        "window_duration": cfg.window_duration,
        "slice_duration": cfg.slice_duration,
        "window_stride": cfg.window_stride,
        "embedding_dim": args.embedding_dim,
        "shared_backbone": not args.no_shared_backbone,
        "augment": bool(args.augment),
        "multihead_heads": int(getattr(model, "mha_heads", 1)),
        "feature_type": feature_type,
    }

    if fine_tune_stats is not None:
        metadata["fine_tune"] = {
            "epochs": int(args.fine_tune_epochs),
            "learning_rate": float(args.fine_tune_lr),
            "weight_decay": float(args.fine_tune_weight_decay),
            "patience": int(max(1, args.fine_tune_patience)),
            "augment": bool(args.fine_tune_augment),
            "best_epoch": float(fine_tune_stats.get("best_epoch", 0.0)),
            "train_loss": float(fine_tune_stats.get("train_loss", 0.0)),
            "train_accuracy": float(fine_tune_stats.get("train_acc", 0.0)),
            "val_loss": float(fine_tune_stats.get("val_loss", 0.0)),
            "val_accuracy": float(fine_tune_stats.get("val_acc", 0.0)),
        }

    if l3_config is not None:
        metadata["openl3"] = {
            "embedding_dim": int(l3_config.embedding_dim),
            "content_type": l3_config.content_type,
            "input_repr": l3_config.input_repr,
            "hop_size": float(l3_config.hop_size),
            "center": bool(l3_config.center),
            "batch_size": int(l3_config.batch_size),
        }

    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "mapping": mapping,
            "meta": metadata,
        },
        args.save_path,
    )
    print(f"✓ Model saved to {args.save_path}")


if __name__ == "__main__":
    main()
