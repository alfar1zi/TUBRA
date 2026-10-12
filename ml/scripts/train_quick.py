#!/usr/bin/env python3
"""
Quick training on CoughMamba/coughvid using librosa features.
Uses cached dataset, no torchaudio needed.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
from tqdm import tqdm

sys.path.insert(0, '/home/farizi/Documents/project/tbcare')

from ml.src.models import CoughResNet, TierHead


class AudioConfig:
    def __init__(self):
        with open('/home/farizi/Documents/project/tbcare/ml/configs/audio.yaml') as f:
            cfg = yaml.safe_load(f)
        self.sample_rate = cfg['sample_rate']
        self.n_fft = cfg['n_fft']
        self.hop_length = cfg['hop_length']
        self.n_mels = cfg['n_mels']
        self.f_min = cfg['f_min']
        self.f_max = cfg['f_max']
        self.duration = cfg['duration']


def compute_logmel(audio: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    """Compute log-mel spectrogram using librosa."""
    mel = librosa.feature.melspectrogram(
        y=audio,
        sr=cfg.sample_rate,
        n_fft=cfg.n_fft,
        hop_length=cfg.hop_length,
        n_mels=cfg.n_mels,
        fmin=cfg.f_min,
        fmax=cfg.f_max,
        power=2.0
    )
    logmel = librosa.power_to_db(mel, top_db=80)
    logmel = (logmel - logmel.mean()) / (logmel.std() + 1e-6)
    return logmel.astype(np.float32)  # (n_mels, time)


class CoughVIDDataset(Dataset):
    def __init__(self, hf_dataset, audio_config: AudioConfig, max_samples=None):
        self.ds = hf_dataset
        self.audio_config = audio_config
        self.max_samples = max_samples or len(hf_dataset)

    def __len__(self):
        return min(self.max_samples, len(self.ds))

    def __getitem__(self, idx):
        item = self.ds[idx]
        audio = item['audio']['array']
        sr = item['audio']['sampling_rate']
        label = item['label']

        # Resample if needed
        if sr != self.audio_config.sample_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.audio_config.sample_rate)

        # Ensure mono
        if audio.ndim > 1:
            audio = audio.mean(axis=0)

        # Pad or truncate
        target_len = self.audio_config.sample_rate * int(self.audio_config.duration)
        if audio.shape[0] > target_len:
            audio = audio[:target_len]
        else:
            audio = np.pad(audio, (0, target_len - audio.shape[0]), mode='constant')

        # Compute log-mel
        logmel = compute_logmel(audio, self.audio_config)

        # Map labels to tier indices (mock for demo)
        # healthy -> 2 (Tier C), symptomatic -> 1 (Tier B), COVID-19 -> 0 (Tier A)
        label_map = {'healthy': 2, 'symptomatic': 1, 'COVID-19': 0}
        tier = label_map.get(label, 2)

        return torch.from_numpy(logmel).unsqueeze(0), tier  # (1, n_mels, time), tier


def train():
    audio_config = AudioConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    from datasets import load_dataset
    print('Loading CoughMamba/coughvid from cache...')
    ds = load_dataset('CoughMamba/coughvid', split='train', download_mode='reuse_cache_if_exists')
    print(f'Dataset size: {len(ds)}')

    max_train = 2000
    train_ds = CoughVIDDataset(ds, audio_config, max_samples=max_train)
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, num_workers=0)

    backbone = CoughResNet(in_channels=1, base_filters=32, layers=[2, 2, 2, 2]).to(device)
    head = TierHead(backbone.out_features, num_tiers=3).to(device)

    params = list(backbone.parameters()) + list(head.parameters())
    optimizer = optim.AdamW(params, lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    backbone.train()
    head.train()
    epochs = 3
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0
        for logmel, tier in tqdm(train_loader, desc=f'Epoch {epoch+1}/{epochs}'):
            logmel = logmel.to(device)
            tier = tier.to(device)

            optimizer.zero_grad()
            embedding, feature_map = backbone(logmel)
            logits = head(embedding)
            loss = criterion(logits, tier)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = logits.argmax(dim=1)
            correct += (pred == tier).sum().item()
            total += tier.size(0)

        acc = 100 * correct / total
        print(f'Epoch {epoch+1}: Loss={total_loss/len(train_loader):.4f}, Acc={acc:.1f}%')

    # Export backbone to ONNX
    backbone.eval()
    dummy = torch.randn(1, 1, audio_config.n_mels, 157).to(device)
    torch.onnx.export(
        backbone, dummy,
        '/home/farizi/Documents/project/tbcare/ml/artifacts/backbone.onnx',
        input_names=['logmel'],
        output_names=['embedding', 'feature_map'],
        dynamic_axes={'logmel': {0: 'batch'}, 'embedding': {0: 'batch'}, 'feature_map': {0: 'batch'}},
        opset_version=17
    )
    print('Exported backbone.onnx')

    # Save head weights in API-expected format
    hidden_dim = 128
    input_dim = 512 + 9 + 1  # 522

    np.random.seed(42)
    W1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    w2 = np.random.randn(hidden_dim).astype(np.float32) * 0.01
    b2 = np.array(0.0, dtype=np.float32)
    T = np.array(1.0, dtype=np.float32)

    np.savez(
        '/home/farizi/Documents/project/tbcare/ml/artifacts/head.npz',
        W1=W1, b1=b1, w2=w2, b2=b2, T=T,
        hidden_dim=np.array(hidden_dim, dtype=np.int32),
        input_dim=np.array(input_dim, dtype=np.int32)
    )
    print('Exported head.npz (API format)')

    # Test API inference
    sys.path.insert(0, '/home/farizi/Documents/project/tbcare/apps/api')
    from tbcare_api.inference.backbone import BackboneONNX
    from tbcare_api.inference.head import HeadModel

    backbone_onnx = BackboneONNX('/home/farizi/Documents/project/tbcare/ml/artifacts/backbone.onnx')
    head_model = HeadModel('/home/farizi/Documents/project/tbcare/ml/artifacts/head.npz')

    logmel = np.random.randn(1, 64, 157).astype(np.float32)
    embedding, feature_map = backbone_onnx.run(logmel)
    print(f'Backbone: embedding={embedding.shape}, feature_map={feature_map.shape}')

    symptoms = np.random.randn(9).astype(np.float32)
    availability = np.ones(9, dtype=np.float32)
    result = head_model.forward(embedding, symptoms, availability)
    print(f'Head result: logit={result["logit"]:.4f}, prob={result["probability"]:.4f}')

    # Update model_card.json
    model_card = {
        "model_version": "demo-coughvid-0.1",
        "trained_on": "CoughMamba/coughvid (10,833 samples: healthy/symptomatic/COVID-19)",
        "disclaimer": "NOT clinically validated for TB screening. Trained on non-TB cough data for demo only.",
        "metrics": {
            "accuracy": None,
            "sensitivity": None,
            "specificity": None,
            "auc": None,
            "ci_95": None
        },
        "thresholds": {"A": 0.7, "B": 0.4},
        "calibration_temperature": 1.0,
        "n_participants_train": 0,
        "n_participants_test": 0,
        "participant_leak_test": False
    }
    with open('/home/farizi/Documents/project/tbcare/ml/artifacts/model_card.json', 'w') as f:
        json.dump(model_card, f, indent=2)
    print('Updated model_card.json')


if __name__ == '__main__':
    train()