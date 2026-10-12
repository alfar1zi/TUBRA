#!/usr/bin/env python3
"""
Create demo ONNX artifacts for the pipeline to work end-to-end.
Does not train - just exports randomly initialized models.
Model card honestly states: demo only, not clinically validated.
"""

import os
import sys
import json
import yaml
import numpy as np
import torch
import torch.nn as nn

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


def main():
    audio_config = AudioConfig()
    device = torch.device('cpu')

    backbone = CoughResNet(in_channels=1, base_filters=64, layers=[2, 2, 2, 2]).to(device)
    head = TierHead(backbone.out_features, num_tiers=3).to(device)

    # Initialize with Xavier
    def init_weights(m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    backbone.apply(init_weights)
    head.apply(init_weights)

    # Set head bias to produce reasonable initial tier distribution
    with torch.no_grad():
        head.net[-1].bias[2] = 1.0  # bias towards Tier C (index 2)
        head.net[-1].bias[1] = 0.0
        head.net[-1].bias[0] = -1.0

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
    # API expects: W1 (hidden, input), b1 (hidden), w2 (hidden), b2 (1), T (scalar), hidden_dim, input_dim
    hidden_dim = 128
    input_dim = 512 + 9 + 1  # 522

    np.random.seed(42)
    W1 = np.random.randn(hidden_dim, input_dim).astype(np.float32) * 0.01
    b1 = np.zeros(hidden_dim, dtype=np.float32)
    b2 = np.array(0.0, dtype=np.float32)  # 0-d scalar
    T = np.array(1.0, dtype=np.float32)  # 0-d scalar
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
        "model_version": "demo-random-0.1",
        "trained_on": "Random initialization (no training). Placeholder for demo pipeline.",
        "disclaimer": "NOT clinically validated for TB screening. Random weights - produces arbitrary output.",
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
    main()