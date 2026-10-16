from dataclasses import dataclass


@dataclass
class AudioConfig:
    sample_rate: int = 16000
    n_fft: int = 400
    hop_length: int = 160
    n_mels: int = 64
    f_min: float = 50.0
    f_max: float = 8000.0
    duration: float = 5.0
    
    @property
    def target_frames(self) -> int:
        """Number of frames for 5s audio at 16kHz with 160 hop."""
        return int(self.sample_rate * self.duration / self.hop_length)


# Load from YAML if exists
import yaml
from pathlib import Path

_config_path = Path(__file__).parent / "audio.yaml"
if _config_path.exists():
    with open(_config_path) as f:
        _yaml_config = yaml.safe_load(f)
    # Update defaults
    AudioConfig.sample_rate = _yaml_config.get("sample_rate", 16000)
    AudioConfig.n_fft = _yaml_config.get("n_fft", 400)
    AudioConfig.hop_length = _yaml_config.get("hop_length", 160)
    AudioConfig.n_mels = _yaml_config.get("n_mels", 64)
    AudioConfig.f_min = _yaml_config.get("f_min", 50.0)
    AudioConfig.f_max = _yaml_config.get("f_max", 8000.0)
    AudioConfig.duration = _yaml_config.get("duration", 5.0)