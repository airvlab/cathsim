import yaml
import numpy as np
from pathlib import Path

config_path = Path(__file__).parent / "env_config.yaml"
with open(config_path.as_posix()) as f:
    env_config = yaml.safe_load(f)
