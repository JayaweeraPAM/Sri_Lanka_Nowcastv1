"""Central config loader — every script does: from src.utils.config import load_config"""
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "config.yaml"


def load_config(config_path: Path = DEFAULT_CONFIG_PATH) -> dict:
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    for key, rel_path in cfg["paths"].items():
        abs_path = PROJECT_ROOT / rel_path
        abs_path.mkdir(parents=True, exist_ok=True)
        cfg["paths"][key] = abs_path
    return cfg


if __name__ == "__main__":
    cfg = load_config()
    print("Loaded config:")
    for section, values in cfg.items():
        print(f"  {section}: {values}")
