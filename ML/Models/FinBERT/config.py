import os
from pathlib import Path


def load_env(env_path=None):
    if env_path is None:
        env_path = Path(__file__).resolve().parent / ".env"
    env_path = Path(env_path)
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"").strip("'")
        os.environ.setdefault(key, value)


def get_int(key, default=None):
    value = os.environ.get(key)
    if value is None:
        return default
    return int(value)


def get_float(key, default=None):
    value = os.environ.get(key)
    if value is None:
        return default
    return float(value)


def get_str(key, default=None):
    value = os.environ.get(key)
    if value is None:
        return default
    return value
