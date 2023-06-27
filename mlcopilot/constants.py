import os
from pathlib import Path

__all__ = [
    "MLCOPILOT_DB_PATH",
    "TOP_K",
    "EMBED_DIM",
    "bin_map",
    "inverse_bin_map",
    "q_num",
    "MLCOPILOT_DB_BACKEND",
    "MLCOPILOT_DB_NAME",
    "MLCOPILOT_DB_HOST",
    "MLCOPILOT_DB_PORT",
    "MLCOPILOT_DB_USER",
    "MLCOPILOT_DB_PASSWORD",
]

TOP_K = 3
EMBED_DIM = 1536
TOKEN_LIMIT = 4096
TOKEN_COMPLETION_LIMIT = 800

MLCOPILOT_DB_BACKEND = os.environ.get("MLCOPILOT_DB_BACKEND", "sqlite")

MLCOPILOT_DB_PATH = Path(
    os.environ.get("MLCOPILOT_DB_PATH", Path.home() / ".mlcopilot" / "mlcopilot.db")
).expanduser()

MLCOPILOT_DB_NAME = os.environ.get("MLCOPILOT_DB_NAME", "mlcopilot")
MLCOPILOT_DB_HOST = os.environ.get("MLCOPILOT_DB_HOST", "localhost")
MLCOPILOT_DB_PORT = os.environ.get("MLCOPILOT_DB_PORT", 5432)
MLCOPILOT_DB_USER = os.environ.get("MLCOPILOT_DB_USER", "postgres")
MLCOPILOT_DB_PASSWORD = os.environ.get("MLCOPILOT_DB_PASSWORD", "")

bin_map = {
    0.1: "very small",
    0.3: "small",
    0.5: "medium",
    0.7: "large",
    0.9: "very large",
}

inverse_bin_map = {v: k for k, v in bin_map.items()}
inverse_bin_map.update(
    {
        "very low": 0.1,
        "low": 0.3,
        "high": 0.7,
        "very high": 0.9,
        "extremely large": 0.9,
        "extremely small": 0.1,
        "extra large": 0.9,
        "extra small": 0.1,
        "very medium": 0.5,
        "very small": 0.1,
        "small": 0.3,
        "large": 0.7,
        "very large": 0.9,
    }
)

q_num = sorted(list(bin_map.keys()))
