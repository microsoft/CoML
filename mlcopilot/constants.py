import os
from pathlib import Path

__all__ = [
    "MLCOPILOT_DB_PATH",
    "TOP_K",
    "EMBED_DIM",
    "bin_map",
    "inverse_bin_map",
    "q_num",
]

TOP_K = 3
EMBED_DIM = 1536


MLCOPILOT_DB_PATH = Path(
    os.environ.get("MLCOPILOT_DB_PATH", Path.home() / ".mlcopilot" / "mlcopilot.db")
).expanduser()

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
