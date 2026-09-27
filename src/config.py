"""Configuración de recursos por máquina. Se sobreescribe en config.local.json (no versionado), sección por sección:

{
  "duckdb": {"memory_limit": "8GB", "threads": 8},
  "torch":  {"device": "cuda", "amp": true, "threads": 8},
  "gru":    {"seeds": 5, "max_epochs": 40, "patience": 3, "val_frac": 0.15, "batch": 512, "pred_batch": 4096,
             "pre_epochs": 5},
  "env":    {"HSA_OVERRIDE_GFX_VERSION": "10.3.0"}
}

torch.device: "auto" (GPU si hay), "cpu" o "cuda" (NVIDIA o AMD/ROCm; falla si no hay GPU).
env: variables de entorno que se fijan antes de inicializar la GPU (p. ej. ROCm en Radeon no listadas oficialmente).
"""
import json
import os
from pathlib import Path

LOCAL = Path("config.local.json")
DEFAULTS = {
    # Sin límite, DuckDB toma ~80% de la RAM y puede congelar máquinas de 16 GB
    "duckdb": {"memory_limit": "4GB", "threads": 4},
    "torch": {"device": "auto", "amp": True, "threads": None},
    "gru": {"seeds": 5, "max_epochs": 40, "patience": 3, "val_frac": 0.15, "batch": 512, "pred_batch": 4096,
            "pre_epochs": 5},
    "env": {},
}

_local = json.loads(LOCAL.read_text()) if LOCAL.exists() else {}
CFG = {k: {**v, **_local.get(k, {})} for k, v in DEFAULTS.items()}
for k, v in CFG["env"].items():
    os.environ.setdefault(k, str(v))  # lo que ya venga del shell tiene prioridad


def device():
    import torch
    want = CFG["torch"]["device"]
    if want == "auto":
        want = "cuda" if torch.cuda.is_available() else "cpu"
    if want == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config.local.json pide GPU pero torch no la ve (¿rueda CPU de PyTorch?)")
    return torch.device(want)
