"""LightGBM y split por vehículo, sin dependencias pesadas (no importa PyTorch ni scikit-survival).

Separado de src/models.py para poder correr experimentos solo con el LightGBM (p. ej. src/geo_compare.py) en máquinas
donde PyTorch no está disponible. src/models.py importa todo desde acá, así que el pipeline completo no cambia.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from src.features import ALT

SEED, K = 0, 5
MOD = Path("models")
GBM_PARAMS = dict(n_estimators=500, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                  subsample_freq=1, colsample_bytree=0.5, reg_lambda=1.0, verbose=-1, random_state=SEED)
if (MOD / "gbm_params.json").exists():  # hiperparámetros elegidos por `python -m src.models tune` (solo con folds de train)
    GBM_PARAMS.update(json.load(open(MOD / "gbm_params.json"))["best"])


def split(f):
    veh = f.groupby("v")["failed"].first()
    rng = np.random.default_rng(SEED)
    test = set()
    for lab in (0, 1):
        vs = veh[veh == lab].index.values
        test |= set(rng.choice(vs, int(round(len(vs) * 0.2)), replace=False))
    fold = pd.Series(-1, index=veh.index)
    tr = veh[~veh.index.isin(test)]
    for k, (_, te) in enumerate(StratifiedGroupKFold(K, shuffle=True, random_state=SEED).split(tr, tr, tr.index)):
        fold[tr.index[te]] = k
    return f["v"].map(fold).values


def geo_mono(cols):
    """Restricción monótona creciente para la altitud (si está entre las features); vacío si no."""
    cols = list(cols)
    return {"monotone_constraints": [1 if c == ALT else 0 for c in cols]} if ALT in cols else {}


def fit_gbm(X, y):
    return lgb.LGBMClassifier(**GBM_PARAMS, **geo_mono(X.columns)).fit(X, y)
