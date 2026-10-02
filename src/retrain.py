"""Reentreno mensual de los LightGBM con todos los vehículos -> models/prod/<fecha>/ (README, "Reentreno en producción")."""
import json
import subprocess
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from src.evaluate import cluster_ci
from src.features import HORIZONS, feature_cols
from src.models import K, MOD, fit_gbm
from src.temporal import known_at

PROD = MOD / "prod"


def main(t=None):
    f = pd.read_parquet("data/features.parquet")
    f["day"] = pd.to_datetime(f["day"])
    t = pd.Timestamp(t) if t else f["day"].max() + pd.Timedelta(days=1)
    cols = feature_cols(f)
    cur = PROD / "current"
    prev = json.load(open(cur / "meta.json")) if cur.exists() else None
    out = PROD / str(t.date())
    out.mkdir(parents=True, exist_ok=True)
    meta = {"t": str(t.date()), "prev": prev and prev["t"], "git": subprocess.run(
        ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip(), "H": {}}
    ok = True
    for h in HORIZONS:
        y = f[f"y{h}"].values
        rows = known_at(f, h, t).values
        r = {"filas": int(rows.sum()), "vehiculos_con_evento": int(f.loc[rows & (y == 1), "v"].nunique())}
        # en vivo: el vigente sobre las etiquetas conocidas desde su reentreno
        if prev:
            new = rows & ~known_at(f, h, pd.Timestamp(prev["t"])).values
            if 0 < y[new].sum() < new.sum():
                p = lgb.Booster(model_file=str(cur / f"gbm{h}.txt")).predict(f.loc[new, cols])
                r["vigente_auc_en_vivo"] = [roc_auc_score(y[new], p), cluster_ci(y[new], p, f["v"].values[new], n=200)[0]]
        # control del candidato: AUC con folds por vehículo sobre lo conocido en t
        idx = np.where(rows)[0]
        oof = np.zeros(len(idx))
        for tr, va in GroupKFold(K).split(idx, groups=f["v"].values[idx]):
            oof[va] = fit_gbm(f.iloc[idx[tr]][cols], y[idx[tr]]).predict_proba(f.iloc[idx[va]][cols])[:, 1]
        r["cv_auc"] = [roc_auc_score(y[idx], oof), cluster_ci(y[idx], oof, f["v"].values[idx], n=200)[0]]
        # se rechaza solo si el candidato es claramente peor que el vigente (IC entero por debajo)
        if prev and r["cv_auc"][1][1] < prev["H"][str(h)]["cv_auc"][0]:
            ok = False
        fit_gbm(f.iloc[idx][cols], y[idx]).booster_.save_model(str(out / f"gbm{h}.txt"))
        meta["H"][str(h)] = r
        print(f"H{h}: {json.dumps(r, default=float)}", flush=True)
    meta["promovido"] = ok
    json.dump(meta, open(out / "meta.json", "w"), indent=2, default=float)
    if ok:
        cur.unlink(missing_ok=True)
        cur.symlink_to(out.name)
    print(f"{out}: {'promovido' if ok else 'RECHAZADO, sigue ' + str(prev['t'])}")
    return ok


if __name__ == "__main__":
    sys.exit(0 if main(next(iter(sys.argv[1:]), None)) else 1)
