"""
Treinamento dos modelos do Atlas Renovável do Nordeste — granularidade MENSAL
==============================================================================
Reformula o problema de regressão espacial (train.py, 134 estações, 1 valor
anual por estação) como um problema de INTERPOLAÇÃO TEMPORAL: estimar o
potencial renovável esperado de uma estação existente em um mês específico,
a partir de LAT, LON, UF e MÊS.

Motivação (ver src/aggregate_monthly.py)
-----------------------------------------
Colapsar 2022-2025 em uma única média por estação descarta o ciclo sazonal
solar/eólico — que é o sinal dominante e fisicamente bem estabelecido (ângulo
de incidência solar, regime de alísios) presente nos próprios dados horários
do INMET. Ao manter a granularidade (estação × mês), o modelo aprende esse
ciclo real sem usar nenhuma fonte de dados externa.

Validação
---------
Com N ≈ 1564 (vs. 134 no problema anual), o R² por fold do K-Fold (k=10,
~156 amostras/fold) já é estável — diferente do caso anual, onde o LOO era
necessário. Por isso aqui:
  - Holdout 80/20 (seed=42)   → linha de base
  - K-Fold (k=10)             → critério de seleção do melhor modelo
  - GroupKFold (k=10, por COD_WMO) → checagem de honestidade: nenhum mês da
    estação de teste aparece no treino. Mede a capacidade de generalização
    espacial pura (extrapolar para uma estação nunca vista), que é uma
    tarefa mais difícil que a interpolação temporal e por isso apresenta
    R² mais baixo — reportado para transparência, não usado para seleção.

Uso:
    python src/train_mensal.py
    python src/train_mensal.py --alvos IP_NE
    python src/train_mensal.py --rapido
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.model_selection import (
    GridSearchCV,
    GroupKFold,
    KFold,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.preprocessing import LabelEncoder

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from train import construir_modelos, montar_pipeline, SEED, KFOLD_K, CV_SEARCH, UF_ENCODER

warnings.filterwarnings("ignore")

ROOT          = Path(__file__).resolve().parents[1]
MONTHLY_CSV   = ROOT / "data" / "processed" / "inmet_medias_mensais.csv"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
MLRUNS_DIR    = ROOT / "mlruns"

# LAT/LON/UF como no problema anual + MES (sazonalidade, real e mensurável a
# partir do próprio timestamp do INMET — não é dado externo).
FEATURES = ["LAT", "LON", "UF_ENC", "MES"]

TARGETS = {
    "SOLAR_IRRAD": "SOLAR_IRRAD_kwh_m2_dia",
    "WIND_SPEED":  "WIND_SPEED_ms",
    "IP_NE":       "IP_NE",
}

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("train_mensal")


def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(MONTHLY_CSV)
    df["UF_ENC"] = UF_ENCODER.transform(df["UF"])

    # IP-NE mensal: normalizado pelo min/max do conjunto estação×mês (não pelo
    # min/max anual) — consistente com a granularidade do problema.
    solar = df["SOLAR_IRRAD_kwh_m2_dia"]
    wind  = df["WIND_SPEED_ms"]
    df["SOLAR_NORM"] = (solar - solar.min()) / (solar.max() - solar.min())
    df["WIND_NORM"]  = (wind  - wind.min())  / (wind.max()  - wind.min())
    df["IP_NE"]      = np.sqrt(df["SOLAR_NORM"] ** 2 + df["WIND_NORM"] ** 2)

    log.info(f"Dataset mensal: {len(df)} linhas (estação×mês), "
              f"{df['COD_WMO'].nunique()} estações; features = {FEATURES}.")
    return df


def avaliar(pipe_best, X, y, groups) -> dict:
    # 1. Holdout 80/20 -----------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
    m = clone(pipe_best).fit(X_tr, y_tr)
    pred_te = m.predict(X_te)
    holdout = {
        "holdout_r2":   r2_score(y_te, pred_te),
        "holdout_mae":  mean_absolute_error(y_te, pred_te),
        "holdout_rmse": root_mean_squared_error(y_te, pred_te),
    }

    # 2. K-Fold (k=10) — critério de seleção --------------------------------
    kf = KFold(n_splits=KFOLD_K, shuffle=True, random_state=SEED)
    cv = cross_validate(clone(pipe_best), X, y, cv=kf,
                         scoring=["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"],
                         n_jobs=-1)
    kfold_pred = cross_val_predict(clone(pipe_best), X, y, cv=kf, n_jobs=-1)
    kfold = {
        "kfold_r2_mean":   cv["test_r2"].mean(),
        "kfold_r2_std":    cv["test_r2"].std(),
        "kfold_mae_mean":  -cv["test_neg_mean_absolute_error"].mean(),
        "kfold_rmse_mean": -cv["test_neg_root_mean_squared_error"].mean(),
    }

    # 3. GroupKFold por estação — checagem de honestidade --------------------
    # Nenhum mês da estação de teste aparece no treino: mede generalização
    # espacial pura (não usado para seleção do melhor modelo).
    gkf = GroupKFold(n_splits=KFOLD_K)
    group_pred = cross_val_predict(clone(pipe_best), X, y, cv=gkf, groups=groups, n_jobs=-1)
    group_check = {"group_r2": r2_score(y, group_pred)}

    return {**holdout, **kfold, **group_check, "_kfold_pred": kfold_pred}


def treinar_alvo(alvo: str, df: pd.DataFrame, modelos: dict) -> tuple[list[dict], dict]:
    coluna = TARGETS[alvo]
    X = df[FEATURES].to_numpy()
    y = df[coluna].to_numpy()
    groups = df["COD_WMO"].to_numpy()

    mlflow.set_experiment(f"Atlas-Renovavel-Mensal-{alvo}")
    log.info(f"\n{'='*60}\n ALVO MENSAL: {alvo}  ({coluna})\n{'='*60}")

    resultados, kfold_preds = [], {}

    for nome, spec in modelos.items():
        pipe = montar_pipeline(spec["estimator"])

        with mlflow.start_run(run_name=nome):
            if spec["search"] == "grid":
                busca = GridSearchCV(pipe, spec["grid"], cv=CV_SEARCH, scoring="r2", n_jobs=-1)
            else:
                busca = RandomizedSearchCV(pipe, spec["grid"], n_iter=12, cv=CV_SEARCH,
                                            scoring="r2", n_jobs=-1, random_state=SEED)
            busca.fit(X, y)
            best = busca.best_estimator_

            met = avaliar(best, X, y, groups)
            kfold_preds[nome] = met.pop("_kfold_pred")

            best_params = {k.replace("model__", ""): v for k, v in busca.best_params_.items()}

            mlflow.log_param("modelo", nome)
            mlflow.log_param("alvo", alvo)
            mlflow.log_param("granularidade", "mensal")
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: float(v) for k, v in met.items()})

            signature = infer_signature(X, best.predict(X))
            try:
                mlflow.sklearn.log_model(best, name="model", signature=signature, input_example=X[:5])
            except TypeError:
                mlflow.sklearn.log_model(best, artifact_path="model", signature=signature, input_example=X[:5])

            log.info(
                f"  {nome:13s} | Holdout R²={met['holdout_r2']:.3f} | "
                f"KFold R²={met['kfold_r2_mean']:.3f}±{met['kfold_r2_std']:.3f} | "
                f"GroupKFold(estação) R²={met['group_r2']:.3f}"
            )

            resultados.append({
                "alvo": alvo, "modelo": nome, "busca": spec["search"],
                "best_params": json.dumps(best_params, ensure_ascii=False),
                **{k: float(v) for k, v in met.items()},
                "_estimator": best,
            })

    return resultados, kfold_preds


def figura_comparacao(df_res: pd.DataFrame, alvos: list[str]) -> None:
    fig, axes = plt.subplots(1, len(alvos), figsize=(5.2 * len(alvos), 4.5), squeeze=False)
    axes = axes[0]
    cores = plt.cm.Set2(np.linspace(0, 1, 8))

    for ax, alvo in zip(axes, alvos):
        sub = df_res[df_res["alvo"] == alvo].sort_values("kfold_r2_mean", ascending=False)
        ax.bar(sub["modelo"], sub["kfold_r2_mean"], color=cores[: len(sub)], edgecolor="white")
        for i, (_, r) in enumerate(sub.iterrows()):
            acima = r["kfold_r2_mean"] >= 0
            ax.text(i, r["kfold_r2_mean"] + (0.01 if acima else -0.01), f"{r['kfold_r2_mean']:.2f}",
                    ha="center", va="bottom" if acima else "top", fontsize=9, fontweight="bold")
        ax.set_title(f"{alvo}")
        ax.set_ylabel("R² (K-Fold, k=10)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Comparação de Modelos — Granularidade Mensal — R² por Alvo (K-Fold)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig09_comparacao_modelos_mensal.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def figura_pred_vs_real(df: pd.DataFrame, kfold_por_alvo: dict, melhores: dict, alvos: list[str]) -> None:
    fig, axes = plt.subplots(1, len(alvos), figsize=(5.0 * len(alvos), 4.7), squeeze=False)
    axes = axes[0]

    for ax, alvo in zip(axes, alvos):
        nome = melhores[alvo]["modelo"]
        y_real = df[TARGETS[alvo]].to_numpy()
        y_pred = kfold_por_alvo[alvo][nome]
        r2 = r2_score(y_real, y_pred)

        ax.scatter(y_real, y_pred, alpha=0.4, s=20, edgecolors="white", color="#2b8cbe")
        lims = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7, label="ideal (y = x)")
        ax.set_title(f"{alvo} — {nome}\n(K-Fold R² = {r2:.3f})")
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Valor previsto")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Previsto vs. Real (K-Fold) — Granularidade Mensal — Melhor Modelo por Alvo",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig10_previsto_vs_real_mensal.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def executar(alvos: list[str], rapido: bool = False) -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    df = carregar_dados()
    modelos = construir_modelos(rapido=rapido)

    todos_resultados = []
    kfold_por_alvo = {}

    for alvo in alvos:
        res, kfold_preds = treinar_alvo(alvo, df, modelos)
        todos_resultados.extend(res)
        kfold_por_alvo[alvo] = kfold_preds

    df_res = pd.DataFrame(todos_resultados)

    melhores = {}
    for alvo in alvos:
        sub = df_res[df_res["alvo"] == alvo]
        linha = sub.loc[sub["kfold_r2_mean"].idxmax()]
        melhores[alvo] = {"modelo": linha["modelo"],
                          "kfold_r2_mean": float(linha["kfold_r2_mean"]),
                          "group_r2": float(linha["group_r2"])}
        joblib.dump(linha["_estimator"], MODELS_DIR / f"best_{alvo}_mensal.joblib")
        log.info(f"Melhor modelo mensal para {alvo}: {linha['modelo']} "
                 f"(R² KFold = {linha['kfold_r2_mean']:.3f} | R² GroupKFold-estação = {linha['group_r2']:.3f}) "
                 f"-> models/best_{alvo}_mensal.joblib")

    df_tabela = df_res.drop(columns=["_estimator"]).round(4)
    df_tabela.to_csv(REPORTS_DIR / "model_comparison_mensal.csv", index=False, encoding="utf-8-sig")

    meta = {
        "features_ordem": FEATURES,
        "alvos": TARGETS,
        "melhores_modelos": melhores,
        "n_linhas": int(len(df)),
        "n_estacoes": int(df["COD_WMO"].nunique()),
        "uf_classes": list(UF_ENCODER.classes_),
        "solar_min": float(df["SOLAR_IRRAD_kwh_m2_dia"].min()),
        "solar_max": float(df["SOLAR_IRRAD_kwh_m2_dia"].max()),
        "wind_min":  float(df["WIND_SPEED_ms"].min()),
        "wind_max":  float(df["WIND_SPEED_ms"].max()),
        "features": {
            f: {"min": float(df[f].min()), "max": float(df[f].max()), "mean": float(df[f].mean())}
            for f in FEATURES
        },
    }
    (MODELS_DIR / "metadata_mensal.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    figura_comparacao(df_tabela, alvos)
    figura_pred_vs_real(df, kfold_por_alvo, melhores, alvos)

    log.info(f"\n{'='*60}\n RESUMO — melhores modelos por alvo (mensal)\n{'='*60}")
    for alvo, info in melhores.items():
        log.info(f"  {alvo:13s}: {info['modelo']:13s} "
                 f"(R² KFold = {info['kfold_r2_mean']:.3f} | R² GroupKFold-estação = {info['group_r2']:.3f})")
    log.info("\nArtefatos:")
    log.info("  reports/model_comparison_mensal.csv")
    log.info("  reports/fig09_comparacao_modelos_mensal.png")
    log.info("  reports/fig10_previsto_vs_real_mensal.png")
    log.info("  models/best_*_mensal.joblib + models/metadata_mensal.json")
    log.info("  mlruns/  (abra com: mlflow ui)")

    return df_tabela


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino mensal — Atlas Renovável do Nordeste")
    parser.add_argument("--alvos", nargs="+", choices=list(TARGETS), default=list(TARGETS))
    parser.add_argument("--rapido", action="store_true")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" TREINO MENSAL — ATLAS RENOVÁVEL DO NORDESTE")
    print(f" Alvos: {args.alvos} | Modo rápido: {args.rapido}")
    print("=" * 60 + "\n")

    df_final = executar(alvos=args.alvos, rapido=args.rapido)

    print("\n✓ Concluído. Tabela comparativa:\n")
    cols = ["alvo", "modelo", "holdout_r2", "kfold_r2_mean", "kfold_r2_std", "group_r2"]
    print(df_final[cols].to_string(index=False))
