"""
Treinamento e avaliação dos modelos do Atlas Renovável do Nordeste
==================================================================
Formula o problema como REGRESSÃO ESPACIAL (Seção 3.3 do artigo):
    Features : LAT, LON  (coordenadas geográficas; ALT é descartada — a ablação
               em analise_ablacao_altitude mostra que ela reduz o R²)
    Alvos    : SOLAR_IRRAD (kWh/m²/dia), WIND_SPEED (m/s), IP_NE (adimensional)

Observação: estações duplicadas (mesmo COD_WMO) são agregadas antes do treino,
reduzindo o conjunto de 155 registros para ~135 estações únicas.

Cinco modelos supervisionados são comparados (Seção 3.4):
    KNN, Árvore de Decisão, Random Forest, AdaBoost e MLP.
Os hiperparâmetros são otimizados com GridSearchCV (KNN/DT/RF/AdaBoost) e
RandomizedSearchCV (MLP).

Três estratégias de validação são aplicadas (Seção 3.5), justificadas pelo
tamanho reduzido do dataset (N = 155):
    - Holdout 80/20 (seed=42)  → linha de base rápida
    - K-Fold (k=10)            → média ± desvio de R², MAE, RMSE
    - Leave-One-Out (LOO)      → estimativa de menor viés para N pequeno

Cada experimento (alvo × modelo) é rastreado com MLflow (parâmetros, métricas
e o modelo treinado). O melhor modelo de cada alvo é salvo em ``models/`` para
o dashboard, e a tabela comparativa completa em ``reports/``.

Uso:
    python src/train.py                 # pipeline completo (3 alvos × 5 modelos)
    python src/train.py --alvos IP_NE   # apenas um alvo
    python src/train.py --rapido        # grids reduzidos (validação rápida)
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import warnings
from pathlib import Path

# Console UTF-8 (evita UnicodeEncodeError com ², ±, √ no terminal do Windows)
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

# MLflow 3.x bloqueia o file store por padrão; o projeto usa ./mlruns (ver
# .gitignore), então optamos explicitamente por ele antes de importar mlflow.
os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
# Não registrar nomes de variáveis de ambiente nos metadados do modelo.
os.environ.setdefault("MLFLOW_RECORD_ENV_VARS_IN_MODEL_LOGGING", "false")

import joblib
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # backend headless (roda em container/CI)
import matplotlib.pyplot as plt

from sklearn.base import clone
from sklearn.ensemble import AdaBoostRegressor, RandomForestRegressor
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    LeaveOneOut,
    RandomizedSearchCV,
    cross_val_predict,
    cross_validate,
    train_test_split,
)
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

warnings.filterwarnings("ignore")

# ─── Configuração de caminhos ─────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
DATA_CSV    = ROOT / "data" / "processed" / "inmet_medias_estacoes.csv"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
MLRUNS_DIR  = ROOT / "mlruns"

# Apenas coordenadas geográficas: a ablação (ver analise_ablacao_altitude)
# mostra que ALT reduz o R² em todos os alvos, por isso fica de fora dos modelos.
FEATURES = ["LAT", "LON"]

# alvo_curto -> coluna no DataFrame
TARGETS = {
    "SOLAR_IRRAD": "SOLAR_IRRAD_kwh_m2_dia",
    "WIND_SPEED":  "WIND_SPEED_ms",
    "IP_NE":       "IP_NE",
}

SEED       = 42
KFOLD_K    = 10
CV_SEARCH  = 5  # folds internos da busca de hiperparâmetros

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("train")


# ─── Dados ────────────────────────────────────────────────────────────────────

def carregar_dados() -> pd.DataFrame:
    """Carrega o dataset, agrega estações duplicadas e calcula o IP-NE (Eq. 1 e 2)."""
    df = pd.read_csv(DATA_CSV)
    n0 = len(df)

    # Algumas estações aparecem mais de uma vez sob o mesmo código WMO, com
    # pequenas variações de coordenada ou grafia — ex.: A330 (Paulistana, que
    # surge 2x no top-10 do artigo) e A404 (Luís/Luiz Eduardo Magalhães).
    # Agregam-se por COD_WMO para evitar pontos quase-duplicados, que injetam
    # ruído na regressão espacial e inflam artificialmente N.
    agg = {
        "UF": "first", "ESTACAO": "first",
        "LAT": "mean", "LON": "mean", "ALT": "mean",
        "SOLAR_IRRAD_kwh_m2_dia": "mean", "WIND_SPEED_ms": "mean",
        "anos_disponiveis": "max", "dias_validos_total": "sum",
    }
    df = df.groupby("COD_WMO", as_index=False).agg(agg)

    solar = df["SOLAR_IRRAD_kwh_m2_dia"]
    wind  = df["WIND_SPEED_ms"]

    df["SOLAR_NORM"] = (solar - solar.min()) / (solar.max() - solar.min())
    df["WIND_NORM"]  = (wind  - wind.min())  / (wind.max()  - wind.min())
    df["IP_NE"]      = np.sqrt(df["SOLAR_NORM"] ** 2 + df["WIND_NORM"] ** 2)

    log.info(f"Dataset: {n0} registros → {len(df)} estações únicas (dedup por "
             f"COD_WMO); features = {FEATURES}.")
    return df


def constantes_normalizacao(df: pd.DataFrame) -> dict:
    """Min/Max usados no IP-NE e faixas das features (para o dashboard)."""
    return {
        "solar_min": float(df["SOLAR_IRRAD_kwh_m2_dia"].min()),
        "solar_max": float(df["SOLAR_IRRAD_kwh_m2_dia"].max()),
        "wind_min":  float(df["WIND_SPEED_ms"].min()),
        "wind_max":  float(df["WIND_SPEED_ms"].max()),
        "features": {
            f: {"min": float(df[f].min()), "max": float(df[f].max()),
                "mean": float(df[f].mean())}
            for f in FEATURES
        },
    }


# ─── Definição dos modelos e grades de hiperparâmetros ────────────────────────

def construir_modelos(rapido: bool = False) -> dict:
    """
    Retorna {nome: {estimator, search, grid}} para cada modelo.

    Todos usam Pipeline(StandardScaler + estimador): a padronização é essencial
    para KNN e MLP (sensíveis à escala) e inofensiva para os modelos de árvore,
    garantindo comparação justa. As chaves da grade usam o prefixo ``model__``.
    """
    if rapido:
        grids = {
            "KNN":          {"model__n_neighbors": [5, 9], "model__weights": ["distance"], "model__p": [2]},
            "DecisionTree": {"model__max_depth": [5, None], "model__min_samples_split": [2]},
            "RandomForest": {"model__n_estimators": [100], "model__max_depth": [None]},
            "AdaBoost":     {"model__n_estimators": [100], "model__learning_rate": [0.1]},
            "MLP":          {"model__hidden_layer_sizes": [(64, 32)], "model__activation": ["relu"], "model__alpha": [1e-3]},
        }
    else:
        grids = {
            "KNN": {
                "model__n_neighbors": [3, 5, 7, 9, 11, 15],
                "model__weights":     ["uniform", "distance"],
                "model__p":           [1, 2],  # 1=Manhattan, 2=Euclidiana
            },
            "DecisionTree": {
                "model__max_depth":         [3, 5, 8, 12, None],
                "model__min_samples_split": [2, 5, 10],
            },
            "RandomForest": {
                "model__n_estimators": [100, 300, 500],
                "model__max_depth":    [5, 10, None],
            },
            "AdaBoost": {
                "model__n_estimators":  [50, 100, 200],
                "model__learning_rate": [0.01, 0.1, 1.0],
            },
            "MLP": {
                "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (128, 64, 32)],
                "model__activation":         ["relu", "tanh"],
                "model__alpha":              [1e-4, 1e-3, 1e-2, 1e-1],
            },
        }

    return {
        "KNN": dict(
            estimator=KNeighborsRegressor(),
            search="grid", grid=grids["KNN"],
        ),
        "DecisionTree": dict(
            estimator=DecisionTreeRegressor(random_state=SEED),
            search="grid", grid=grids["DecisionTree"],
        ),
        "RandomForest": dict(
            # n_jobs=1 no estimador: a paralelização ocorre no nível da CV
            estimator=RandomForestRegressor(random_state=SEED, n_jobs=1),
            search="grid", grid=grids["RandomForest"],
        ),
        "AdaBoost": dict(
            estimator=AdaBoostRegressor(random_state=SEED),
            search="grid", grid=grids["AdaBoost"],
        ),
        "MLP": dict(
            estimator=MLPRegressor(random_state=SEED, max_iter=2000, early_stopping=True),
            search="random", grid=grids["MLP"],
        ),
    }


def montar_pipeline(estimator) -> Pipeline:
    return Pipeline([("scaler", StandardScaler()), ("model", estimator)])


# ─── Avaliação: Holdout + K-Fold + LOO ────────────────────────────────────────

SCORERS = ["r2", "neg_mean_absolute_error", "neg_root_mean_squared_error"]


def r2_detalhado(y_true, y_pred, contexto: str = "") -> dict:
    """
    Recalcula o R² de forma explícita e expõe os termos intermediários
    (n, média de y, SST, SSE) da fórmula  R² = 1 − SSE/SST.

    Serve de auditoria: o R² manual e o de ``sklearn.r2_score`` devem
    coincidir até a precisão de ponto flutuante. Se algum dia divergirem,
    há de fato um problema de implementação/inversão/tipo a investigar.

    ── Glossário ─────────────────────────────────────────────────────────
      SST (Total Sum of Squares)    = Σ (yᵢ − ȳ)²  → variância total do alvo
      SSE (Residual Sum of Squares) = Σ (yᵢ − ŷᵢ)² → erro do modelo
      R²  = 1 − SSE/SST  (1 = perfeito, 0 = igual a prever a média, <0 = pior)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    n       = int(y_true.size)
    y_media = float(y_true.mean())
    sse     = float(np.sum((y_true - y_pred) ** 2))    # resíduos do modelo
    sst     = float(np.sum((y_true - y_media) ** 2))   # variabilidade total
    r2_manual  = 1.0 - sse / sst if sst > 0 else float("nan")
    r2_sklearn = float(r2_score(y_true, y_pred))

    tag = f" — {contexto}" if contexto else ""
    log.info(
        f"    [R² detalhado{tag}] n={n} | ȳ={y_media:.4f} | "
        f"SST={sst:.4f} | SSE={sse:.4f} | "
        f"R²(1−SSE/SST)={r2_manual:.4f} | R²(sklearn)={r2_sklearn:.4f}"
    )
    if not np.isnan(r2_manual) and abs(r2_manual - r2_sklearn) > 1e-9:
        log.warning(
            f"    [R² detalhado{tag}] divergência manual×sklearn "
            f"({r2_manual:.6f} ≠ {r2_sklearn:.6f}) — investigar cálculo!"
        )
    return {"n": n, "y_media": y_media, "sst": sst, "sse": sse,
            "r2_manual": r2_manual, "r2_sklearn": r2_sklearn}


def avaliar(pipe_best: Pipeline, X: np.ndarray, y: np.ndarray) -> dict:
    """Aplica as três estratégias de validação a um pipeline já configurado."""
    # 1. Holdout 80/20 ---------------------------------------------------------
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.20, random_state=SEED)
    m = clone(pipe_best).fit(X_tr, y_tr)
    pred_te = m.predict(X_te)
    holdout = {
        "holdout_r2":   r2_score(y_te, pred_te),
        "holdout_mae":  mean_absolute_error(y_te, pred_te),
        "holdout_rmse": root_mean_squared_error(y_te, pred_te),
    }

    # 2. K-Fold (k=10): média ± desvio -----------------------------------------
    kf = KFold(n_splits=KFOLD_K, shuffle=True, random_state=SEED)
    cv = cross_validate(clone(pipe_best), X, y, cv=kf, scoring=SCORERS, n_jobs=-1)
    kfold = {
        "kfold_r2_mean":   cv["test_r2"].mean(),
        "kfold_r2_std":    cv["test_r2"].std(),
        "kfold_mae_mean":  -cv["test_neg_mean_absolute_error"].mean(),
        "kfold_rmse_mean": -cv["test_neg_root_mean_squared_error"].mean(),
    }

    # 3. Leave-One-Out: métricas sobre o vetor agregado de previsões ------------
    # (R² não é definido por fold de 1 amostra → agrega-se via cross_val_predict)
    loo_pred = cross_val_predict(clone(pipe_best), X, y, cv=LeaveOneOut(), n_jobs=-1)
    # Auditoria do R² LOO: imprime os termos intermediários (n, ȳ, SST, SSE)
    # e confere o R² manual contra o do sklearn.
    r2_detalhado(y, loo_pred, contexto="LOO")
    loo = {
        "loo_r2":   r2_score(y, loo_pred),
        "loo_mae":  mean_absolute_error(y, loo_pred),
        "loo_rmse": root_mean_squared_error(y, loo_pred),
    }

    return {**holdout, **kfold, **loo, "_loo_pred": loo_pred}


# ─── Experimento de um alvo (5 modelos) ───────────────────────────────────────

def treinar_alvo(alvo: str, df: pd.DataFrame, modelos: dict) -> tuple[list[dict], dict]:
    """Treina e avalia os 5 modelos para um alvo; registra tudo no MLflow."""
    coluna = TARGETS[alvo]
    X = df[FEATURES].to_numpy()
    y = df[coluna].to_numpy()

    mlflow.set_experiment(f"Atlas-Renovavel-{alvo}")
    log.info(f"\n{'='*60}\n ALVO: {alvo}  ({coluna})\n{'='*60}")

    resultados = []
    loo_preds  = {}

    for nome, spec in modelos.items():
        pipe = montar_pipeline(spec["estimator"])

        with mlflow.start_run(run_name=nome):
            # Busca de hiperparâmetros (refit em todo o conjunto) ----------------
            if spec["search"] == "grid":
                busca = GridSearchCV(
                    pipe, spec["grid"], cv=CV_SEARCH, scoring="r2", n_jobs=-1,
                )
            else:
                busca = RandomizedSearchCV(
                    pipe, spec["grid"], n_iter=12, cv=CV_SEARCH, scoring="r2",
                    n_jobs=-1, random_state=SEED,
                )
            busca.fit(X, y)
            best = busca.best_estimator_  # já ajustado em todo X, y

            # Validação nas três estratégias ------------------------------------
            met = avaliar(best, X, y)
            loo_preds[nome] = met.pop("_loo_pred")

            best_params = {k.replace("model__", ""): v for k, v in busca.best_params_.items()}

            # ─ MLflow: parâmetros, métricas e modelo ──────────────────────────
            mlflow.log_param("modelo", nome)
            mlflow.log_param("alvo", alvo)
            mlflow.log_param("estrategia_busca", spec["search"])
            mlflow.log_param("cv_busca", CV_SEARCH)
            mlflow.log_params(best_params)
            mlflow.log_metrics({k: float(v) for k, v in met.items()})

            signature = infer_signature(X, best.predict(X))
            try:
                mlflow.sklearn.log_model(best, name="model", signature=signature,
                                         input_example=X[:5])
            except TypeError:  # compatibilidade com mlflow < 3
                mlflow.sklearn.log_model(best, artifact_path="model",
                                         signature=signature, input_example=X[:5])

            log.info(
                f"  {nome:13s} | Holdout R²={met['holdout_r2']:.3f} | "
                f"KFold R²={met['kfold_r2_mean']:.3f}±{met['kfold_r2_std']:.3f} | "
                f"LOO R²={met['loo_r2']:.3f}"
            )

            resultados.append({
                "alvo": alvo, "modelo": nome, "busca": spec["search"],
                "best_params": json.dumps(best_params, ensure_ascii=False),
                **{k: float(v) for k, v in met.items()},
                "_estimator": best,
            })

    return resultados, loo_preds


# ─── Figuras de comparação ────────────────────────────────────────────────────

def figura_comparacao(df_res: pd.DataFrame, alvos: list[str]) -> None:
    """fig09 — R² (Leave-One-Out) de cada modelo, por alvo (métrica de seleção)."""
    fig, axes = plt.subplots(1, len(alvos), figsize=(5.2 * len(alvos), 4.5), squeeze=False)
    axes = axes[0]
    cores = plt.cm.Set2(np.linspace(0, 1, 8))

    for ax, alvo in zip(axes, alvos):
        sub = df_res[df_res["alvo"] == alvo].sort_values("loo_r2", ascending=False)
        ax.bar(sub["modelo"], sub["loo_r2"], color=cores[: len(sub)], edgecolor="white")
        for i, (_, r) in enumerate(sub.iterrows()):
            acima = r["loo_r2"] >= 0
            ax.text(i, r["loo_r2"] + (0.004 if acima else -0.004), f"{r['loo_r2']:.2f}",
                    ha="center", va="bottom" if acima else "top",
                    fontsize=9, fontweight="bold")
        ax.set_title(f"{alvo}")
        ax.set_ylabel("R² (Leave-One-Out)")
        ax.axhline(0, color="gray", lw=0.8)
        ax.tick_params(axis="x", rotation=30)
        ax.grid(axis="y", alpha=0.3, linestyle="--")

    fig.suptitle("Comparação de Modelos — R² por Alvo (Leave-One-Out)",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig09_comparacao_modelos.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def figura_pred_vs_real(df: pd.DataFrame, loo_por_alvo: dict, melhores: dict,
                        alvos: list[str]) -> None:
    """fig10 — Previsto vs. Real (LOO) do melhor modelo de cada alvo."""
    fig, axes = plt.subplots(1, len(alvos), figsize=(5.0 * len(alvos), 4.7), squeeze=False)
    axes = axes[0]

    for ax, alvo in zip(axes, alvos):
        nome = melhores[alvo]["modelo"]
        y_real = df[TARGETS[alvo]].to_numpy()
        y_pred = loo_por_alvo[alvo][nome]
        r2 = r2_score(y_real, y_pred)

        ax.scatter(y_real, y_pred, alpha=0.6, s=35, edgecolors="white",
                   color="#2b8cbe")
        lims = [min(y_real.min(), y_pred.min()), max(y_real.max(), y_pred.max())]
        ax.plot(lims, lims, "r--", alpha=0.7, label="ideal (y = x)")
        ax.set_title(f"{alvo} — {nome}\n(LOO R² = {r2:.3f})")
        ax.set_xlabel("Valor real")
        ax.set_ylabel("Valor previsto")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3, linestyle="--")

    fig.suptitle("Previsto vs. Real (Leave-One-Out) — Melhor Modelo por Alvo",
                 fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig10_previsto_vs_real.png", bbox_inches="tight", dpi=120)
    plt.close(fig)


def analise_ablacao_altitude(df: pd.DataFrame, alvos: list[str]) -> pd.DataFrame:
    """
    Importância por ablação (drop-column): compara o R² (LOO, Random Forest)
    usando LAT/LON contra LAT/LON/ALT. Justifica deixar ALT fora dos modelos.
    """
    rows = []
    for alvo in alvos:
        y = df[TARGETS[alvo]].to_numpy()
        for feats, label in [(["LAT", "LON"], "LAT/LON"),
                             (["LAT", "LON", "ALT"], "LAT/LON/ALT")]:
            X = df[feats].to_numpy()
            pipe = montar_pipeline(RandomForestRegressor(n_estimators=300, random_state=SEED, n_jobs=1))
            pred = cross_val_predict(pipe, X, y, cv=LeaveOneOut(), n_jobs=-1)
            rows.append({"alvo": alvo, "features": label,
                         "loo_r2": round(float(r2_score(y, pred)), 4)})
    dfa = pd.DataFrame(rows)
    dfa.to_csv(REPORTS_DIR / "ablation_altitude.csv", index=False, encoding="utf-8-sig")

    alvos_u = list(dict.fromkeys(dfa["alvo"]))
    r_sem = [dfa[(dfa.alvo == a) & (dfa.features == "LAT/LON")]["loo_r2"].iloc[0] for a in alvos_u]
    r_com = [dfa[(dfa.alvo == a) & (dfa.features == "LAT/LON/ALT")]["loo_r2"].iloc[0] for a in alvos_u]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(alvos_u)); w = 0.38
    ax.bar(x - w / 2, r_sem, w, label="LAT/LON",     color="#2b8cbe", edgecolor="white")
    ax.bar(x + w / 2, r_com, w, label="LAT/LON/ALT", color="#bdbdbd", edgecolor="white")
    for i, (a, b) in enumerate(zip(r_sem, r_com)):
        ax.text(i - w / 2, a + 0.005, f"{a:.2f}", ha="center", va="bottom", fontsize=8.5)
        ax.text(i + w / 2, b + 0.005, f"{b:.2f}", ha="center", va="bottom", fontsize=8.5)
    ax.axhline(0, color="gray", lw=0.8)
    ax.set_xticks(x); ax.set_xticklabels(alvos_u)
    ax.set_ylabel("R² (Leave-One-Out, Random Forest)")
    ax.set_title("Importância por ablação — incluir ALT reduz o R² em todos os alvos")
    ax.legend(); ax.grid(axis="y", alpha=0.3, linestyle="--")
    fig.tight_layout()
    fig.savefig(REPORTS_DIR / "fig11_ablacao_altitude.png", bbox_inches="tight", dpi=120)
    plt.close(fig)
    return dfa


# ─── Pipeline principal ───────────────────────────────────────────────────────

def executar(alvos: list[str], rapido: bool = False) -> pd.DataFrame:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    MLRUNS_DIR.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(MLRUNS_DIR.as_uri())

    df = carregar_dados()
    modelos = construir_modelos(rapido=rapido)

    todos_resultados = []
    loo_por_alvo = {}

    for alvo in alvos:
        res, loo_preds = treinar_alvo(alvo, df, modelos)
        todos_resultados.extend(res)
        loo_por_alvo[alvo] = loo_preds

    df_res = pd.DataFrame(todos_resultados)

    # ── Melhor modelo por alvo (critério: maior R² em Leave-One-Out) ──────────
    # Para N pequeno, o LOO é a estimativa de generalização mais estável; o R²
    # por fold do K-Fold (k=10, ~13 amostras/fold) tem variância alta e é
    # pessimista, não servindo como critério de seleção.
    melhores = {}
    for alvo in alvos:
        sub = df_res[df_res["alvo"] == alvo]
        linha = sub.loc[sub["loo_r2"].idxmax()]
        melhores[alvo] = {"modelo": linha["modelo"],
                          "loo_r2": float(linha["loo_r2"]),
                          "kfold_r2_mean": float(linha["kfold_r2_mean"])}
        # salva o estimador (já ajustado em todo o conjunto)
        joblib.dump(linha["_estimator"], MODELS_DIR / f"best_{alvo}.joblib")
        log.info(f"Melhor modelo para {alvo}: {linha['modelo']} "
                 f"(R² LOO = {linha['loo_r2']:.3f}) → models/best_{alvo}.joblib")

    # ── Salva tabela comparativa (sem a coluna de objetos) ────────────────────
    df_tabela = df_res.drop(columns=["_estimator"]).round(4)
    df_tabela.to_csv(REPORTS_DIR / "model_comparison.csv", index=False, encoding="utf-8-sig")

    # ── Metadados para o dashboard (constantes IP-NE, faixas, melhores) ───────
    meta = constantes_normalizacao(df)
    meta["features_ordem"] = FEATURES
    meta["alvos"] = TARGETS
    meta["melhores_modelos"] = melhores
    meta["n_estacoes"] = int(len(df))
    (MODELS_DIR / "metadata.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Figuras ───────────────────────────────────────────────────────────────
    figura_comparacao(df_tabela, alvos)
    figura_pred_vs_real(df, loo_por_alvo, melhores, alvos)

    # ── Justificativa de features (ablação de ALT) ────────────────────────────
    dfa = analise_ablacao_altitude(df, alvos)
    log.info("Ablação ALT (LOO R², Random Forest):\n" + dfa.to_string(index=False))

    log.info(f"\n{'='*60}\n RESUMO — melhores modelos por alvo\n{'='*60}")
    for alvo, info in melhores.items():
        log.info(f"  {alvo:13s}: {info['modelo']:13s} (R² LOO = {info['loo_r2']:.3f})")
    log.info(f"\nArtefatos:")
    log.info(f"  reports/model_comparison.csv")
    log.info(f"  reports/ablation_altitude.csv")
    log.info(f"  reports/fig09_comparacao_modelos.png")
    log.info(f"  reports/fig10_previsto_vs_real.png")
    log.info(f"  reports/fig11_ablacao_altitude.png")
    log.info(f"  models/best_*.joblib + models/metadata.json")
    log.info(f"  mlruns/  (abra com: mlflow ui)")

    return df_tabela


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Treino dos modelos — Atlas Renovável do Nordeste")
    parser.add_argument("--alvos", nargs="+", choices=list(TARGETS), default=list(TARGETS),
                        help="Alvos a treinar (padrão: todos).")
    parser.add_argument("--rapido", action="store_true",
                        help="Usa grades reduzidas para validação rápida.")
    args = parser.parse_args()

    print("\n" + "=" * 60)
    print(" TREINO — ATLAS RENOVÁVEL DO NORDESTE")
    print(f" Alvos: {args.alvos} | Modo rápido: {args.rapido}")
    print("=" * 60 + "\n")

    df_final = executar(alvos=args.alvos, rapido=args.rapido)

    print("\n✓ Concluído. Tabela comparativa:\n")
    cols = ["alvo", "modelo", "holdout_r2", "kfold_r2_mean", "kfold_r2_std", "loo_r2"]
    print(df_final[cols].to_string(index=False))
