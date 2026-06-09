"""
Atlas Renovável do Nordeste — Dashboard Streamlit
=================================================
Interface interativa que reúne:
  • Visão geral do dataset e do índice IP-NE
  • Análise exploratória (8 figuras do notebook 01_eda.ipynb)
  • Desempenho comparativo dos modelos (tabela + figuras de src/train.py)
  • Previsão interativa: LAT/LON/ALT → irradiação solar, vento e IP-NE
  • Ranking e mapa do potencial renovável por estação

Execução:
    streamlit run app/app.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

# ─── Caminhos ─────────────────────────────────────────────────────────────────

ROOT        = Path(__file__).resolve().parents[1]
DATA_CSV    = ROOT / "data" / "processed" / "inmet_medias_estacoes.csv"
MODELS_DIR  = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIG_DIR     = ROOT / "notebooks"

FEATURES = ["LAT", "LON"]
TARGETS  = {
    "SOLAR_IRRAD": "SOLAR_IRRAD_kwh_m2_dia",
    "WIND_SPEED":  "WIND_SPEED_ms",
    "IP_NE":       "IP_NE",
}
ALVO_LABEL = {
    "SOLAR_IRRAD": "Irradiação Solar (kWh/m²/dia)",
    "WIND_SPEED":  "Velocidade do Vento (m/s)",
    "IP_NE":       "Índice IP-NE (adimensional)",
}
ESTADOS_NE = ["AL", "BA", "CE", "MA", "PB", "PE", "PI", "RN", "SE"]

st.set_page_config(
    page_title="Atlas Renovável do Nordeste",
    page_icon="🌞",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ─── Carregamento de dados e artefatos (com cache) ────────────────────────────

@st.cache_data(show_spinner=False)
def carregar_dados() -> pd.DataFrame:
    df = pd.read_csv(DATA_CSV)
    solar, wind = df["SOLAR_IRRAD_kwh_m2_dia"], df["WIND_SPEED_ms"]
    df["SOLAR_NORM"] = (solar - solar.min()) / (solar.max() - solar.min())
    df["WIND_NORM"]  = (wind  - wind.min())  / (wind.max()  - wind.min())
    df["IP_NE"]      = np.sqrt(df["SOLAR_NORM"] ** 2 + df["WIND_NORM"] ** 2)
    return df


@st.cache_data(show_spinner=False)
def carregar_metricas() -> pd.DataFrame | None:
    f = REPORTS_DIR / "model_comparison.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_data(show_spinner=False)
def carregar_metadata() -> dict | None:
    f = MODELS_DIR / "metadata.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_resource(show_spinner=False)
def carregar_modelos() -> dict:
    modelos = {}
    for alvo in TARGETS:
        f = MODELS_DIR / f"best_{alvo}.joblib"
        if f.exists():
            modelos[alvo] = joblib.load(f)
    return modelos


def computar_ip_ne(solar: float, wind: float, meta: dict | None, df: pd.DataFrame) -> float:
    """IP-NE a partir de solar/vento usando as constantes de normalização."""
    if meta:
        s_min, s_max = meta["solar_min"], meta["solar_max"]
        w_min, w_max = meta["wind_min"], meta["wind_max"]
    else:
        s_min, s_max = df["SOLAR_IRRAD_kwh_m2_dia"].min(), df["SOLAR_IRRAD_kwh_m2_dia"].max()
        w_min, w_max = df["WIND_SPEED_ms"].min(), df["WIND_SPEED_ms"].max()
    s_norm = (solar - s_min) / (s_max - s_min)
    w_norm = (wind - w_min) / (w_max - w_min)
    return float(np.sqrt(s_norm ** 2 + w_norm ** 2))


# ─── Carrega tudo ─────────────────────────────────────────────────────────────

df       = carregar_dados()
metricas = carregar_metricas()
metadata = carregar_metadata()
modelos  = carregar_modelos()

ARTEFATOS_OK = bool(modelos)


# ─── Barra lateral ────────────────────────────────────────────────────────────

st.sidebar.title("🌞🌬️ Atlas Renovável")
st.sidebar.caption("Potencial de geração renovável no Nordeste do Brasil")
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Análise Exploratória", "Desempenho dos Modelos",
     "Previsão Interativa", "Ranking & Mapa"],
)

if not ARTEFATOS_OK:
    st.sidebar.warning(
        "Modelos não encontrados em `models/`.\n\n"
        "Execute o treino primeiro:\n\n"
        "```\npython src/train.py\n```"
    )
else:
    st.sidebar.success("Modelos carregados ✔")

st.sidebar.divider()
st.sidebar.metric("Estações", f"{len(df)}")
st.sidebar.metric("Estados", f"{df['UF'].nunique()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "Visão Geral":
    st.title("Atlas Renovável do Nordeste")
    st.markdown(
        "Sistema de **aprendizado de máquina** para mapeamento e previsão do "
        "potencial de geração de energia **solar, eólica e híbrida** nos nove "
        "estados do Nordeste. O índice **IP-NE** estende a metodologia IP-PB "
        "(Ferreira et al., 2023) para escala regional, cobrindo **155 estações** "
        "automáticas do INMET (2022–2025)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Estações", f"{len(df)}")
    c2.metric("Solar médio", f"{df['SOLAR_IRRAD_kwh_m2_dia'].mean():.2f}", "kWh/m²/dia")
    c3.metric("Vento médio", f"{df['WIND_SPEED_ms'].mean():.2f}", "m/s")
    c4.metric("IP-NE mediano", f"{df['IP_NE'].median():.3f}")

    st.subheader("O índice IP-NE")
    st.latex(r"IP_{NE} = \sqrt{SOLAR_{norm}^2 + WIND_{norm}^2}, \qquad 0 \leq IP_{NE} \leq \sqrt{2}\approx 1{,}414")
    st.markdown(
        "Cada variável é normalizada para [0, 1] pelo mín–máx regional. O índice é "
        "a distância euclidiana no espaço normalizado: quanto maior, melhor o "
        "**potencial híbrido** (solar + eólico) da localidade."
    )

    st.subheader("Resumo por estado")
    resumo = (
        df.groupby("UF")
        .agg(Estações=("ESTACAO", "count"),
             Solar=("SOLAR_IRRAD_kwh_m2_dia", "mean"),
             Vento=("WIND_SPEED_ms", "mean"),
             IP_NE_mediana=("IP_NE", "median"))
        .reindex(ESTADOS_NE)
        .round(3)
    )
    st.dataframe(resumo, width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISE EXPLORATÓRIA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Análise Exploratória":
    st.title("Análise Exploratória de Dados")
    st.caption("Oito visualizações geradas em `notebooks/01_eda.ipynb`.")

    figuras = [
        ("fig01_estacoes_por_estado.png", "Distribuição de estações por estado"),
        ("fig02_mapa_solar_vento.png",    "Mapa geográfico — solar e vento"),
        ("fig03_boxplot_solar_vento_uf.png", "Boxplots de solar e vento por estado"),
        ("fig04_ipne_scatter.png",        "Diagrama IP-NE (solar × vento normalizados)"),
        ("fig05_ipne_por_estado.png",     "IP-NE por estado"),
        ("fig06_correlacoes.png",         "Correlações com coordenadas"),
        ("fig07_heatmap_correlacao.png",  "Heatmap de correlação"),
        ("fig08_top10_ipne.png",          "Top 10 estações por IP-NE"),
    ]
    for arquivo, legenda in figuras:
        caminho = FIG_DIR / arquivo
        if caminho.exists():
            st.image(str(caminho), caption=legenda, width="stretch")
        else:
            st.info(f"Figura ausente: `{arquivo}`")
        st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — DESEMPENHO DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Desempenho dos Modelos":
    st.title("Desempenho dos Modelos")
    if metricas is None:
        st.warning("Tabela de métricas não encontrada. Execute `python src/train.py`.")
        st.stop()

    st.markdown(
        "Cinco modelos de regressão (KNN, Árvore de Decisão, Random Forest, "
        "AdaBoost e MLP) avaliados por **Holdout 80/20**, **K-Fold (k=10)** e "
        "**Leave-One-Out (LOO)**, para três alvos."
    )

    alvo = st.selectbox("Alvo", list(TARGETS), format_func=lambda a: ALVO_LABEL[a])
    sub = metricas[metricas["alvo"] == alvo].copy()

    # destaca o melhor por R² (Leave-One-Out — métrica de seleção para N pequeno)
    melhor_idx = sub["loo_r2"].idxmax()
    melhor_modelo = sub.loc[melhor_idx, "modelo"]
    st.success(f"**Melhor modelo para {alvo}:** {melhor_modelo} "
               f"(R² LOO = {sub.loc[melhor_idx, 'loo_r2']:.3f})")

    cols_show = {
        "modelo": "Modelo",
        "holdout_r2": "R² Holdout", "holdout_mae": "MAE Holdout", "holdout_rmse": "RMSE Holdout",
        "kfold_r2_mean": "R² K-Fold", "kfold_r2_std": "± R² K-Fold",
        "kfold_mae_mean": "MAE K-Fold", "kfold_rmse_mean": "RMSE K-Fold",
        "loo_r2": "R² LOO", "loo_mae": "MAE LOO", "loo_rmse": "RMSE LOO",
    }
    tabela = sub[list(cols_show)].rename(columns=cols_show).set_index("Modelo").round(3)
    st.dataframe(
        tabela.style.highlight_max(subset=["R² Holdout", "R² K-Fold", "R² LOO"], color="#c6efce"),
        width="stretch",
    )

    st.subheader("Comparação visual")
    c1, c2 = st.columns(2)
    f9 = REPORTS_DIR / "fig09_comparacao_modelos.png"
    f10 = REPORTS_DIR / "fig10_previsto_vs_real.png"
    if f9.exists():
        c1.image(str(f9), caption="R² por modelo e alvo (Leave-One-Out)", width="stretch")
    if f10.exists():
        c2.image(str(f10), caption="Previsto vs. real (LOO)", width="stretch")

    f11 = REPORTS_DIR / "fig11_ablacao_altitude.png"
    if f11.exists():
        st.image(str(f11),
                 caption="Importância por ablação: incluir ALT reduz o R² — por isso os modelos usam apenas LAT/LON.",
                 width="content")

    with st.expander("Hiperparâmetros selecionados (GridSearch / RandomizedSearch)"):
        st.dataframe(sub[["modelo", "busca", "best_params"]].set_index("modelo"),
                     width="stretch")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — PREVISÃO INTERATIVA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Previsão Interativa":
    st.title("Previsão Interativa do Potencial Renovável")
    st.markdown(
        "Informe as **coordenadas geográficas** de um ponto do Nordeste para "
        "estimar a irradiação solar, a velocidade do vento e o índice IP-NE."
    )

    if not ARTEFATOS_OK:
        st.warning("Modelos não encontrados. Execute `python src/train.py` primeiro.")
        st.stop()

    # As features (e a ordem) vêm do metadata gravado pelo treino — hoje LAT/LON.
    feat_order = (metadata or {}).get("features_ordem", ["LAT", "LON"])
    fr = (metadata or {}).get("features", {})
    rotulos  = {"LAT": "Latitude (graus)", "LON": "Longitude (graus)", "ALT": "Altitude (m)"}
    fallback = {"LAT": (-18.0, -1.0, -8.0), "LON": (-48.0, -34.0, -40.0), "ALT": (0.0, 1300.0, 300.0)}

    cols = st.columns(len(feat_order))
    valores = {}
    for col, f in zip(cols, feat_order):
        if f in fr:
            fmin, fmax, fdef = float(fr[f]["min"]), float(fr[f]["max"]), float(fr[f]["mean"])
        else:
            fmin, fmax, fdef = fallback.get(f, (0.0, 100.0, 50.0))
        margem = (fmax - fmin) * 0.15 or 1.0
        valores[f] = col.number_input(
            rotulos.get(f, f), value=round(fdef, 4),
            min_value=fmin - margem, max_value=fmax + margem, format="%.4f",
        )

    if st.button("Prever", type="primary", width="stretch"):
        X = np.array([[valores[f] for f in feat_order]])
        lat, lon = valores.get("LAT"), valores.get("LON")
        pred = {alvo: float(modelos[alvo].predict(X)[0]) for alvo in modelos}

        solar = pred.get("SOLAR_IRRAD")
        wind  = pred.get("WIND_SPEED")
        ipne_modelo = pred.get("IP_NE")

        m1, m2, m3 = st.columns(3)
        if solar is not None:
            m1.metric("Irradiação Solar", f"{solar:.2f}", "kWh/m²/dia")
        if wind is not None:
            m2.metric("Velocidade do Vento", f"{wind:.2f}", "m/s")
        if ipne_modelo is not None:
            pct = (df["IP_NE"] < ipne_modelo).mean() * 100
            m3.metric("IP-NE (modelo)", f"{ipne_modelo:.3f}", f"percentil {pct:.0f}")

        if solar is not None and wind is not None:
            ipne_calc = computar_ip_ne(solar, wind, metadata, df)
            st.caption(
                f"IP-NE recalculado a partir do solar/vento previstos: "
                f"**{ipne_calc:.3f}** (verificação cruzada com o modelo de IP-NE)."
            )

        # Localização do ponto vs. estações existentes
        if lat is not None and lon is not None:
            st.subheader("Localização do ponto previsto")
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(df["LON"], df["LAT"], c=df["IP_NE"], cmap="RdYlGn",
                            s=45, alpha=0.7, edgecolors="grey", linewidths=0.3)
            ax.scatter([lon], [lat], marker="*", s=400, color="blue",
                       edgecolors="white", linewidths=1.2, label="Ponto previsto", zorder=5)
            plt.colorbar(sc, ax=ax, label="IP-NE das estações")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.legend(loc="best")
            ax.grid(alpha=0.3, linestyle="--")
            st.pyplot(fig, width="content")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — RANKING & MAPA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Ranking & Mapa":
    st.title("Ranking & Mapa do Potencial Renovável")

    metrica = st.radio("Métrica", ["IP_NE", "SOLAR_IRRAD", "WIND_SPEED"],
                       format_func=lambda a: ALVO_LABEL[a], horizontal=True)
    col = TARGETS[metrica]

    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Mapa das estações")
        fig, ax = plt.subplots(figsize=(7, 6.5))
        sc = ax.scatter(df["LON"], df["LAT"], c=df[col], cmap="RdYlGn",
                        s=55, alpha=0.8, edgecolors="grey", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label=ALVO_LABEL[metrica])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3, linestyle="--")
        st.pyplot(fig, width="stretch")

    with c2:
        st.subheader("Top 15 estações")
        top = (df.nlargest(15, col)[["ESTACAO", "UF", col]]
               .reset_index(drop=True).round(3))
        top.index += 1
        top.columns = ["Estação", "UF", ALVO_LABEL[metrica]]
        st.dataframe(top, width="stretch", height=560)

    st.divider()
    st.subheader("Distribuição por estado")
    ordem = df.groupby("UF")[col].median().sort_values(ascending=False)
    st.bar_chart(ordem)
