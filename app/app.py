"""
Atlas Renovável do Nordeste — Dashboard Streamlit
=================================================
Interface interativa que reúne:
  • Visão geral do dataset e do índice IP-NE
  • Análise exploratória (8 figuras do notebook 01_eda.ipynb)
  • Desempenho comparativo dos modelos — pipeline principal: granularidade
    mensal (src/train_mensal.py); versão anual (src/train.py) disponível como
    diagnóstico metodológico (por que a granularidade mudou)
  • Previsão interativa: LAT/LON/UF/MÊS → irradiação solar, vento e IP-NE
  • Ranking e mapa do potencial renovável por estação (média anual ou por mês)

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

ROOT          = Path(__file__).resolve().parents[1]
DATA_CSV      = ROOT / "data" / "processed" / "inmet_medias_estacoes.csv"
MONTHLY_CSV   = ROOT / "data" / "processed" / "inmet_medias_mensais.csv"
MODELS_DIR    = ROOT / "models"
REPORTS_DIR   = ROOT / "reports"
FIG_DIR       = ROOT / "notebooks"

FEATURES         = ["LAT", "LON", "UF_ENC"]
FEATURES_MENSAL  = ["LAT", "LON", "UF_ENC", "MES"]
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
MESES_PT = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril", 5: "Maio", 6: "Junho",
    7: "Julho", 8: "Agosto", 9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro",
}

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


# ─── Pipeline principal: granularidade MENSAL (estação × mês) ─────────────────
# A versão anual (134 estações, 1 valor cada) tinha R² ≈ 0 — pouca variância
# espacial sobra quando se colapsa o ciclo sazonal solar/eólico em uma média.
# Agregar por (estação, mês) preserva esse sinal sazonal real (ver
# src/aggregate_monthly.py e src/train_mensal.py) sem usar fonte externa.
# A versão anual fica disponível como diagnóstico metodológico mais abaixo.

@st.cache_data(show_spinner=False)
def carregar_dados_mensal() -> pd.DataFrame | None:
    if not MONTHLY_CSV.exists():
        return None
    df = pd.read_csv(MONTHLY_CSV)
    solar, wind = df["SOLAR_IRRAD_kwh_m2_dia"], df["WIND_SPEED_ms"]
    df["SOLAR_NORM"] = (solar - solar.min()) / (solar.max() - solar.min())
    df["WIND_NORM"]  = (wind  - wind.min())  / (wind.max()  - wind.min())
    df["IP_NE"]      = np.sqrt(df["SOLAR_NORM"] ** 2 + df["WIND_NORM"] ** 2)
    return df


@st.cache_data(show_spinner=False)
def agregar_mensal_por_estacao(df_mensal: pd.DataFrame) -> pd.DataFrame:
    """Média anual por estação — usada no mapa/ranking para evitar 12 linhas por estação."""
    return (df_mensal.groupby("COD_WMO", as_index=False)
            .agg(UF=("UF", "first"), ESTACAO=("ESTACAO", "first"),
                 LAT=("LAT", "mean"), LON=("LON", "mean"),
                 SOLAR_IRRAD_kwh_m2_dia=("SOLAR_IRRAD_kwh_m2_dia", "mean"),
                 WIND_SPEED_ms=("WIND_SPEED_ms", "mean"),
                 IP_NE=("IP_NE", "mean")))


@st.cache_data(show_spinner=False)
def carregar_metricas_mensal() -> pd.DataFrame | None:
    f = REPORTS_DIR / "model_comparison_mensal.csv"
    return pd.read_csv(f) if f.exists() else None


@st.cache_data(show_spinner=False)
def carregar_metadata_mensal() -> dict | None:
    f = MODELS_DIR / "metadata_mensal.json"
    return json.loads(f.read_text(encoding="utf-8")) if f.exists() else None


@st.cache_resource(show_spinner=False)
def carregar_modelos_mensal() -> dict:
    modelos = {}
    for alvo in TARGETS:
        f = MODELS_DIR / f"best_{alvo}_mensal.joblib"
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

df_mensal       = carregar_dados_mensal()
metricas_mensal = carregar_metricas_mensal()
metadata_mensal = carregar_metadata_mensal()
modelos_mensal  = carregar_modelos_mensal()
df_estacoes_mensal = agregar_mensal_por_estacao(df_mensal) if df_mensal is not None else None

ARTEFATOS_MENSAL_OK = bool(modelos_mensal) and df_mensal is not None


# ─── Barra lateral ────────────────────────────────────────────────────────────

st.sidebar.title("🌞🌬️ Atlas Renovável")
st.sidebar.caption("Potencial de geração renovável no Nordeste do Brasil")
pagina = st.sidebar.radio(
    "Navegação",
    ["Visão Geral", "Análise Exploratória", "Desempenho dos Modelos",
     "Previsão Interativa", "Ranking & Mapa"],
)

if not ARTEFATOS_MENSAL_OK:
    st.sidebar.warning(
        "Modelos mensais não encontrados em `models/`.\n\n"
        "Execute o treino primeiro:\n\n"
        "```\npython src/aggregate_monthly.py\npython src/train_mensal.py\n```"
    )
else:
    st.sidebar.success("Modelos (mensal) carregados ✔")

st.sidebar.caption(
    "Pipeline principal: granularidade **mensal** (estação × mês). "
    "A versão anual original fica disponível como diagnóstico metodológico "
    "na página *Desempenho dos Modelos*."
)

st.sidebar.divider()
st.sidebar.metric("Estações", f"{len(df)}")
st.sidebar.metric("Estados", f"{df['UF'].nunique()}")


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ═══════════════════════════════════════════════════════════════════════════════
if pagina == "Visão Geral":
    st.title("Atlas Renovável do Nordeste")
    n_est = (metadata or {}).get("n_estacoes", len(df))
    st.markdown(
        "Sistema de **aprendizado de máquina** para mapeamento e previsão do "
        "potencial de geração de energia **solar, eólica e híbrida** nos nove "
        "estados do Nordeste. O índice **IP-NE** estende a metodologia IP-PB "
        f"(Ferreira et al., 2023) para escala regional, cobrindo **{n_est} estações** "
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
    st.dataframe(resumo, width='stretch')

    st.info(
        "Este panorama usa a média anual por estação (1 valor por estação, "
        "como na primeira versão do projeto). Os **modelos preditivos** "
        "(página *Desempenho dos Modelos*) usam a granularidade "
        "**estação × mês**, que preserva o ciclo sazonal solar/eólico — "
        "ver detalhes na aba de diagnóstico metodológico.",
        icon="ℹ️",
    )


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
            st.image(str(caminho), caption=legenda, width='stretch')
        else:
            st.info(f"Figura ausente: `{arquivo}`")
        st.divider()


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — DESEMPENHO DOS MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Desempenho dos Modelos":
    st.title("Desempenho dos Modelos")

    if metricas_mensal is None:
        st.warning("Tabela de métricas mensais não encontrada. Execute `python src/train_mensal.py`.")
        st.stop()

    st.markdown(
        "Cinco modelos de regressão (KNN, Árvore de Decisão, Random Forest, "
        "AdaBoost e MLP) avaliados na granularidade **estação × mês** "
        "(N ≈ 1.564) por **Holdout 80/20** e **K-Fold (k=10)** — critério de "
        "seleção do melhor modelo. Um **GroupKFold por estação** é reportado à "
        "parte como checagem de honestidade (nenhum mês da estação de teste "
        "aparece no treino), não usado para seleção."
    )

    alvo = st.selectbox("Alvo", list(TARGETS), format_func=lambda a: ALVO_LABEL[a])
    sub = metricas_mensal[metricas_mensal["alvo"] == alvo].copy()

    # destaca o melhor por R² (K-Fold — critério de seleção, ver train_mensal.py)
    melhor_idx = sub["kfold_r2_mean"].idxmax()
    melhor_modelo = sub.loc[melhor_idx, "modelo"]
    st.success(
        f"**Melhor modelo para {alvo}:** {melhor_modelo} "
        f"(R² K-Fold = {sub.loc[melhor_idx, 'kfold_r2_mean']:.3f} | "
        f"R² GroupKFold-estação = {sub.loc[melhor_idx, 'group_r2']:.3f})"
    )

    cols_show = {
        "modelo": "Modelo",
        "holdout_r2": "R² Holdout", "holdout_mae": "MAE Holdout", "holdout_rmse": "RMSE Holdout",
        "kfold_r2_mean": "R² K-Fold", "kfold_r2_std": "± R² K-Fold",
        "kfold_mae_mean": "MAE K-Fold", "kfold_rmse_mean": "RMSE K-Fold",
        "group_r2": "R² GroupKFold (estação)",
    }
    tabela = sub[list(cols_show)].rename(columns=cols_show).set_index("Modelo").round(3)
    st.dataframe(
        tabela.style.highlight_max(
            subset=["R² Holdout", "R² K-Fold", "R² GroupKFold (estação)"], color="#c6efce"
        ),
        width='stretch',
    )

    st.subheader("Comparação visual")
    c1, c2 = st.columns(2)
    f9 = REPORTS_DIR / "fig09_comparacao_modelos_mensal.png"
    f10 = REPORTS_DIR / "fig10_previsto_vs_real_mensal.png"
    if f9.exists():
        c1.image(str(f9), caption="R² por modelo e alvo (K-Fold)", width='stretch')
    if f10.exists():
        c2.image(str(f10), caption="Previsto vs. real (K-Fold)", width='stretch')

    with st.expander("Hiperparâmetros selecionados (GridSearch / RandomizedSearch)"):
        st.dataframe(sub[["modelo", "busca", "best_params"]].set_index("modelo"),
                     width='stretch')

    st.divider()
    with st.expander("🕰️ Versão anterior (granularidade anual) — diagnóstico metodológico", expanded=False):
        st.markdown(
            "A primeira versão do projeto colapsava 2022–2025 em **um único valor "
            "por estação** (N = 134) e usava Leave-One-Out como critério de "
            "seleção, já que o K-Fold com tão poucas amostras por fold era "
            "instável. Mesmo assim o R² ficou **próximo de zero** — ao resumir "
            "os dados horários em uma média anual, o ciclo sazonal solar/eólico "
            "(o sinal mais forte e fisicamente bem estabelecido do problema) é "
            "descartado, restando só a variação espacial entre estações, que é "
            "fraca em escala regional.\n\n"
            "A correção foi agregar por **(estação, mês)** em vez de por "
            "estação (`src/aggregate_monthly.py`), preservando o sinal sazonal "
            "real já presente nos dados horários do INMET — sem usar nenhuma "
            "fonte externa. Isso elevou N de 134 para ≈1.564 e o R² K-Fold de "
            "~0,05–0,26 para ~0,73–0,92 (acima). Os resultados anuais ficam "
            "abaixo, para registro do diagnóstico."
        )
        if metricas is not None:
            sub_anual = metricas[metricas["alvo"] == alvo].copy()
            melhor_idx_a = sub_anual["loo_r2"].idxmax()
            st.info(
                f"Melhor modelo anual para {alvo}: {sub_anual.loc[melhor_idx_a, 'modelo']} "
                f"(R² LOO = {sub_anual.loc[melhor_idx_a, 'loo_r2']:.3f})"
            )
            cols_show_a = {
                "modelo": "Modelo",
                "holdout_r2": "R² Holdout", "kfold_r2_mean": "R² K-Fold", "loo_r2": "R² LOO",
            }
            st.dataframe(
                sub_anual[list(cols_show_a)].rename(columns=cols_show_a).set_index("Modelo").round(3),
                width='stretch',
            )
        ca, cb = st.columns(2)
        f9a, f10a = REPORTS_DIR / "fig09_comparacao_modelos.png", REPORTS_DIR / "fig10_previsto_vs_real.png"
        if f9a.exists():
            ca.image(str(f9a), caption="Anual — R² por modelo (Leave-One-Out)", width='stretch')
        if f10a.exists():
            cb.image(str(f10a), caption="Anual — Previsto vs. real (LOO)", width='stretch')
        f11 = REPORTS_DIR / "fig11_ablacao_altitude.png"
        if f11.exists():
            st.image(str(f11),
                     caption="Ablação de features (versão anual): UF melhora o R² em todos os alvos; ALT reduz.",
                     width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — PREVISÃO INTERATIVA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Previsão Interativa":
    st.title("Previsão Interativa do Potencial Renovável")
    st.markdown(
        "Informe as **coordenadas geográficas**, o **estado** e o **mês** de um "
        "ponto do Nordeste para estimar a irradiação solar, a velocidade do "
        "vento e o IP-NE esperados naquele mês (modelo treinado na granularidade "
        "estação × mês)."
    )

    if not ARTEFATOS_MENSAL_OK:
        st.warning("Modelos mensais não encontrados. Execute `python src/train_mensal.py` primeiro.")
        st.stop()

    # Features e ordem vêm do metadata gravado pelo treino mensal
    feat_order  = (metadata_mensal or {}).get("features_ordem", FEATURES_MENSAL)
    uf_classes  = (metadata_mensal or {}).get("uf_classes", ESTADOS_NE)
    fr = (metadata_mensal or {}).get("features", {})
    rotulos  = {"LAT": "Latitude (graus)", "LON": "Longitude (graus)", "ALT": "Altitude (m)"}
    fallback = {"LAT": (-18.0, -1.0, -8.0), "LON": (-48.0, -34.0, -40.0), "ALT": (0.0, 1300.0, 300.0)}

    # UF e MES são selecionados separadamente (não são contínuos)
    uf_selecionado = st.selectbox("Estado (UF)", uf_classes, index=uf_classes.index("CE") if "CE" in uf_classes else 0)
    uf_enc = uf_classes.index(uf_selecionado)
    mes_selecionado = st.selectbox("Mês", list(MESES_PT), format_func=lambda m: MESES_PT[m], index=5)

    # Inputs numéricos para as demais features (LAT, LON)
    feats_num = [f for f in feat_order if f not in ("UF_ENC", "MES")]
    cols = st.columns(len(feats_num))
    valores: dict[str, float] = {"UF_ENC": float(uf_enc), "MES": float(mes_selecionado)}
    for col, f in zip(cols, feats_num):
        if f in fr:
            fmin, fmax, fdef = float(fr[f]["min"]), float(fr[f]["max"]), float(fr[f]["mean"])
        else:
            fmin, fmax, fdef = fallback.get(f, (0.0, 100.0, 50.0))
        margem = (fmax - fmin) * 0.15 or 1.0
        valores[f] = col.number_input(
            rotulos.get(f, f), value=round(fdef, 4),
            min_value=fmin - margem, max_value=fmax + margem, format="%.4f",
        )

    if st.button("Prever", type="primary", width='stretch'):
        X = np.array([[valores[f] for f in feat_order]])
        lat, lon = valores.get("LAT"), valores.get("LON")
        pred = {alvo: float(modelos_mensal[alvo].predict(X)[0]) for alvo in modelos_mensal}

        solar = pred.get("SOLAR_IRRAD")
        wind  = pred.get("WIND_SPEED")
        ipne_modelo = pred.get("IP_NE")

        m1, m2, m3 = st.columns(3)
        if solar is not None:
            m1.metric("Irradiação Solar", f"{solar:.2f}", "kWh/m²/dia")
        if wind is not None:
            m2.metric("Velocidade do Vento", f"{wind:.2f}", "m/s")
        if ipne_modelo is not None:
            pct = (df_mensal["IP_NE"] < ipne_modelo).mean() * 100
            m3.metric("IP-NE (modelo)", f"{ipne_modelo:.3f}", f"percentil {pct:.0f}")

        if solar is not None and wind is not None:
            ipne_calc = computar_ip_ne(solar, wind, metadata_mensal, df_mensal)
            st.caption(
                f"IP-NE recalculado a partir do solar/vento previstos: "
                f"**{ipne_calc:.3f}** (verificação cruzada com o modelo de IP-NE)."
            )

        # Localização do ponto vs. estações existentes (médias anuais por estação)
        if lat is not None and lon is not None:
            st.subheader("Localização do ponto previsto")
            fig, ax = plt.subplots(figsize=(7, 6))
            sc = ax.scatter(df_estacoes_mensal["LON"], df_estacoes_mensal["LAT"],
                            c=df_estacoes_mensal["IP_NE"], cmap="RdYlGn",
                            s=45, alpha=0.7, edgecolors="grey", linewidths=0.3)
            ax.scatter([lon], [lat], marker="*", s=400, color="blue",
                       edgecolors="white", linewidths=1.2, label="Ponto previsto", zorder=5)
            plt.colorbar(sc, ax=ax, label="IP-NE médio anual das estações")
            ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
            ax.legend(loc="best")
            ax.grid(alpha=0.3, linestyle="--")
            st.pyplot(fig, width='stretch')


# ═══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — RANKING & MAPA
# ═══════════════════════════════════════════════════════════════════════════════
elif pagina == "Ranking & Mapa":
    st.title("Ranking & Mapa do Potencial Renovável")

    if not ARTEFATOS_MENSAL_OK:
        st.warning("Dados mensais não encontrados. Execute `python src/aggregate_monthly.py` primeiro.")
        st.stop()

    c0a, c0b = st.columns(2)
    metrica = c0a.radio("Métrica", ["IP_NE", "SOLAR_IRRAD", "WIND_SPEED"],
                        format_func=lambda a: ALVO_LABEL[a], horizontal=True)
    periodo = c0b.selectbox(
        "Período", ["Média anual"] + list(MESES_PT),
        format_func=lambda p: p if p == "Média anual" else MESES_PT[p],
    )
    col = TARGETS[metrica]

    if periodo == "Média anual":
        df_periodo = df_estacoes_mensal
    else:
        df_periodo = df_mensal[df_mensal["MES"] == periodo]

    c1, c2 = st.columns([3, 2])

    with c1:
        st.subheader("Mapa das estações")
        fig, ax = plt.subplots(figsize=(7, 6.5))
        sc = ax.scatter(df_periodo["LON"], df_periodo["LAT"], c=df_periodo[col], cmap="RdYlGn",
                        s=55, alpha=0.8, edgecolors="grey", linewidths=0.3)
        plt.colorbar(sc, ax=ax, label=ALVO_LABEL[metrica])
        ax.set_xlabel("Longitude"); ax.set_ylabel("Latitude")
        ax.grid(alpha=0.3, linestyle="--")
        st.pyplot(fig, width='stretch')

    with c2:
        st.subheader("Top 15 estações")
        top = (df_periodo.nlargest(15, col)[["ESTACAO", "UF", col]]
               .reset_index(drop=True).round(3))
        top.index += 1
        top.columns = ["Estação", "UF", ALVO_LABEL[metrica]]
        st.dataframe(top, width='stretch', height=560)

    st.divider()
    st.subheader("Distribuição por estado")
    ordem = df_periodo.groupby("UF")[col].median().sort_values(ascending=False)
    st.bar_chart(ordem)
