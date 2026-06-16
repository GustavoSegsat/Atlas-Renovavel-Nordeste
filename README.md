# Atlas Renovável do Nordeste 🌞🌬️

**Disciplina:** Machine Learning I e Projeto 3
**Instituição:** CESAR School

| Membro | GitHub |
|---|---|
| Leonardo José Amaral de Méllo | [@ljam2](https://github.com/ljam2) |
| Gustavo Carneiro Ismael de Carvalho | [@gcic](https://github.com/gcic) |

---

Sistema de **aprendizado de máquina** para mapeamento e previsão do potencial de
geração de energia **solar, eólica e híbrida** nos nove estados do Nordeste do
Brasil. O índice **IP-NE** estende a metodologia IP-PB (Ferreira et al., 2023)
para escala regional, usando dados abertos do **INMET** (2022–2025).

O problema é formulado como **regressão espacial**: prever, a partir de
coordenadas geográficas, a irradiação solar, a velocidade do vento e o índice
IP-NE de qualquer ponto da região.

---

## 📂 Estrutura do projeto

```
Atlas-Renovavel-Nordeste/
├── src/fetch_inmet.py                 # coleta + processamento dos dados do INMET
├── src/aggregate_monthly.py           # agrega dados horários em climatologia mensal por estação
├── data/processed/                    # dataset por estação e dataset mensal (estação × mês)
├── notebooks/
│   ├── 01_eda.ipynb                   # análise exploratória (8 figuras)
│   ├── 02_modeling.ipynb              # modelagem e validação
│   └── fig01..fig08*.png              # figuras do EDA
├── src/train_mensal.py                # pipeline de treino PRINCIPAL (estação × mês) + MLflow
├── src/train.py                       # pipeline anual — mantido como diagnóstico metodológico
├── app/app.py                         # dashboard Streamlit
├── reports/                           # tabelas e figuras de resultados (geradas)
├── models/                            # melhores modelos + metadata (gerados)
├── mlruns/                            # experimentos do MLflow (gerados)
├── Dockerfile · docker-compose.yml    # containerização
└── requirements.txt
```

> `data/raw/`, `data/processed/`, `models/`, `mlruns/` e `reports/` são gerados e não versionados.

---

## 🚀 Como executar (local)

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. Coletar os dados do INMET (gera data/raw/*.csv e data/processed/inmet_medias_estacoes.csv)
python src/fetch_inmet.py

# 3. Agregar a climatologia mensal (gera data/processed/inmet_medias_mensais.csv)
python src/aggregate_monthly.py

# 4. Treinar os modelos
python src/train_mensal.py          # pipeline PRINCIPAL (estação × mês) — gera models/, reports/ e mlruns/
python src/train.py                 # pipeline anual — mantido como diagnóstico metodológico (ver seção de Resultados)
#    Variações: --alvos IP_NE   |   --rapido

# 5. Visualizar os experimentos no MLflow
> O treino define `MLFLOW_ALLOW_FILE_STORE=true` automaticamente. Para abrir a
> UI manualmente em outro terminal, exporte a mesma variável antes de `mlflow ui`.
 # abre em http://localhost:5000

# 6. Abrir o dashboard
streamlit run app/app.py     # abre em http://localhost:8501
```



---

## 🐳 Como executar (Docker)

```bash
docker compose up --build
```

Sobem três serviços: `trainer` (treina uma vez), `mlflow`
(http://localhost:5001) e `dashboard` (http://localhost:8501).

---

## 🧠 Metodologia

| Item | Definição |
|---|---|
| **Granularidade** | estação × mês (N ≈ 1.564) — ver justificativa em [Por que mensal?](#-por-que-granularidade-mensal) |
| **Features** | `LAT`, `LON`, `UF` (estado codificado), `MES` (1–12; altitude descartada via ablação) |
| **Alvos** | `SOLAR_IRRAD` (kWh/m²/dia), `WIND_SPEED` (m/s), `IP_NE` |
| **IP-NE** | `√(SOLAR_norm² + WIND_norm²)`, normalização mín–máx sobre o conjunto estação×mês |
| **Modelos** | KNN, Árvore de Decisão, Random Forest, AdaBoost, MLP |
| **Busca** | `GridSearchCV` (4 modelos) e `RandomizedSearchCV` (MLP) |
| **Validação** | Holdout 80/20, K-Fold (k=10) — critério de seleção — e GroupKFold por estação (checagem de honestidade) |
| **Métricas** | R², MAE, RMSE |
| **Dedup** | estações com mesmo `COD_WMO` são agregadas (155 registros → 134 estações) |

---

## 📊 Principais resultados (R² K-Fold, k=10)

| Alvo | Melhor modelo | R² (K-Fold) | R² (GroupKFold por estação) |
|---|---|---|---|
| Velocidade do vento | Random Forest | ≈ 0,92 | ≈ 0,17 |
| Irradiação solar | Random Forest | ≈ 0,82 | ≈ 0,26 |
| Índice IP-NE | Random Forest | ≈ 0,85 | ≈ 0,33 |

**Leitura dos resultados:** o modelo aprende bem o padrão estação×mês (R² K-Fold
alto), mas o GroupKFold por estação — que testa em estações nunca vistas no
treino — mostra que a generalização espacial pura ainda é mais difícil que a
interpolação temporal, o que é esperado: o sinal sazonal (mês) é mais forte
que o sinal espacial (LAT/LON/UF) nesta região. Os números exatos ficam em
[`reports/model_comparison_mensal.csv`](reports/model_comparison_mensal.csv).

### 🕰️ Por que granularidade mensal?

A primeira versão do projeto (`src/train.py`) colapsava 2022–2025 em **um único
valor por estação** (N = 134) e usava Leave-One-Out como critério de seleção.
O R² ficou **próximo de zero** (≈ 0,05–0,26) — ao resumir os dados horários em
uma média anual, o ciclo sazonal solar/eólico (o sinal mais forte e
fisicamente bem estabelecido do problema: ângulo de incidência solar, regime
de alísios) é descartado, restando só a variação espacial entre estações, que
é fraca em escala regional.

A correção foi agregar por **(estação, mês)** em vez de por estação
(`src/aggregate_monthly.py` → `src/train_mensal.py`), preservando esse sinal
sazonal real já presente nos dados horários do INMET, **sem usar nenhuma fonte
externa**. Isso elevou N de 134 para ≈1.564 e o R² K-Fold de ~0,05–0,26 para
~0,73–0,92. A versão anual é mantida no repositório e no dashboard (aba
"Versão anterior — diagnóstico metodológico" em *Desempenho dos Modelos*) como
registro desse diagnóstico, não como resultado final.

---

## 📚 Fonte dos dados

INMET — Instituto Nacional de Meteorologia, Banco de Dados Históricos
(`portal.inmet.gov.br`). Estações automáticas dos 9 estados do Nordeste
(AL, BA, CE, MA, PB, PE, PI, RN, SE), 2022–2025.
