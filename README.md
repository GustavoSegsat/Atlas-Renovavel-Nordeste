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
├── Script do dataset/coleta.py        # coleta + processamento dos dados do INMET
├── data/processed/                    # dataset consolidado (média por estação)
├── notebooks/
│   ├── 01_eda.ipynb                   # análise exploratória (8 figuras)
│   ├── 02_modeling.ipynb              # modelagem e validação
│   └── fig01..fig08*.png              # figuras do EDA
├── src/train.py                       # pipeline de treino + MLflow
├── app/app.py                         # dashboard Streamlit
├── reports/                           # tabelas e figuras de resultados (geradas)
├── models/                            # melhores modelos + metadata (gerados)
├── mlruns/                            # experimentos do MLflow (gerados)
├── Dockerfile · docker-compose.yml    # containerização
└── requirements.txt
```

> `data/raw/`, `models/`, `mlruns/` e `reports/` são gerados e não versionados.

---

## 🚀 Como executar (local)

```bash
# 1. Ambiente
python -m venv .venv
.venv\Scripts\activate            # Windows  (Linux/Mac: source .venv/bin/activate)
pip install -r requirements.txt

# 2. (Opcional) Recoletar os dados do INMET — já há um CSV processado no repositório
python "Script do dataset/coleta.py"

# 3. Treinar os modelos (gera models/, reports/ e mlruns/)
python src/train.py
#    Variações: python src/train.py --alvos IP_NE   |   --rapido

# 4. Visualizar os experimentos no MLflow
> O `train.py` define `MLFLOW_ALLOW_FILE_STORE=true` automaticamente. Para abrir a
> UI manualmente em outro terminal, exporte a mesma variável antes de `mlflow ui`.         
 # abre em http://localhost:5000

# 5. Abrir o dashboard
streamlit run app/app.py     # abre em http://localhost:8501
```



---

## 🐳 Como executar (Docker)

```bash
docker compose up --build
```

Sobem três serviços: `trainer` (treina uma vez), `mlflow`
(http://localhost:5000) e `dashboard` (http://localhost:8501).

---

## 🧠 Metodologia

| Item | Definição |
|---|---|
| **Features** | `LAT`, `LON` (a altitude é descartada — reduz o R², ver ablação) |
| **Alvos** | `SOLAR_IRRAD` (kWh/m²/dia), `WIND_SPEED` (m/s), `IP_NE` |
| **IP-NE** | `√(SOLAR_norm² + WIND_norm²)`, com normalização mín–máx regional (0 a √2) |
| **Modelos** | KNN, Árvore de Decisão, Random Forest, AdaBoost, MLP |
| **Busca** | `GridSearchCV` (4 modelos) e `RandomizedSearchCV` (MLP) |
| **Validação** | Holdout 80/20, K-Fold (k=10) e Leave-One-Out (métrica de seleção) |
| **Métricas** | R², MAE, RMSE |
| **Dedup** | estações com mesmo `COD_WMO` são agregadas (155 registros → 134 estações) |

---

## 📊 Principais resultados (R² Leave-One-Out)

| Alvo | Melhor modelo | R² (LOO) |
|---|---|---|
| Velocidade do vento | KNN | ≈ 0,15 |
| Índice IP-NE | Random Forest | ≈ 0,09 |
| Irradiação solar | KNN | ≈ 0,06 |

**Leitura honesta dos resultados:** o **vento** tem estrutura espacial clara
(litoral × interior) e é o alvo mais previsível; a **irradiação solar** é quase
uniforme no semiárido, restando pouca variância espacial para aprender a partir de
coordenadas (R² ≈ 0). O R² = 0,89 do Kriging no artigo original foi obtido em **um
único estado** com grade densa — cenário bem mais favorável que o Leave-One-Out
inter-regional usado aqui. Os números exatos ficam em
[`reports/model_comparison.csv`](reports/model_comparison.csv).

---

## 📚 Fonte dos dados

INMET — Instituto Nacional de Meteorologia, Banco de Dados Históricos
(`portal.inmet.gov.br`). Estações automáticas dos 9 estados do Nordeste
(AL, BA, CE, MA, PB, PE, PI, RN, SE), 2022–2025.
