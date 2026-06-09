# Atlas Renovável do Nordeste — imagem única usada por treino, MLflow e dashboard
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    # MLflow 3.x exige opt-in explícito para o backend de arquivos (./mlruns)
    MLFLOW_ALLOW_FILE_STORE=true

WORKDIR /app

# libgomp1: runtime OpenMP usado por scikit-learn / numpy
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# 8501 = Streamlit | 5000 = MLflow UI
EXPOSE 8501 5000

# Padrão: dashboard (o docker-compose sobrescreve o command de cada serviço)
CMD ["streamlit", "run", "app/app.py", \
     "--server.port=8501", "--server.address=0.0.0.0"]
