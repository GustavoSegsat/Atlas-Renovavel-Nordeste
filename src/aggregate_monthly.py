"""
Agrega os dados horários já coletados pelo INMET (data/raw/*.csv) em uma
climatologia MENSAL por estação, em vez da média anual única usada em
inmet_medias_estacoes.csv.

Motivação
---------
A irradiação solar e o vento no Nordeste variam fortemente com a estação do
ano (ângulo solar, regime de alísios). Ao colapsar 2022-2025 em um único
valor por estação, esse sinal sazonal — que é real e mensurável a partir dos
próprios dados do INMET — se perde, restando apenas a variação espacial
(fraca em escala regional). Agregar por (estação, mês) preserva esse sinal
sazonal sem usar nenhuma fonte externa: ainda são os mesmos registros
horários do INMET, apenas agrupados em uma granularidade mais fina.

Aplica os mesmos filtros de qualidade do fetch_inmet.py:
  • Apenas horas 05h–18h UTC para irradiação solar
  • Mínimo de MIN_HORAS_SOLAR_DIA leituras horárias válidas por dia
  • Mínimo de MIN_HORAS_VENTO_DIA leituras válidas por dia para vento

Saída:
  data/processed/inmet_medias_mensais.csv
    COD_WMO, UF, ESTACAO, LAT, LON, ALT, MES,
    SOLAR_IRRAD_kwh_m2_dia, WIND_SPEED_ms, dias_validos_solar, dias_validos_vento

Uso:
    python src/aggregate_monthly.py
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd

from fetch_inmet import HORA_SOLAR_MAX, HORA_SOLAR_MIN, MIN_HORAS_SOLAR_DIA, MIN_HORAS_VENTO_DIA

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
log = logging.getLogger(__name__)

ROOT     = Path(__file__).parent.parent
RAW_DIR  = ROOT / "data" / "raw"
PROC_DIR = ROOT / "data" / "processed"
META_CSV = PROC_DIR / "inmet_medias_estacoes.csv"
OUT_CSV  = PROC_DIR / "inmet_medias_mensais.csv"

NOME_RE = re.compile(r"([A-Z]{2})_([A-Za-z0-9]+)_(\d{4})\.csv")


def main() -> None:
    meta = pd.read_csv(META_CSV)[["COD_WMO", "UF", "ESTACAO", "LAT", "LON", "ALT"]]
    meta = meta.drop_duplicates("COD_WMO").set_index("COD_WMO")

    solar_frames, vento_frames = [], []

    for f in sorted(RAW_DIR.glob("*.csv")):
        m = NOME_RE.match(f.name)
        if not m:
            continue
        _, cod, _ano = m.groups()
        if cod not in meta.index:
            continue  # estação sem média anual válida (já descartada no fetch_inmet)

        d = pd.read_csv(f, parse_dates=["datetime_utc"])
        d["hora"] = d["datetime_utc"].dt.hour
        d["mes"]  = d["datetime_utc"].dt.month

        # ── Irradiação solar: mesma janela e limiar diário do fetch_inmet.py ──
        ds = d.loc[(d["hora"] >= HORA_SOLAR_MIN) & (d["hora"] <= HORA_SOLAR_MAX) &
                   d["rad_kJm2"].notna()].copy()
        ds["data"] = ds["datetime_utc"].dt.date
        diario = (ds.groupby("data")
                    .agg(soma=("rad_kJm2", "sum"), n=("rad_kJm2", "count"), mes=("mes", "first"))
                    .query("n >= @MIN_HORAS_SOLAR_DIA"))
        if not diario.empty:
            diario["solar_kwh"] = diario["soma"] / 3600.0
            diario["COD_WMO"] = cod
            solar_frames.append(diario[["mes", "solar_kwh", "COD_WMO"]])

        # ── Vento: mesmo limiar diário do fetch_inmet.py ──────────────────────
        dv = d.loc[d["vento_ms"].notna()].copy()
        dv["data"] = dv["datetime_utc"].dt.date
        diariov = (dv.groupby("data")
                     .agg(media=("vento_ms", "mean"), n=("vento_ms", "count"), mes=("mes", "first"))
                     .query("n >= @MIN_HORAS_VENTO_DIA"))
        if not diariov.empty:
            diariov["COD_WMO"] = cod
            vento_frames.append(diariov[["mes", "media", "COD_WMO"]])

    if not solar_frames or not vento_frames:
        log.error("Nenhum dado horário encontrado em %s — rode fetch_inmet.py primeiro.", RAW_DIR)
        return

    daily_solar = pd.concat(solar_frames, ignore_index=True)
    daily_vento = pd.concat(vento_frames, ignore_index=True)

    mensal_solar = (daily_solar.groupby(["COD_WMO", "mes"])
                     .agg(SOLAR_IRRAD_kwh_m2_dia=("solar_kwh", "mean"),
                          dias_validos_solar=("solar_kwh", "count"))
                     .reset_index())
    mensal_vento = (daily_vento.groupby(["COD_WMO", "mes"])
                     .agg(WIND_SPEED_ms=("media", "mean"),
                          dias_validos_vento=("media", "count"))
                     .reset_index())

    df = mensal_solar.merge(mensal_vento, on=["COD_WMO", "mes"], how="inner")
    df = df.merge(meta, on="COD_WMO", how="inner")
    df.rename(columns={"mes": "MES"}, inplace=True)
    df = df[["COD_WMO", "UF", "ESTACAO", "LAT", "LON", "ALT", "MES",
             "SOLAR_IRRAD_kwh_m2_dia", "WIND_SPEED_ms",
             "dias_validos_solar", "dias_validos_vento"]]
    df["SOLAR_IRRAD_kwh_m2_dia"] = df["SOLAR_IRRAD_kwh_m2_dia"].round(4)
    df["WIND_SPEED_ms"] = df["WIND_SPEED_ms"].round(4)
    df.sort_values(["UF", "ESTACAO", "MES"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
    log.info("Dataset mensal salvo -> %s  (%d linhas, %d estacoes x ~12 meses)",
              OUT_CSV, len(df), df["COD_WMO"].nunique())


if __name__ == "__main__":
    main()
