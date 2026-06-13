"""
Coleta e processa dados horários do INMET para o Nordeste (2022-2025).

Fonte: ZIPs históricos públicos do portal INMET
  https://portal.inmet.gov.br/uploads/dadoshistoricos/{ANO}.zip

Limpeza de falhas de sensor (sem imputação — dados reais apenas):
  • |valor| >= 9990  (cobre 9999, -9999, 9999.9, etc.)  → NaN
  • Radiação negativa                                    → NaN
  • Vento negativo                                       → NaN
  • Campos vazios / "-"                                  → NaN

Filtro solar: apenas horas 05–18 UTC (metodologia Ferreira et al., 2023).

Saídas:
  data/zips/{ANO}.zip                         – ZIP bruto (cache)
  data/raw/{UF}_{COD}_{ANO}.csv               – série horária limpa por estação/ano
  data/processed/inmet_medias_estacoes.csv    – uma linha por estação (input para train.py)

Schema de inmet_medias_estacoes.csv:
  COD_WMO, UF, ESTACAO, LAT, LON, ALT,
  SOLAR_IRRAD_kwh_m2_dia, WIND_SPEED_ms,
  anos_disponiveis, dias_validos_total
"""

from __future__ import annotations

import io
import logging
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ─── Configurações ────────────────────────────────────────────────────────────

BASE_ZIP_URL = "https://portal.inmet.gov.br/uploads/dadoshistoricos"
YEARS        = [2022, 2023, 2024, 2025]
NE_TAG       = "_NE_"                   # presente em todos os arquivos do Nordeste

# Limiar de falha de sensor: qualquer leitura com |valor| >= este limite é NaN
SENSOR_FAIL = 9990

# Horas UTC consideradas para irradiação solar (igual ao artigo de referência)
HORA_SOLAR_MIN = 5
HORA_SOLAR_MAX = 18

# Mínimo de horas válidas para um dia entrar na média
MIN_HORAS_SOLAR_DIA = 8   # ao menos 8 leituras diurnas válidas (cobertura mínima do dia)
MIN_HORAS_VENTO_DIA = 1   # ao menos 1 leitura válida

ZIP_DIR  = Path(__file__).parent.parent / "data" / "zips"
RAW_DIR  = Path(__file__).parent.parent / "data" / "raw"
PROC_DIR = Path(__file__).parent.parent / "data" / "processed"

for d in (ZIP_DIR, RAW_DIR, PROC_DIR):
    d.mkdir(parents=True, exist_ok=True)

RETRY_LIMIT = 3
RETRY_DELAY = 10   # segundos entre tentativas
CHUNK_SIZE  = 4 * 1024 * 1024  # 4 MB

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ─── Download com cache ────────────────────────────────────────────────────────

def download_zip(year: int) -> Path | None:
    dest = ZIP_DIR / f"{year}.zip"
    if dest.exists() and dest.stat().st_size > 1_000_000:
        log.info("ZIP %d: cache (%s MB).", year, f"{dest.stat().st_size/1e6:.0f}")
        return dest

    url = f"{BASE_ZIP_URL}/{year}.zip"
    log.info("Baixando %s …", url)
    for attempt in range(1, RETRY_LIMIT + 1):
        try:
            r = requests.get(url, timeout=300, stream=True)
            if r.status_code == 200:
                with open(dest, "wb") as f:
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                log.info("  Salvo: %s (%.0f MB)", dest.name,
                         dest.stat().st_size / 1e6)
                return dest
            if r.status_code == 404:
                log.warning("  ZIP %d não disponível (HTTP 404).", year)
                return None
            log.warning("  HTTP %s (tentativa %d)", r.status_code, attempt)
        except requests.RequestException as exc:
            log.warning("  Erro de rede: %s (tentativa %d)", exc, attempt)
        if attempt < RETRY_LIMIT:
            time.sleep(RETRY_DELAY)
    log.error("Falha ao baixar ZIP %d.", year)
    return None


# ─── Parse do CSV INMET ───────────────────────────────────────────────────────

def _to_float(series: pd.Series) -> pd.Series:
    """Converte coluna string (decimal ,) para float; strings vazias e '-' → NaN."""
    cleaned = (
        series.astype(str)
              .str.strip()
              .str.replace(",", ".", regex=False)
    )
    # Trata "-" isolado e variações de "nulo" como vazio antes do cast numérico
    cleaned = cleaned.where(~cleaned.isin(["-", "nan", "null", "NULL", ""]), other="")
    return pd.to_numeric(cleaned, errors="coerce")


def _limpar_sensor(s: pd.Series, negativo_invalido: bool = True) -> pd.Series:
    """Remove falhas de sensor. Nunca preenche — NaN permanece NaN."""
    s = _to_float(s)
    # Threshold de falha de sensor (9999, -9999, 9999.9, etc.)
    s = s.where(s.abs() < SENSOR_FAIL, other=np.nan)
    if negativo_invalido:
        s = s.where(s >= 0, other=np.nan)
    return s


def _encontrar_coluna(colunas: list[str], fragmento: str) -> str | None:
    """Retorna o nome exato da coluna que contém o fragmento (case-insensitive)."""
    for c in colunas:
        if fragmento.upper() in c.upper():
            return c
    return None


def parse_inmet_csv(content: bytes) -> tuple[dict, pd.DataFrame]:
    """
    Lê bytes de um CSV do INMET, extrai metadados e série horária limpa.

    Retorna:
        meta  – dict com COD_WMO, UF, ESTACAO, LAT, LON, ALT
        df    – DataFrame com colunas [datetime_utc, hora, rad_kJm2, vento_ms]
                Apenas registros reais do INMET; falhas de sensor → NaN.
    """
    try:
        text = content.decode("latin-1")
    except UnicodeDecodeError:
        text = content.decode("utf-8", errors="replace")

    linhas = text.splitlines()

    # ── Metadados (primeiras 8 linhas) ────────────────────────────────────────
    raw_meta: dict[str, str] = {}
    for linha in linhas[:8]:
        if ";" in linha:
            chave, _, valor = linha.partition(";")
            raw_meta[chave.strip().rstrip(":")] = valor.strip().rstrip(";")

    def _meta_float(campo: str) -> float:
        v = raw_meta.get(campo, "")
        try:
            return float(str(v).replace(",", ".").strip())
        except (ValueError, AttributeError):
            return np.nan

    meta = {
        "COD_WMO": raw_meta.get("CODIGO (WMO)", "").strip(),
        "UF":      raw_meta.get("UF",            "").strip(),
        "ESTACAO": raw_meta.get("ESTACAO",        "").strip(),
        "LAT":     _meta_float("LATITUDE"),
        "LON":     _meta_float("LONGITUDE"),
        "ALT":     _meta_float("ALTITUDE"),
    }

    # ── Dados horários (linha 9 = cabeçalho; linha 10+ = registros) ──────────
    if len(linhas) < 10:
        return meta, pd.DataFrame()

    cabecalho_raw = linhas[8]
    colunas = [c.strip() for c in cabecalho_raw.split(";")]

    dados_text = "\n".join(linhas[9:])
    try:
        df = pd.read_csv(
            io.StringIO(dados_text),
            sep=";",
            header=None,
            names=colunas,
            dtype=str,
            na_values=["", " ", "///", "null", "NULL"],
            engine="python",
        )
    except Exception as exc:
        log.debug("Erro no read_csv: %s", exc)
        return meta, pd.DataFrame()

    # Remove coluna fantasma criada pelo `;` final de cada linha
    if df.columns[-1] in ("", " ", "nan", "None"):
        df = df.iloc[:, :-1]

    # ── Datetime ──────────────────────────────────────────────────────────────
    col_data = _encontrar_coluna(list(df.columns), "Data")
    col_hora = _encontrar_coluna(list(df.columns), "Hora")
    if col_data is None or col_hora is None:
        return meta, pd.DataFrame()

    hora_limpa = (
        df[col_hora].astype(str)
        .str.replace(" UTC", "", regex=False)
        .str.strip()
        .str.zfill(4)
    )
    df["datetime_utc"] = pd.to_datetime(
        df[col_data].str.strip() + " " +
        hora_limpa.str[:2] + ":" + hora_limpa.str[2:],
        format="%Y/%m/%d %H:%M",
        errors="coerce",
    )
    df.dropna(subset=["datetime_utc"], inplace=True)
    df["hora"] = df["datetime_utc"].dt.hour

    # ── Radiação global (kJ/m²) ───────────────────────────────────────────────
    col_rad = _encontrar_coluna(list(df.columns), "RADIACAO GLOBAL")
    if col_rad:
        df["rad_kJm2"] = _limpar_sensor(df[col_rad], negativo_invalido=True)
    else:
        df["rad_kJm2"] = np.nan

    # ── Velocidade do vento (m/s) ─────────────────────────────────────────────
    col_vento = _encontrar_coluna(list(df.columns), "VENTO, VELOCIDADE HORARIA")
    if col_vento:
        df["vento_ms"] = _limpar_sensor(df[col_vento], negativo_invalido=True)
    else:
        df["vento_ms"] = np.nan

    # ── Resultado final ───────────────────────────────────────────────────────
    resultado = df[["datetime_utc", "hora", "rad_kJm2", "vento_ms"]].copy()
    resultado.sort_values("datetime_utc", inplace=True)
    resultado.reset_index(drop=True, inplace=True)
    return meta, resultado


# ─── Identificação da estação pelo nome do arquivo ───────────────────────────

def meta_do_arquivo(nome_arquivo: str) -> tuple[str, str]:
    """
    Extrai (UF, COD_WMO) do nome padrão INMET:
      INMET_NE_{UF}_{COD}_{CIDADE}_{...}.CSV
    """
    partes = Path(nome_arquivo).stem.split("_")
    uf  = partes[2] if len(partes) > 2 else "XX"
    cod = partes[3] if len(partes) > 3 else "XXXX"
    return uf, cod


# ─── Pipeline principal ───────────────────────────────────────────────────────

def main() -> None:
    # ── Fase 1: download e parse ──────────────────────────────────────────────
    # Acumula: {(UF, COD): {"meta": dict, "frames": [df_ano1, df_ano2, ...]}}
    estacoes: dict[tuple[str, str], dict] = {}

    for year in YEARS:
        zip_path = download_zip(year)
        if zip_path is None:
            continue

        try:
            zf = zipfile.ZipFile(zip_path)
        except zipfile.BadZipFile:
            log.error("ZIP %d corrompido. Delete %s e tente novamente.", year, zip_path)
            continue

        arquivos_ne = sorted(n for n in zf.namelist()
                             if NE_TAG in n and n.upper().endswith(".CSV"))
        log.info("ZIP %d: %d arquivos NE.", year, len(arquivos_ne))

        for nome_arq in arquivos_ne:
            uf_arq, cod_arq = meta_do_arquivo(nome_arq)
            chave = (uf_arq, cod_arq)

            raw_path = RAW_DIR / f"{uf_arq}_{cod_arq}_{year}.csv"

            # Cache: só reutiliza se o arquivo existe E tem ao menos um valor
            # válido de sensor (evita arquivos com timestamps mas sem rad/vento).
            if raw_path.exists() and raw_path.stat().st_size > 200:
                df_ano = pd.read_csv(raw_path, parse_dates=["datetime_utc"])
                has_data = (df_ano["rad_kJm2"].notna().any() or
                            df_ano["vento_ms"].notna().any())
                if has_data:
                    meta_cache = {}
                    if chave not in estacoes:
                        estacoes[chave] = {"meta": meta_cache, "frames": []}
                    estacoes[chave]["frames"].append(df_ano)
                    log.info("  [%d] %s_%s: cache (%d linhas)",
                             year, uf_arq, cod_arq, len(df_ano))
                    continue
                log.info("  [%d] %s_%s: cache ignorado (sensores todos NaN) — reprocessando.",
                         year, uf_arq, cod_arq)

            # Lê e processa o CSV do ZIP
            try:
                conteudo = zf.read(nome_arq)
            except KeyError:
                continue

            meta, df_ano = parse_inmet_csv(conteudo)

            if df_ano.empty:
                log.warning("  [%d] %s_%s: nenhum dado válido", year, uf_arq, cod_arq)
                continue

            # Salva raw (série horária limpa)
            df_ano.to_csv(raw_path, index=False)

            n_rad   = df_ano["rad_kJm2"].notna().sum()
            n_vento = df_ano["vento_ms"].notna().sum()
            log.info("  [%d] %s_%s (%s): %d registros | rad=%d vento=%d",
                     year, uf_arq, cod_arq, meta.get("ESTACAO", "?"),
                     len(df_ano), n_rad, n_vento)

            if chave not in estacoes:
                estacoes[chave] = {"meta": meta, "frames": []}
            elif not estacoes[chave]["meta"].get("COD_WMO"):
                estacoes[chave]["meta"] = meta   # preenche se veio do cache vazio

            estacoes[chave]["frames"].append(df_ano)

        zf.close()

    if not estacoes:
        log.error("Nenhum dado coletado.")
        return

    # ── Fase 2: reconstruir metadados para entradas que vieram só de cache ────
    # Para estações cujo meta ficou vazio (só cache), relê o primeiro ZIP disponível
    # e extrai apenas o cabeçalho.
    estacoes_sem_meta = [(k, v) for k, v in estacoes.items()
                         if not v["meta"].get("COD_WMO")]
    if estacoes_sem_meta:
        log.info("Recuperando metadados de %d estações (apenas cache raw).",
                 len(estacoes_sem_meta))
        zips_abertos: dict[int, zipfile.ZipFile] = {}
        for (uf_k, cod_k), dados in estacoes_sem_meta:
            encontrou = False
            for year in YEARS:
                zp = ZIP_DIR / f"{year}.zip"
                if not zp.exists():
                    continue
                if year not in zips_abertos:
                    try:
                        zips_abertos[year] = zipfile.ZipFile(zp)
                    except zipfile.BadZipFile:
                        continue
                zf2 = zips_abertos[year]
                candidatos = [n for n in zf2.namelist()
                              if f"_{uf_k}_{cod_k}_" in n and n.upper().endswith(".CSV")]
                if candidatos:
                    conteudo = zf2.read(candidatos[0])
                    meta_ok, _ = parse_inmet_csv(conteudo)
                    dados["meta"] = meta_ok
                    encontrou = True
                    break
            if not encontrou:
                log.warning("Metadados não encontrados para %s_%s.", uf_k, cod_k)
        for zf3 in zips_abertos.values():
            zf3.close()

    # ── Fase 3: agregação por estação ─────────────────────────────────────────
    log.info("Agregando %d estações …", len(estacoes))
    linhas_saida: list[dict] = []

    for (uf_k, cod_k), dados in estacoes.items():
        meta = dados["meta"]
        frames = dados["frames"]

        if not frames:
            continue

        df_all = pd.concat(frames, ignore_index=True)
        df_all.sort_values("datetime_utc", inplace=True)

        anos_presentes = sorted(df_all["datetime_utc"].dt.year.dropna().unique().astype(int))
        anos_disponiveis = len(anos_presentes)

        # ── Irradiação solar ──────────────────────────────────────────────────
        # Filtra apenas horas diurnas (05h–18h UTC) conforme o artigo
        df_solar = df_all.loc[
            (df_all["hora"] >= HORA_SOLAR_MIN) &
            (df_all["hora"] <= HORA_SOLAR_MAX) &
            df_all["rad_kJm2"].notna()
        ].copy()

        df_solar["data"] = df_solar["datetime_utc"].dt.date

        # Soma horária por dia → total diário em kJ/m²
        # Só conta dias com ao menos MIN_HORAS_SOLAR_DIA leituras válidas
        diario_solar = (
            df_solar.groupby("data")["rad_kJm2"]
            .agg(soma="sum", n="count")
            .query("n >= @MIN_HORAS_SOLAR_DIA")
        )

        if diario_solar.empty:
            solar_kwh_dia = np.nan
            dias_validos   = 0
        else:
            # Converte kJ/m²/dia → kWh/m²/dia  (÷ 3600)
            diario_solar["solar_kwh"] = diario_solar["soma"] / 3600.0
            solar_kwh_dia = float(diario_solar["solar_kwh"].mean())
            dias_validos  = len(diario_solar)

        # ── Velocidade do vento ───────────────────────────────────────────────
        df_vento = df_all.loc[df_all["vento_ms"].notna()].copy()
        df_vento["data"] = df_vento["datetime_utc"].dt.date

        diario_vento = (
            df_vento.groupby("data")["vento_ms"]
            .agg(media="mean", n="count")
            .query("n >= @MIN_HORAS_VENTO_DIA")
        )

        if diario_vento.empty:
            vento_ms = np.nan
        else:
            vento_ms = float(diario_vento["media"].mean())

        # Usa COD_WMO do meta se disponível; caso contrário, usa o código do arquivo
        cod_wmo = meta.get("COD_WMO") or cod_k

        linhas_saida.append({
            "COD_WMO":              cod_wmo,
            "UF":                   meta.get("UF",      uf_k),
            "ESTACAO":              meta.get("ESTACAO", ""),
            "LAT":                  meta.get("LAT",     np.nan),
            "LON":                  meta.get("LON",     np.nan),
            "ALT":                  meta.get("ALT",     np.nan),
            "SOLAR_IRRAD_kwh_m2_dia": round(solar_kwh_dia, 4) if not np.isnan(solar_kwh_dia) else np.nan,
            "WIND_SPEED_ms":        round(vento_ms, 4)        if not np.isnan(vento_ms)        else np.nan,
            "anos_disponiveis":     anos_disponiveis,
            "dias_validos_total":   dias_validos,
        })

    # ── Fase 4: salva ─────────────────────────────────────────────────────────
    df_out = pd.DataFrame(linhas_saida)

    # Remove estações onde qualquer variável-alvo está ausente
    # (sensor zerado/ausente em todos os anos não gera dado real)
    df_out.dropna(subset=["SOLAR_IRRAD_kwh_m2_dia", "WIND_SPEED_ms"], how="any", inplace=True)
    df_out.sort_values(["UF", "ESTACAO"], inplace=True)
    df_out.reset_index(drop=True, inplace=True)

    out_path = PROC_DIR / "inmet_medias_estacoes.csv"
    df_out.to_csv(out_path, index=False, encoding="utf-8-sig")
    log.info("Dataset salvo → %s  (%d estações)", out_path, len(df_out))

    # ── Resumo final ──────────────────────────────────────────────────────────
    print("\n=== Resumo ===")
    print(f"Estações coletadas  : {len(df_out)}")
    print(f"Estados cobertos    : {sorted(df_out['UF'].unique())}")
    print(f"Anos cobertos       : {YEARS}")
    print(f"Com solar válido    : {df_out['SOLAR_IRRAD_kwh_m2_dia'].notna().sum()}")
    print(f"Com vento válido    : {df_out['WIND_SPEED_ms'].notna().sum()}")
    print(f"Solar médio (kWh/m²/dia) : {df_out['SOLAR_IRRAD_kwh_m2_dia'].mean():.3f}")
    print(f"Vento médio (m/s)        : {df_out['WIND_SPEED_ms'].mean():.3f}")
    print(f"Dias válidos (total) : {df_out['dias_validos_total'].sum():,}")
    print(f"\nArquivo gerado:")
    print(f"  {out_path}")
    raw_n = len(list(RAW_DIR.glob("*.csv")))
    print(f"  {RAW_DIR}/*.csv  ({raw_n} arquivos brutos)")


if __name__ == "__main__":
    main()
