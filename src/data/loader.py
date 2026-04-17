import os
import pandas as pd
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError, HTTPError

ROOT = Path(__file__).resolve().parent.parent.parent
BRAZIL_COMBINED_FILENAME = "covid_brazil_combined.csv"
BRAZIL_COMBINED_RELEASE_URL = os.environ.get(
    "GNN_COVID_BRAZIL_DATA_URL",
    "https://github.com/Youssef-Malek2004/GNNs-Covid-Mobility/releases/latest/download/covid_brazil_combined.csv",
)


def _download_file(url: str, destination: Path, chunk_size: int = 1024 * 1024) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = destination.with_suffix(destination.suffix + ".part")
    with urlopen(url) as response, tmp_path.open("wb") as out_f:
        while True:
            chunk = response.read(chunk_size)
            if not chunk:
                break
            out_f.write(chunk)
    tmp_path.replace(destination)
    return destination


def ensure_brazil_combined_data(data_dir: str = None) -> Path:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    combined = data_dir / BRAZIL_COMBINED_FILENAME
    if combined.exists():
        return combined

    print(f"[loader] Brazil combined CSV missing. Downloading from release: {BRAZIL_COMBINED_RELEASE_URL}")
    try:
        return _download_file(BRAZIL_COMBINED_RELEASE_URL, combined)
    except (HTTPError, URLError, OSError) as exc:
        raise FileNotFoundError(
            "Brazil combined CSV not found locally and automatic download failed. "
            f"Expected file: {combined}. "
            "Set GNN_COVID_BRAZIL_DATA_URL to a valid release asset URL or place the file manually."
        ) from exc


def load_brazil_covid(data_dir: str = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else ROOT / "data"
    combined = data_dir / BRAZIL_COMBINED_FILENAME
    if combined.exists():
        return pd.read_csv(combined, parse_dates=["date"])

    files = [
        "cases-brazil-cities-time_2020.csv",
        "cases-brazil-cities-time_2021.csv",
        "cases-brazil-cities-time_2022.csv",
        "cases-brazil-cities-time.csv",
    ]
    dfs = [pd.read_csv(data_dir / f) for f in files if (data_dir / f).exists()]
    if not dfs:
        combined = ensure_brazil_combined_data(data_dir)
        return pd.read_csv(combined, parse_dates=["date"])
    df = pd.concat(dfs, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["ibgeID", "date"]).reset_index(drop=True)
    df.to_csv(combined, index=False)
    print(f"[loader] Saved combined Brazil COVID → {combined}")
    return df


def load_spain_covid(data_dir: str = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else ROOT / "data" / "Spain"
    path = data_dir / "provincias_covid19_datos_sanidad_nueva_serie.csv"
    if not path.exists():
        raise FileNotFoundError(f"Spain COVID file not found: {path}")
    df = pd.read_csv(path)
    df["Fecha"] = pd.to_datetime(df["Fecha"])
    df = df.sort_values(["cod_ine", "Fecha"]).reset_index(drop=True)
    return df
