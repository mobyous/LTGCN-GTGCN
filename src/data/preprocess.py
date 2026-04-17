import pandas as pd
from pathlib import Path
from typing import Optional, Set

ROOT = Path(__file__).resolve().parent.parent.parent


def filter_brazil_covid(
    covid_df: pd.DataFrame,
    centrality_path: Optional[str] = None,
    population_path: Optional[str] = None,
    city_whitelist: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Filter Brazil COVID data to centrality cities and clip negatives.
    Does NOT normalize — normalization happens after train/val/test split.
    """
    centrality_path = centrality_path or str(ROOT / "data" / "Centrality_indices.xlsx")
    population_path = population_path or str(ROOT / "data" / "cleaned_population_2022.csv")

    centrality_df = pd.read_excel(centrality_path)
    valid_ids = set(centrality_df["Codmundv"].dropna().astype(int).unique())
    if city_whitelist:
        valid_ids &= city_whitelist

    df = covid_df[covid_df["ibgeID"].isin(valid_ids)].copy()
    df["newCases"]  = df["newCases"].clip(lower=0)
    df["newDeaths"] = df["newDeaths"].clip(lower=0)

    pop_df = pd.read_csv(population_path)
    df = df.merge(pop_df[["ibgeID", "population"]], on="ibgeID", how="left")
    df["cases_per_100k"]  = (df["newCases"]  / df["population"]) * 1e5
    df["deaths_per_100k"] = (df["newDeaths"] / df["population"]) * 1e5

    print(f"[preprocess] Brazil: {df['ibgeID'].nunique()} cities, {len(df):,} rows")
    return df


def filter_spain_covid(
    covid_df: pd.DataFrame,
    centrality_path: Optional[str] = None,
    city_whitelist: Optional[Set[int]] = None,
) -> pd.DataFrame:
    """
    Filter Spain COVID data to centrality provinces and clip negatives.
    Does NOT normalize.
    """
    centrality_path = centrality_path or str(ROOT / "data" / "Spain" / "centrality_provinces.csv")
    centrality_df = pd.read_csv(centrality_path)
    valid_ids = set(centrality_df["Codmundv"].dropna().astype(int).unique())
    if city_whitelist:
        valid_ids &= city_whitelist

    df = covid_df[covid_df["cod_ine"].isin(valid_ids)].copy()
    df["Casos"] = df["Casos"].clip(lower=0)

    print(f"[preprocess] Spain: {df['cod_ine'].nunique()} provinces, {len(df):,} rows")
    return df


def drop_constant_nodes_brazil(df: pd.DataFrame, min_std: float = 1e-6) -> pd.DataFrame:
    """
    Remove cities whose newCases series has near-zero variance.
    These cities produce degenerate z-scores and inflate metrics.
    """
    stds = df.groupby("ibgeID")["newCases"].std(ddof=0)
    valid = stds[stds > min_std].index
    dropped = len(stds) - len(valid)
    if dropped:
        print(f"[preprocess] Dropped {dropped} constant Brazil cities")
    return df[df["ibgeID"].isin(valid)].copy()


def drop_constant_nodes_spain(df: pd.DataFrame, min_std: float = 1e-6) -> pd.DataFrame:
    stds = df.groupby("cod_ine")["Casos"].std(ddof=0)
    valid = stds[stds > min_std].index
    dropped = len(stds) - len(valid)
    if dropped:
        print(f"[preprocess] Dropped {dropped} constant Spain provinces")
    return df[df["cod_ine"].isin(valid)].copy()
