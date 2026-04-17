import os
import numpy as np
import pandas as pd
import networkx as nx
from collections import defaultdict
from pathlib import Path
from typing import Optional, Set, Tuple, List
from sklearn.preprocessing import MinMaxScaler
from torch_geometric.utils import from_networkx
from torch_geometric.data import Data

ROOT = Path(__file__).resolve().parent.parent.parent


# ─── Brazil ───────────────────────────────────────────────────────────────────

def top_k_brazil_cities(k: int, candidate_ids: Optional[Set[int]] = None) -> Set[int]:
    """Return the top-k Brazilian city IDs by population from cleaned_population_2022.csv."""
    pop_df = pd.read_csv(ROOT / "data" / "cleaned_population_2022.csv")
    if candidate_ids:
        pop_df = pop_df[pop_df["ibgeID"].isin(candidate_ids)]
    top = pop_df.nlargest(k, "population")["ibgeID"].astype(int)
    return set(top.tolist())


def extract_brazil_backbone(
    centrality_path: Optional[str] = None,
    mobility_path: Optional[str] = None,
    alpha: float = 0.01,
    top_k: int = 5,
    cache_path: Optional[str] = None,
    city_whitelist: Optional[Set[int]] = None,
) -> Tuple[pd.DataFrame, Set[int]]:
    centrality_path = centrality_path or str(ROOT / "data" / "Centrality_indices.xlsx")
    mobility_path   = mobility_path   or str(ROOT / "data" / "Road_and_waterway_connections_database_2016.xlsx")

    # Use a whitelist-aware cache path so different city subsets don't collide
    if cache_path is None:
        n_cities = len(city_whitelist) if city_whitelist else "all"
        cache_path = str(ROOT / "data" / f"mobility_backbone_brazil_{n_cities}.csv")

    if os.path.exists(cache_path):
        df = pd.read_csv(cache_path)
        cities = set(df["source"].astype(int)).union(df["target"].astype(int))
        print(f"[graph] Loaded cached Brazil backbone ({cache_path}): {len(df)} edges, {len(cities)} nodes")
        return df, cities

    centrality_df = pd.read_excel(centrality_path)
    valid_nodes = set(centrality_df["Codmundv"].dropna().astype(int).unique())
    if city_whitelist:
        valid_nodes &= city_whitelist

    edges_df = pd.read_excel(mobility_path)
    edges_df = edges_df.rename(columns={
        "CODMUNDV_A": "source",
        "CODMUNDV_B": "target",
        "VAR05": "weekly_flow",
    })
    edges_df = edges_df[
        edges_df["source"].isin(valid_nodes) & edges_df["target"].isin(valid_nodes)
    ].copy()

    backbone_df, cities = _disparity_backbone(edges_df, "weekly_flow", alpha, top_k)
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    backbone_df.to_csv(cache_path, index=False)
    print(f"[graph] Brazil backbone: {len(backbone_df)} edges → {cache_path}")
    return backbone_df, cities


def build_brazil_graph(
    backbone_df: pd.DataFrame,
    centrality_path: Optional[str] = None,
) -> Tuple[Data, List[int]]:
    G = nx.Graph()
    backbone_cities = sorted(
        set(backbone_df["source"].astype(int)).union(backbone_df["target"].astype(int))
    )
    for city_id in backbone_cities:
        G.add_node(city_id)
    for _, row in backbone_df.iterrows():
        s, t = int(row["source"]), int(row["target"])
        if s != t and not G.has_edge(s, t):
            G.add_edge(s, t, edge_weight=float(row["weekly_flow"]))

    pyg = from_networkx(G, group_edge_attrs=["edge_weight"])
    node_order = sorted(G.nodes())
    print(f"[graph] Brazil: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return pyg, node_order


# ─── Spain ────────────────────────────────────────────────────────────────────

def load_spain_mobility(
    mobility_path: Optional[str] = None,
    centrality_path: Optional[str] = None,
    cache_path: Optional[str] = None,
) -> pd.DataFrame:
    mobility_path   = mobility_path   or str(ROOT / "data" / "Spain" / "Spanish-Mobility-Raw.xlsx")
    centrality_path = centrality_path or str(ROOT / "data" / "Spain" / "centrality_provinces.csv")
    cache_path      = cache_path      or str(ROOT / "data" / "Spain" / "mobility_cleaned_final.csv")

    if os.path.exists(cache_path):
        return pd.read_csv(cache_path)

    rename_map = {"Illes Balears": "Balears", "Valencia": "Valencia/València"}
    invalid = {"FR", "PT", "ex"}

    centrality_df = pd.read_csv(centrality_path)
    valid_cities = set(centrality_df["nomemun"])

    df = pd.read_excel(mobility_path, sheet_name="Data", skiprows=2)
    df = df.dropna(subset=["COD. PROV. ORIGEN", "COD. PROV. DESTINO", "VIAJES"])
    df = df[df["COD. PROV. ORIGEN"] != df["COD. PROV. DESTINO"]]
    df["VIAJES"] = pd.to_numeric(df["VIAJES"], errors="coerce")
    df = df.dropna(subset=["VIAJES"])
    df["PROVINCIA ORIGEN"]  = df["PROVINCIA ORIGEN"].map(lambda x: rename_map.get(x, x))
    df["PROVINCIA DESTINO"] = df["PROVINCIA DESTINO"].map(lambda x: rename_map.get(x, x))
    df = df[
        (~df["PROVINCIA ORIGEN"].isin(invalid)) &
        (~df["PROVINCIA DESTINO"].isin(invalid)) &
        (df["PROVINCIA ORIGEN"].isin(valid_cities)) &
        (df["PROVINCIA DESTINO"].isin(valid_cities))
    ]
    scaler = MinMaxScaler()
    df["weight"] = scaler.fit_transform(df[["VIAJES"]])
    df = df.rename(columns={"COD. PROV. ORIGEN": "origin", "COD. PROV. DESTINO": "destination"})
    df = df[["origin", "destination", "weight", "PROVINCIA ORIGEN", "PROVINCIA DESTINO"]].copy()
    df.to_csv(cache_path, index=False)
    print(f"[graph] Saved cleaned Spain mobility → {cache_path}")
    return df


def extract_spain_backbone(
    mobility_df: pd.DataFrame,
    cities: List[str],
    alpha: float = 0.01,
    top_k: int = 5,
) -> pd.DataFrame:
    avg_matrix = pd.DataFrame(0.0, index=cities, columns=cities)
    for _, row in mobility_df.iterrows():
        i, j = row["PROVINCIA ORIGEN"], row["PROVINCIA DESTINO"]
        if i in avg_matrix.index and j in avg_matrix.columns:
            avg_matrix.at[i, j] = row["weight"]

    degree   = defaultdict(int)
    strength = defaultdict(float)
    edges = []
    for src in cities:
        for dst in cities:
            if src != dst:
                w = avg_matrix.at[src, dst]
                if w > 0:
                    edges.append((src, dst, w))
                    degree[src] += 1
                    strength[src] += w

    records = []
    for src, dst, w in edges:
        si, ki = strength[src], degree[src]
        sj, kj = strength[dst], degree[dst]
        pij_i = (1 - w / si) ** (ki - 1) if ki > 1 and si > 0 else 1.0
        pij_j = (1 - w / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1.0
        records.append({"source": src, "target": dst, "weight": w, "pij": min(pij_i, pij_j)})

    df = pd.DataFrame(records)
    if df.empty:
        return df

    df["edge_key"] = list(zip(df["source"], df["target"]))
    topk = defaultdict(set)
    for node in cities:
        node_edges = df[df["source"] == node].nlargest(top_k, "weight")
        topk[node].update(zip(node_edges["source"], node_edges["target"]))
    df["keep"] = (df["pij"] < alpha) | df["edge_key"].apply(lambda x: x in topk[x[0]])
    backbone = df[df["keep"]].copy()
    print(f"[graph] Spain backbone: {len(backbone)} edges retained")
    return backbone


def build_spain_graph(
    backbone_df: pd.DataFrame,
    centrality_path: Optional[str] = None,
) -> Tuple[Data, List[str]]:
    centrality_path = centrality_path or str(ROOT / "data" / "Spain" / "centrality_provinces.csv")
    centrality_df = pd.read_csv(centrality_path)
    valid_names = set(centrality_df["nomemun"])

    G = nx.Graph()
    for name in valid_names:
        G.add_node(name)

    for _, row in backbone_df.iterrows():
        src, tgt = row["source"], row["target"]
        if src in valid_names and tgt in valid_names and src != tgt and not G.has_edge(src, tgt):
            G.add_edge(src, tgt, edge_weight=float(row["weight"]))

    pyg = from_networkx(G, group_edge_attrs=["edge_weight"])
    node_order = sorted(G.nodes())
    print(f"[graph] Spain: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
    return pyg, node_order


# ─── Shared ───────────────────────────────────────────────────────────────────

def _disparity_backbone(
    edges_df: pd.DataFrame,
    weight_col: str,
    alpha: float,
    top_k: int,
) -> Tuple[pd.DataFrame, Set[int]]:
    degree   = defaultdict(int)
    strength = defaultdict(float)
    for _, row in edges_df.iterrows():
        i, j = int(row["source"]), int(row["target"])
        w = float(row[weight_col])
        degree[i] += 1;  degree[j] += 1
        strength[i] += w; strength[j] += w

    pij_list = []
    for _, row in edges_df.iterrows():
        i, j   = int(row["source"]), int(row["target"])
        Aij    = float(row[weight_col])
        si, ki = strength[i], degree[i]
        sj, kj = strength[j], degree[j]
        pij_i  = (1 - Aij / si) ** (ki - 1) if ki > 1 and si > 0 else 1.0
        pij_j  = (1 - Aij / sj) ** (kj - 1) if kj > 1 and sj > 0 else 1.0
        pij_list.append(min(pij_i, pij_j))

    edges_df = edges_df.copy()
    edges_df["pij"]      = pij_list
    edges_df["edge_key"] = list(zip(edges_df["source"].astype(int), edges_df["target"].astype(int)))

    all_nodes = set(edges_df["source"].astype(int)).union(edges_df["target"].astype(int))
    topk_neighbors = defaultdict(set)
    for node in all_nodes:
        node_edges = edges_df[
            (edges_df["source"] == node) | (edges_df["target"] == node)
        ].nlargest(top_k, weight_col)
        for _, r in node_edges.iterrows():
            topk_neighbors[node].add((int(r["source"]), int(r["target"])))

    edges_df["keep"] = (edges_df["pij"] < alpha) | edges_df["edge_key"].apply(
        lambda x: x in topk_neighbors[x[0]] or x in topk_neighbors[x[1]]
    )
    backbone = edges_df[edges_df["keep"]].copy()
    cities   = set(backbone["source"].astype(int)).union(backbone["target"].astype(int))
    return backbone, cities
