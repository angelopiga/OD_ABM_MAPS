"""Social network construction for the ABM.

Pipeline role
-------------
Each simulation a weighted, undirected social network is constructed
from the agent income distribution.

Network formation
-----------------
Edge weights follow an income-assortative kernel
(McPherson, Smith-Lovin & Cook 2001):

    W_ij = exp(−β_income · |s_i − s_j|)

where ``s_i`` is the normalised income score (rank, …).  Log-multiplicative noise
breaks weight ties and adds stochastic variation across simulation runs.

Two construction algorithms are supported (selector ``algorithm``):

* ``"topk"`` — each node selects its ``K`` highest-weight candidates as
  neighbours; reciprocal matches are accepted, unilateral ones with
  probability ``p_unilateral``.  Approximately regular graph.
* ``"sampled"`` — each node draws ``K/2`` neighbours without replacement
  from the row-normalised weight distribution; union symmetrisation.

Optional Watts–Strogatz rewiring (``ws_rewire_p > 0``) rewires each edge
with the given probability to a new endpoint; by default uniform (Watts
& Strogatz 1998).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import networkx as nx
import scipy.sparse as sp

from .parameters import (
    NETWORK_ALGORITHM,
    NETWORK_AVG_DEGREE,
    NETWORK_BETA_INCOME,
    NETWORK_INCOME_NORMALIZATION,
    NETWORK_NOISE_STRENGTH_SD,
    NETWORK_RANK_BINS,
    NETWORK_WS_REWIRE_P,
    NETWORK_WS_TARGET_EXPONENT,
)

# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def normalize_incomes_array(
    incomes: np.ndarray,
    method: str | None = "rank",
    rank_bins: int | None = None,
) -> np.ndarray:
    """Normalize a 1-D array of incomes with the chosen method."""
    x = incomes.astype(float)
    n = x.size
    if n == 0:
        return x.copy()
    if method == "rank":
        order = np.argsort(x)
        ranks = np.empty(n, dtype=float)
        ranks[order] = np.arange(n, dtype=float)
        s = ranks / max(n - 1, 1)
        if (rank_bins is not None) and (rank_bins >= 2):
            s = np.round(s * (rank_bins - 1)) / (rank_bins - 1)
        return s
    else:
        return x.copy()


def build_weight_matrix(
    incomes: np.ndarray,
    beta_income: float,
    norm_method: str,
    rank_bins: int | None,
    noise_strength_sd: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Build the N×N agent weight matrix W.

    Computes

        W_ij = exp(−β_income · |s_i − s_j|)

    where ``s_i`` is the normalised income score.  
    The weight matrix is symmetrised and the diagonal is zeroed.

    Parameters
    ----------
    incomes : np.ndarray
        Raw income values for all N agents.
    beta_income : float
        Income-homophily strength; higher values produce more assortative networks.
    norm_method : str
        Income normalisation method passed to ``normalize_incomes_array``.
    rank_bins : int or None
        Discretisation bins for the ``"rank"`` normalisation method.
    noise_strength_sd : float
        Standard deviation of log-multiplicative noise added to ``W``.
    rng : np.random.Generator
        Random generator for noise draws.

    Returns
    -------
    np.ndarray
        N×N weight matrix with zero diagonal, symmetrised.

    Notes
    -----
    The income-assortative kernel follows Schulz (2022).
    Log-multiplicative noise prevents degenerate weight ties.
    The matrix is always symmetrised as 0.5 * (W + W.T).
    """
    eps = 1e-12

    # Step 1: normalise incomes to a common scale.
    inc_norm = normalize_incomes_array(incomes, method=norm_method, rank_bins=rank_bins)

    # Step 2: pairwise income dissimilarity (absolute difference).
    diff  = inc_norm[:, None] - inc_norm[None, :]
    delta = np.abs(diff)

    # Step 3: income-assortativity weights.
    W = np.exp(-beta_income * delta)

    # Step 4: log-multiplicative noise to break weight ties and add stochasticity.
    if noise_strength_sd > 0.0:
        Z = rng.normal(0.0, noise_strength_sd, size=W.shape)
        np.fill_diagonal(Z, 0.0)
        W = np.exp(np.log(W + eps) + (Z - (noise_strength_sd ** 2) / 2.0))

    # Step 5: zero out self-loops.
    np.fill_diagonal(W, 0.0)

    # Step 6: symmetrisation (always applied).
    W = 0.5 * (W + W.T)

    return W


def _edges_from_W(
    W: np.ndarray,
    avg_degree: int,
    algorithm: str,
    rng: np.random.Generator,
    p_unilateral: float = 0.5,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    """Build an edge list from the weight matrix W using the chosen algorithm.

    Two construction algorithms are supported:

    * ``"topk"`` — each node selects its ``K = avg_degree`` highest-weight
      candidates.  An edge ``(i, j)`` is accepted if mutually selected, or
      with probability ``p_unilateral`` in the unilateral case.
    * ``"sampled"`` — each node draws ``K = floor(avg_degree / 2)``
      neighbours without replacement from the row-normalised weight
      distribution ``p_ij = W_ij / Σ_j W_ij``, then union symmetrisation.

    Edge weight is always the symmetric average ``0.5 · (W_ij + W_ji)``.

    Parameters
    ----------
    W : np.ndarray
        N×N weight matrix with zero diagonal.
    avg_degree : int
        Target mean degree.
    algorithm : {"topk", "sampled"}
        Construction algorithm.
    rng : np.random.Generator
    p_unilateral : float
        Unilateral-acceptance probability for ``"topk"``.

    Returns
    -------
    edges : set of (u, v) tuples with u < v
    weights : dict (u, v) → float
    """
    N = W.shape[0]

    if algorithm == "topk":
        K = min(max(int(avg_degree), 0), max(N - 1, 0))
        if K == 0:
            return set(), {}
        scores = W.copy()
        np.fill_diagonal(scores, -np.inf)
        topk_idx = np.argpartition(scores, -K, axis=1)[:, -K:]
        candidate_sets = [set(row.tolist()) for row in topk_idx]
    elif algorithm == "sampled":
        K = max(int(avg_degree) // 2, 1)
        K = min(K, max(N - 1, 0))
        if K == 0:
            return set(), {}
        W_local = W.copy()
        np.fill_diagonal(W_local, 0.0)
        row_sums = W_local.sum(axis=1)
        candidate_sets = []
        for i in range(N):
            s = row_sums[i]
            if s <= 0.0:
                pool = np.delete(np.arange(N), i)
                chosen = rng.choice(pool, size=min(K, pool.size), replace=False)
            else:
                chosen = rng.choice(N, size=K, replace=False, p=W_local[i] / s)
            candidate_sets.append(set(chosen.tolist()))
    else:
        raise ValueError(f"Unknown algorithm '{algorithm}'.")

    edges: set[tuple[int, int]] = set()
    weights: dict[tuple[int, int], float] = {}
    for i in range(N):
        for j in candidate_sets[i]:
            if i == j:
                continue
            if algorithm == "topk" \
                    and (i not in candidate_sets[j]) \
                    and (rng.random() >= p_unilateral):
                continue
            u, v = (i, j) if i < j else (j, i)
            if (u, v) not in edges:
                edges.add((u, v))
                weights[(u, v)] = 0.5 * (W[u, v] + W[v, u])
    return edges, weights


def _ws_rewire_edges(
    edges: set[tuple[int, int]],
    weights: dict[tuple[int, int], float],
    N: int,
    W: np.ndarray,
    p_rewire: float,
    rng: np.random.Generator,
    target_exponent: float = 0.0,
    target_attribute: np.ndarray | None = None,
    max_attempts: int = 20,
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], float]]:
    """Apply Watts–Strogatz edge rewiring to an existing edge list.

    For each undirected edge ``(u, v)``, with probability ``p_rewire``
    replace ``v`` with a candidate node ``k ∉ {u, N_u}``.  The candidate
    distribution is:

    * ``γ = 0`` (uniform WS, Watts & Strogatz 1998): ``P(k) ∝ 1``.
    * ``γ > 0`` (income-targeted): ``P(k) ∝ target_attribute[k]^γ``,
      concentrating new long-range ties on high-income nodes (hegemonic
      channel).

    No self-loops, no multi-edges.  Edge count is preserved.

    Parameters
    ----------
    edges, weights : edge set and per-edge weights of the starting graph.
    N : int
        Number of nodes.
    W : np.ndarray
        Original weight matrix (used to assign a weight to rewired edges).
    p_rewire : float in [0, 1]
        Per-edge rewiring probability.
    rng : np.random.Generator
    target_exponent : float, default 0.0
        Preferential-attachment exponent γ.
    target_attribute : np.ndarray of shape (N,) or None
        Node-level attribute (typically incomes) used when γ > 0.
    max_attempts : int
        Maximum candidate draws per edge before keeping the original.

    Returns
    -------
    new_edges, new_weights
    """
    if p_rewire <= 0.0 or not edges:
        return edges, weights

    use_targeted = (target_exponent > 0.0)
    if use_targeted:
        if target_attribute is None:
            raise ValueError(
                "target_exponent > 0 requires target_attribute to be provided."
            )
        y = np.asarray(target_attribute, dtype=float)
        if y.shape != (N,):
            raise ValueError(f"target_attribute shape {y.shape} != (N={N},).")
        if not np.all(np.isfinite(y)) or np.any(y <= 0):
            raise ValueError("target_attribute must be strictly positive and finite.")
        log_y = target_exponent * np.log(y)
        log_y -= log_y.max()
        pvec = np.exp(log_y)
        pvec /= pvec.sum()
    else:
        pvec = None

    neigh: list[set[int]] = [set() for _ in range(N)]
    for (u, v) in edges:
        neigh[u].add(v)
        neigh[v].add(u)

    new_edges: set[tuple[int, int]] = set()
    new_weights: dict[tuple[int, int], float] = {}

    for (u, v) in edges:
        if rng.random() >= p_rewire:
            new_edges.add((u, v))
            new_weights[(u, v)] = weights[(u, v)]
            continue

        # Isolation guard: do not leave v with degree 0.
        if len(neigh[v]) <= 1:
            new_edges.add((u, v))
            new_weights[(u, v)] = weights[(u, v)]
            continue

        success = False
        for _ in range(max_attempts):
            if use_targeted:
                k = int(rng.choice(N, p=pvec))
            else:
                k = int(rng.integers(0, N))
            if k == u or k == v or k in neigh[u]:
                continue
            neigh[u].discard(v)
            neigh[v].discard(u)
            neigh[u].add(k)
            neigh[k].add(u)
            a, b = (u, k) if u < k else (k, u)
            new_edges.add((a, b))
            new_weights[(a, b)] = float(W[a, b])
            success = True
            break

        if not success:
            new_edges.add((u, v))
            new_weights[(u, v)] = weights[(u, v)]

    return new_edges, new_weights


# ---------------------------------------------------------------------------
# Public functions — network construction
# ---------------------------------------------------------------------------

def build_social_network(
    df_agents: pd.DataFrame,
    avg_degree: int,
    beta_income: float,
    noise_strength_sd: float,
    income_normalization: str | None,
    rank_bins: int | None,
    algorithm: str,
    ws_rewire_p: float,
    ws_target_exponent: float,
    p_unilateral: float = 0.5,
    rng: np.random.Generator | None = None,
) -> tuple[nx.Graph, sp.csr_matrix, sp.csr_matrix]:
    """Build an income-assortative social network.

    Algorithm
    ---------
    1. Build the N×N weight matrix ``W`` via ``build_weight_matrix``.
    2. Dispatch to the selected construction algorithm via ``_edges_from_W``.
    3. If ``ws_rewire_p > 0``, apply Watts–Strogatz rewiring.
    4. Apply global isolation guard (min_degree ≥ 1).
    5. Return the NetworkX graph, a binary CSR adjacency matrix, and a
       row-normalised weighted CSR adjacency matrix.

    Parameters
    ----------
    df_agents : pd.DataFrame
        Agent DataFrame with columns ``"agent_id"`` and ``"income"``.
    avg_degree : int
        Target average degree.
    beta_income : float
        Income-homophily strength.
    noise_strength_sd : float
        Log-multiplicative noise standard deviation.  Forced to 0 when
        ``algorithm = "sampled"`` (stochasticity is intrinsic to sampling).
    income_normalization : str or None
        Income normalisation method.
    rank_bins : int or None
        Discretisation bins for ``"rank"`` normalisation.
    algorithm : {"topk", "sampled"}
        Edge-formation algorithm.
    ws_rewire_p : float in [0, 1]
        Per-edge WS rewiring probability.  0 disables rewiring.
    ws_target_exponent : float, ≥ 0
        Preferential-attachment exponent γ for rewiring targets.
    p_unilateral : float, default 0.5
        Unilateral-acceptance probability (top-K algorithm only).
    rng : np.random.Generator or None, optional
        Random generator; created from entropy if ``None``.

    Returns
    -------
    G : nx.Graph
        Undirected NetworkX graph with ``agent_id`` as node labels.
    adj : sp.csr_matrix
        Binary adjacency matrix (int8, CSR).
    adj_weighted : sp.csr_matrix
        Row-normalised weighted adjacency matrix (float, CSR).
    """
    if rng is None:
        rng = np.random.default_rng()
    if algorithm not in ("topk", "sampled"):
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choose 'topk' or 'sampled'."
        )
    if not (0.0 <= ws_rewire_p <= 1.0):
        raise ValueError(f"ws_rewire_p must be in [0, 1], got {ws_rewire_p}.")
    if ws_target_exponent < 0.0:
        raise ValueError(
            f"ws_target_exponent must be ≥ 0, got {ws_target_exponent}."
        )

    agent_ids = df_agents["agent_id"].values
    incomes   = df_agents["income"].values.astype(float)
    N = agent_ids.shape[0]

    G = nx.Graph()
    G.add_nodes_from(agent_ids)
    if N < 2:
        adj = sp.csr_matrix((N, N), dtype=np.int8)
        adj_weighted = sp.csr_matrix((N, N), dtype=float)
        return G, adj, adj_weighted

    # Noise is disabled for "sampled" (stochasticity is intrinsic).
    effective_noise = noise_strength_sd if algorithm == "topk" else 0.0
    W = build_weight_matrix(
        incomes=incomes,
        beta_income=beta_income,
        norm_method=income_normalization,
        rank_bins=rank_bins,
        noise_strength_sd=effective_noise,
        rng=rng,
    )

    edges, weights = _edges_from_W(
        W=W,
        avg_degree=avg_degree,
        algorithm=algorithm,
        rng=rng,
        p_unilateral=p_unilateral,
    )

    if ws_rewire_p > 0.0:
        edges, weights = _ws_rewire_edges(
            edges=edges,
            weights=weights,
            N=N,
            W=W,
            p_rewire=ws_rewire_p,
            rng=rng,
            target_exponent=ws_target_exponent,
            target_attribute=incomes if ws_target_exponent > 0.0 else None,
        )

    # Global isolation guard: ensure min_degree ≥ 1.
    has_neighbour = np.zeros(N, dtype=bool)
    for (u, v) in edges:
        has_neighbour[u] = True
        has_neighbour[v] = True
    isolated_nodes = np.where(~has_neighbour)[0]
    if isolated_nodes.size > 0:
        neigh_sets: list[set[int]] = [set() for _ in range(N)]
        for (u, v) in edges:
            neigh_sets[u].add(v)
            neigh_sets[v].add(u)
        for i in isolated_nodes:
            i = int(i)
            scores = W[i].copy()
            scores[i] = -np.inf
            for nb in neigh_sets[i]:
                scores[nb] = -np.inf
            j = int(np.argmax(scores))
            if not np.isfinite(scores[j]):
                continue
            u, v = (i, j) if i < j else (j, i)
            if (u, v) not in edges:
                edges.add((u, v))
                weights[(u, v)] = 0.5 * (W[u, v] + W[v, u])
                neigh_sets[u].add(v)
                neigh_sets[v].add(u)

    # Build graph and sparse adjacency matrices from the edge set.
    if edges:
        G.add_edges_from((agent_ids[u], agent_ids[v]) for (u, v) in edges)

        iu = np.fromiter((u for u, _ in edges), dtype=np.int32)
        ju = np.fromiter((v for _, v in edges), dtype=np.int32)
        row = np.concatenate([iu, ju])
        col = np.concatenate([ju, iu])
        data_bin = np.ones(row.size, dtype=np.int8)
        adj = sp.coo_matrix((data_bin, (row, col)), shape=(N, N)).tocsr()

        row = np.array([u for (u, v) in weights] + [v for (u, v) in weights], dtype=np.int32)
        col = np.array([v for (u, v) in weights] + [u for (u, v) in weights], dtype=np.int32)
        data = np.array([w for w in weights.values()] * 2, dtype=float)
        adj_weighted = sp.coo_matrix((data, (row, col)), shape=(N, N)).tocsr()
        row_sums = np.array(adj_weighted.sum(axis=1)).ravel()
        row_sums[row_sums == 0] = 1.0
        D_inv = sp.diags(1.0 / row_sums)
        adj_weighted = D_inv @ adj_weighted
    else:
        adj = sp.csr_matrix((N, N), dtype=np.int8)
        adj_weighted = sp.csr_matrix((N, N), dtype=float)

    return G, adj, adj_weighted


def build_network(
    df_agents: pd.DataFrame,
    Y: int,
    *,
    avg_degree: int = NETWORK_AVG_DEGREE,
    beta_income: float = NETWORK_BETA_INCOME,
    income_normalization: str = NETWORK_INCOME_NORMALIZATION,
    rank_bins: int | None = NETWORK_RANK_BINS,
    noise_strength_sd: float = NETWORK_NOISE_STRENGTH_SD,
    algorithm: str = NETWORK_ALGORITHM,
    ws_rewire_p: float = NETWORK_WS_REWIRE_P,
    ws_target_exponent: float = NETWORK_WS_TARGET_EXPONENT,
    rng: np.random.Generator,
) -> tuple[nx.Graph, sp.csr_matrix, sp.csr_matrix]:
    """Build the social network for year ``Y``.

    Thin wrapper around ``build_social_network`` that stores ``"income"`` as
    a node attribute on the returned graph.

    Parameters
    ----------
    df_agents : pd.DataFrame
        Agent DataFrame for year ``Y``; columns ``"agent_id"``, ``"income"``.
    Y : int
        Simulation year.
    avg_degree : int, optional
        Target average degree.
    beta_income : float, optional
        Income-homophily strength.
    income_normalization : str, optional
        Income normalisation method.
    rank_bins : int or None, optional
        Rank discretisation bins.
    noise_strength_sd : float, optional
        Log-multiplicative noise standard deviation.
    algorithm : {"topk", "sampled"}, optional
        Edge-formation algorithm.
    ws_rewire_p : float in [0, 1], optional
        Per-edge Watts–Strogatz rewiring probability (0 disables).
    ws_target_exponent : float, ≥ 0, optional
        Preferential-attachment exponent γ for rewiring targets.
    rng : np.random.Generator
        Random generator.

    Returns
    -------
    G : nx.Graph
        Social network with ``"income"`` as node attribute.
    adj : sp.csr_matrix
        Binary adjacency matrix (CSR).
    adj_weighted : sp.csr_matrix
        Row-normalised weighted adjacency matrix (CSR).
    """
    

    G, A, W = build_social_network(
        df_agents=df_agents,
        avg_degree=avg_degree,
        beta_income=beta_income,
        income_normalization=income_normalization,
        rank_bins=rank_bins,
        noise_strength_sd=noise_strength_sd,
        algorithm=algorithm,
        ws_rewire_p=ws_rewire_p,
        ws_target_exponent=ws_target_exponent,
        rng=rng,
    )

    nx.set_node_attributes(G, df_agents.set_index("agent_id")["income"].to_dict(), "income")
    return G, A, W
