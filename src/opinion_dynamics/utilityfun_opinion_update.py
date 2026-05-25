"""Opinion update functions: social signal, two-channel mix, Bernoulli step."""

import numpy as np


def compute_social_signal(
    alpha: np.ndarray,
    neighbor_lists: list[np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Return discretized social signal si_soc_i under conformity rule.

    For each agent i a neighbor j is drawn uniformly from N_i.  The signal
    activates with probability equal to the opinion gap |alpha_j - alpha_i|;
    when active its value is sgn(alpha_j - alpha_i) in {-1, +1}:

        With probability |alpha_j - alpha_i|:
            si_soc_i = sgn(alpha_j - alpha_i)
        With probability 1 - |alpha_j - alpha_i|:
            si_soc_i = 0

    Expected value:
        E[si_soc_i | alpha_i, alpha_j] = |alpha_j - alpha_i| * sgn(alpha_j - alpha_i)

    The activation probability grows linearly with the opinion gap; the
    signal amplitude is discrete (+/-1 or 0).  All three channels (private,
    social) operate on the same {-1, 0, +1} scale.

    Properties:
      - si_soc_i = 0 whenever alpha_i = alpha_j (aligned neighbors are silent).
      - Symmetric: swapping alpha -> 1 - alpha for both agents negates the
        signal; no structural bias toward either extreme.
      - No additional free parameters.
      - Agents with no neighbors receive si_soc = 0.

    Parameters
    ----------
    alpha : np.ndarray, shape (N,)
        Current opinion of all agents in [0, 1].
    neighbor_lists : list of np.ndarray
        neighbor_lists[i] contains the position indices of agent i's neighbors.
    rng : np.random.Generator
        Random generator used to draw one neighbor per agent and the
        activation uniform variate.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Social signal si_soc_i in {-1, 0, +1} for each agent.
    """
    N = len(neighbor_lists)

    has_neighbors = np.array([len(nl) > 0 for nl in neighbor_lists], dtype=bool)

    # Draw one random neighbor index per agent; isolated agents map to
    # themselves (their contribution is zeroed out at the end).
    j_indices = np.array(
        [
            int(rng.integers(0, len(neighbor_lists[i]))) if has_neighbors[i] else i
            for i in range(N)
        ],
        dtype=np.intp,
    )
    resolved = np.array(
        [
            neighbor_lists[i][j_indices[i]]
            for i in range(N)
        ],
        dtype=np.intp,
    )

    alpha_j = alpha[resolved]          # opinion of drawn neighbor
    alpha_i = alpha                    # opinion of focal agent

    gap = alpha_j - alpha_i            # signed gap in (-1, 1)
    abs_gap = np.abs(gap)              # activation probability

    # Activation: draw x ~ Uniform[0, 1] per agent; activate if x < |gap|.
    x = rng.uniform(0.0, 1.0, size=N)
    activated = x < abs_gap

    # Signal: discrete +/-1 when activated, 0 otherwise.
    # np.sign returns 0.0 when gap == 0.0, which is correct.
    si_soc = np.where(activated, np.sign(gap), 0.0)
    # si_soc = np.where(activated, (gap), 0.0)


    # Isolated agents contribute no social pressure.
    si_soc[~has_neighbors] = 0.0

    return si_soc


def combine_signals_two_channel(
    s_priv: np.ndarray,
    xi_soc: np.ndarray,
    w_priv: float,
) -> np.ndarray:
    """Return p_i = w_priv * s_priv + (1 - w_priv) * xi_soc, clipped to [-1, +1].

    Flat two-channel mix with one free parameter.  The social-to-private
    weight ratio w_soc / w_priv = (1 - w_priv) / w_priv is constant
    regardless of signal activation — resolving the variable effective
    weight problem of the nested formulation.

    Parameters
    ----------
    s_priv : np.ndarray, shape (N,)
        Unified private signal in {-1, 0, +1}.
    xi_soc : np.ndarray, shape (N,)
        Social signal in {-1, 0, +1} (F1 discretized).
    w_priv : float
        Weight of the private channel, in (0, 1).  w_soc = 1 - w_priv.

    Returns
    -------
    np.ndarray, shape (N,), dtype float64
        Combined signal p_i in [-1, +1].
    """
    w_soc = 1.0 - w_priv
    p = w_priv * s_priv + w_soc * xi_soc
    return np.clip(p, -1.0, 1.0)


def bernoulli_update(
    alpha: np.ndarray,
    p: np.ndarray,
    nu: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one Bernoulli update step and return new alpha."""
    # prob_up = max(p_i, 0), prob_down = max(-p_i, 0)
    prob_up = np.maximum(p, 0.0)
    prob_down = np.maximum(-p, 0.0)
    # draw u_i ~ Uniform(0, 1) for each agent
    u = rng.random(len(alpha))
    # vectorized assignment via np.where
    alpha_new = np.where(
        u < prob_up,
        np.minimum(alpha + nu, 1.0),
        np.where(
            u < prob_up + prob_down,
            np.maximum(alpha - nu, 0.0),
            alpha,
        ),
    )
    return alpha_new.astype(np.float64)