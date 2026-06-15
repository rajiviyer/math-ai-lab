import numpy as np
import pandas as pd


def simulate_bayes_binary(
    n,
    p_h,
    p_e_given_h,
    p_e_given_not_h,
    hypothesis_name="Hypothesis",
    evidence_name="Evidence",
    seed=None,
):
    rng = np.random.default_rng(seed)

    # Sample H
    h = rng.binomial(1, p_h, n).astype(bool)

    # Sample E | H
    p_e = np.where(
        h,
        p_e_given_h,
        p_e_given_not_h,
    )

    e = rng.binomial(1, p_e).astype(bool)

    # Counts
    h_and_e = np.sum(h & e)
    h_and_not_e = np.sum(h & ~e)
    not_h_and_e = np.sum(~h & e)
    not_h_and_not_e = np.sum(~h & ~e)

    table = pd.DataFrame(
        {
            hypothesis_name: [
                h_and_e,
                h_and_not_e,
                h.sum(),
            ],
            f"Not {hypothesis_name}": [
                not_h_and_e,
                not_h_and_not_e,
                (~h).sum(),
            ],
            "Total": [
                e.sum(),
                (~e).sum(),
                n,
            ],
        },
        index=[
            evidence_name,
            f"Not {evidence_name}",
            "Total",
        ],
    )

    return {
        "hypothesis": h,
        "evidence": e,
        "table": table,
    }

import numpy as np
import pandas as pd


def simulate_bayes(
    priors: dict[str, float],
    likelihoods: dict[str, float],
    n: int = 100_000,
    hypothesis_name: str = "Hypothesis",
    evidence_name: str = "Evidence",
    seed: int | None = None,
) -> dict:
    if set(priors) != set(likelihoods):
        raise ValueError("Priors and likelihoods must have the same hypotheses.")

    if not np.isclose(sum(priors.values()), 1.0):
        raise ValueError("Priors must sum to 1.")

    states = list(priors.keys())
    prior_probs = list(priors.values())

    rng = np.random.default_rng(seed)

    hypotheses = rng.choice(
        states,
        size=n,
        p=prior_probs,
    )

    evidence_probabilities = np.array([
        likelihoods[h]
        for h in hypotheses
    ])

    evidence = rng.random(n) < evidence_probabilities

    df = pd.DataFrame(
        {
            hypothesis_name: hypotheses,
            evidence_name: evidence,
        }
    )

    table = pd.crosstab(
        df[evidence_name],
        df[hypothesis_name],
        margins=True,
    )

    posterior = (
        df[df[evidence_name]]
        [hypothesis_name]
        .value_counts(normalize=True)
        .reindex(states, fill_value=0)
    )

    theoretical_evidence_probability = sum(
        priors[h] * likelihoods[h]
        for h in states
    )

    posterior_exact = {
        h: (
            priors[h]
            * likelihoods[h]
            / theoretical_evidence_probability
        )
        for h in states
    }

    posterior_explanations = {
        h: (
            f"P({h} | {evidence_name}) = "
            f"{posterior_exact[h]:.4f}"
        )
        for h in states
    }

    return {
        "data": df,
        "table": table,
        "posterior_simulated": posterior,
        "posterior_exact": posterior_exact,
        "posterior_explanations": posterior_explanations,
        "evidence_probability_simulated": float(evidence.mean()),
        "evidence_probability_exact": float(theoretical_evidence_probability),
    }