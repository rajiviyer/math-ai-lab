# Math AI Lab

A personal learning lab for studying mathematics and AI concepts through interactive Jupyter notebooks and reusable Python models.

## Concepts

| Concept | Theory | Problems |
|---|---|---|
| Conditional Probability | ✓ | ✓ |
| Bayes' Rule | ✓ | ✓ |
| Markov Processes | — | ✓ |
| Monte Carlo | — | ✓ |
| Probability (PMF / PDF / CDF) | ✓ | ✓ |

Notebooks live under `concepts/<topic>/theory/` and `concepts/<topic>/problems/`.

## Library

`src/probmodels` contains reusable functions used across notebooks:

- `bayes.py` — `bayes_update`, `explain_update`, `make_bayes_tree`, `normalize`
- `conditional.py` — `conditional_probability`, `conditional_probability_from_counts`, `are_independent`
- `simulation.py` — `simulate_bayes`, `simulate_bayes_binary`

## Setup

```bash
uv sync
jupyter lab
```

Requires Python 3.11+.
