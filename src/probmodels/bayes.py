from graphviz import Digraph
import numpy as np


def normalize(weights: dict[str, float]) -> dict[str, float]:
    total = sum(weights.values())

    if total == 0:
        raise ValueError("Total probability cannot be zero.")

    return {
        key: value / total
        for key, value in weights.items()
    }


def bayes_update(
    priors: dict[str, float],
    likelihoods: dict[str, float],
) -> dict[str, float]:
    weighted = {
        h: priors[h] * likelihoods[h]
        for h in priors
    }

    return normalize(weighted)


def explain_update(
    priors: dict[str, float],
    likelihoods: dict[str, float],
    population_size: int | None = None,
    evidence_name: str = "Evidence",
    no_evidence_name: str = "No Evidence",
    draw_tree: bool = False,
    rankdir: str = "LR",
    edge_label_style: str = "short",
    show_joint_probabilities: bool = True,
    show_posterior_node: bool = True,
) -> dict:
    if set(priors) != set(likelihoods):
        raise ValueError("Priors and likelihoods must have the same hypotheses.")

    weighted = {
        h: priors[h] * likelihoods[h]
        for h in priors
    }

    evidence_probability = sum(weighted.values())

    posteriors = {
        h: weighted[h] / evidence_probability
        for h in priors
    }

    result = {
        "weighted": weighted,
        "evidence_probability": evidence_probability,
        "posteriors": posteriors,
    }

    if population_size is None:
        return result

    counts = {}

    for h in priors:
        hypothesis_count = round(population_size * priors[h])
        evidence_count = round(hypothesis_count * likelihoods[h])
        no_evidence_count = hypothesis_count - evidence_count

        counts[h] = {
            "hypothesis_count": hypothesis_count,
            "evidence_count": evidence_count,
            "no_evidence_count": no_evidence_count,
        }

    total_evidence_count = sum(
        counts[h]["evidence_count"]
        for h in counts
    )

    result["counts"] = counts
    result["total_evidence_count"] = total_evidence_count

    if draw_tree:
        result["tree"] = make_bayes_tree(
            priors=priors,
            likelihoods=likelihoods,
            counts=counts,
            population_size=population_size,
            evidence_name=evidence_name,
            no_evidence_name=no_evidence_name,
            posteriors=posteriors,
            total_evidence_count=total_evidence_count,
            rankdir=rankdir,
            edge_label_style=edge_label_style,
            show_joint_probabilities=show_joint_probabilities,
            show_posterior_node=show_posterior_node,
        )

    return result


def make_bayes_tree(
    priors: dict[str, float],
    likelihoods: dict[str, float],
    counts: dict,
    population_size: int,
    evidence_name: str,
    no_evidence_name: str,
    posteriors: dict[str, float],
    total_evidence_count: int,
    rankdir: str = "LR",
    edge_label_style: str = "short",  # "short", "full", "none"
    show_joint_probabilities: bool = True,
    show_posterior_node: bool = True,
) -> Digraph:
    dot = Digraph("bayes_tree")

    dot.attr(rankdir=rankdir)
    dot.attr(nodesep="0.6", ranksep="1.2")

    dot.attr(
        "node",
        shape="box",
        style="rounded,filled",
        fontname="Arial",
        fontsize="11",
    )

    dot.attr(
        "edge",
        fontname="Arial",
        fontsize="10",
    )

    def edge_label(probability: float, expression: str) -> str:
        if edge_label_style == "short":
            return f"{probability:.0%}"
        if edge_label_style == "full":
            return f"{expression} = {probability:.2%}"
        if edge_label_style == "none":
            return ""
        raise ValueError("edge_label_style must be 'short', 'full', or 'none'.")

    dot.node(
        "root",
        f"Population\n{population_size:,}",
        fillcolor="#e8f1ff",
    )

    evidence_nodes = []

    for index, hypothesis in enumerate(priors):
        h_id = f"H{index}"
        e_id = f"E{index}"
        ne_id = f"NE{index}"

        c = counts[hypothesis]

        hypothesis_label = f"{hypothesis}\n{c['hypothesis_count']:,}"

        evidence_joint_probability = (
            priors[hypothesis] * likelihoods[hypothesis]
        )

        no_evidence_joint_probability = (
            priors[hypothesis] * (1 - likelihoods[hypothesis])
        )

        if show_joint_probabilities:
            evidence_label = (
                f"{evidence_name}\n"
                f"{c['evidence_count']:,}\n"
                f"P = {evidence_joint_probability:.2%}"
            )

            no_evidence_label = (
                f"{no_evidence_name}\n"
                f"{c['no_evidence_count']:,}\n"
                f"P = {no_evidence_joint_probability:.2%}"
            )
        else:
            evidence_label = f"{evidence_name}\n{c['evidence_count']:,}"
            no_evidence_label = f"{no_evidence_name}\n{c['no_evidence_count']:,}"

        dot.node(
            h_id,
            hypothesis_label,
            fillcolor="#f5f5f5",
        )

        dot.node(
            e_id,
            evidence_label,
            fillcolor="#e7ffe7",
        )

        dot.node(
            ne_id,
            no_evidence_label,
            fillcolor="#fff3f3",
        )

        dot.edge(
            "root",
            h_id,
            label=edge_label(
                priors[hypothesis],
                f"P({hypothesis})",
            ),
        )

        dot.edge(
            h_id,
            e_id,
            label=edge_label(
                likelihoods[hypothesis],
                f"P({evidence_name} | {hypothesis})",
            ),
        )

        dot.edge(
            h_id,
            ne_id,
            label=edge_label(
                1 - likelihoods[hypothesis],
                f"P({no_evidence_name} | {hypothesis})",
            ),
        )

        evidence_nodes.append(e_id)

    if show_posterior_node:
        posterior_lines = [
            f"P({hypothesis} | {evidence_name}) = {posteriors[hypothesis]:.2%}"
            for hypothesis in posteriors
        ]

        dot.node(
            "posterior",
            f"Total {evidence_name}\n{total_evidence_count:,}\n\n"
            + "\n".join(posterior_lines),
            fillcolor="#efe3ff",
        )

        for e_id in evidence_nodes:
            dot.edge(
                e_id,
                "posterior",
                style="dashed",
                label="combine",
            )

    return dot