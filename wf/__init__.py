"""Latch workflow: predict a two-gene CRISPRa combination from measured single-gene responses.

Wraps predict.py and the released neural heads in results/. No training happens here.
"""
import os
import sys
from pathlib import Path

from latch import small_task, workflow
from latch.resources.launch_plan import LaunchPlan
from latch.types import LatchAuthor, LatchFile, LatchMetadata, LatchParameter

REPO = Path(__file__).resolve().parent.parent

metadata = LatchMetadata(
    display_name="Perturbation Composition: predict a two-gene combination",
    author=LatchAuthor(name="Aarnav Bhat", github="https://github.com/aarnavbhat23"),
    repository="https://github.com/aarnavbhat23/perturbation-composition-lab",
    license="MIT",
    parameters={
        "first_gene": LatchParameter(
            display_name="First gene",
            description="A gene with a measured single-gene response in the Norman CRISPRa reference.",
        ),
        "second_gene": LatchParameter(
            display_name="Second gene",
            description="A different gene with a measured single-gene response.",
        ),
    },
)


@small_task
def predict_task(first_gene: str, second_gene: str) -> LatchFile:
    os.chdir(REPO)  # predict.py loads results/ with relative paths
    sys.path.insert(0, str(REPO))
    from predict import predict

    name = f"{first_gene}+{second_gene}-prediction.csv"
    local = REPO / name
    predict(first_gene, second_gene).to_csv(local, index=False)
    return LatchFile(str(local), f"latch:///perturbation-composition/{name}")


@workflow(metadata)
def perturbation_composition(first_gene: str = "CEBPA", second_gene: str = "KLF1") -> LatchFile:
    """Predict the expression change from switching on two genes together.

    Uses the measured response to each gene alone (Norman et al. 2019 CRISPRa data),
    adds them, and applies a small released neural model that corrects for the
    interaction. Output is a CSV with one row per gene: `additive_delta` (simple sum)
    and `neural_delta` (sum plus learned interaction), on the upstream processed-expression scale.

    Only genes with a measured single-gene reference are supported. This does not
    predict new genes, new cell types, or absolute expression.
    See the [repository](https://github.com/aarnavbhat23/perturbation-composition-lab) for the full audit.
    """
    return predict_task(first_gene=first_gene, second_gene=second_gene)


LaunchPlan(
    perturbation_composition,
    "CEBPA + KLF1 (held-out test pair)",
    {"first_gene": "CEBPA", "second_gene": "KLF1"},
)
