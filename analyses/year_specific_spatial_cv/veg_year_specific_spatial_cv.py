"""Year-specific evaluation of the primary vegetation spatial-CV OOF predictions.

This diagnostic does not refit any model. It re-scores the archived out-of-fold
predictions separately within each year, then reports unweighted macro-averages
across 2018-2024.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "results" / "oof_predictions_veg_spatial.csv"
DEFAULT_OUTPUT = ROOT / "results" / "veg_year_specific_spatial_cv.csv"

PREDICTION_COLUMNS = {
    "Stack": "pred_Stack",
    "AEF": "pred_AEF",
    "AEF+Stack": "pred_AEF+Stack",
}


def evaluate(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    required = {"year", "y_true", *PREDICTION_COLUMNS.values()}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    rows: list[dict[str, object]] = []
    for year, group in df.groupby("year", sort=True):
        y_true = group["y_true"].astype(int)
        if y_true.nunique() < 2:
            raise ValueError(f"Year {year} contains only one outcome class; ROC-AUC is undefined.")

        row: dict[str, object] = {
            "year": int(year),
            "n": int(len(group)),
            "positive_prevalence": float(y_true.mean()),
        }
        for label, column in PREDICTION_COLUMNS.items():
            row[f"{label}_ROC_AUC"] = roc_auc_score(y_true, group[column])
            row[f"{label}_PR_AUC"] = average_precision_score(y_true, group[column])
        rows.append(row)

    out = pd.DataFrame(rows)
    metric_columns = [
        "Stack_ROC_AUC",
        "AEF_ROC_AUC",
        "AEF+Stack_ROC_AUC",
        "Stack_PR_AUC",
        "AEF_PR_AUC",
        "AEF+Stack_PR_AUC",
    ]
    column_order = ["year", "n", "positive_prevalence", *metric_columns]
    out = out[column_order]

    macro = {column: float(out[column].mean()) for column in metric_columns}
    macro_row: dict[str, object] = {
        "year": "Macro-average",
        "n": None,
        "positive_prevalence": None,
        **macro,
    }
    return pd.DataFrame([*out.to_dict("records"), macro_row], columns=column_order)


def write_csv(result: pd.DataFrame, output_path: Path) -> None:
    columns = list(result.columns)
    lines = [",".join(columns)]
    for _, row in result.iterrows():
        year = str(row["year"])
        if year == "Macro-average":
            n = ""
            prevalence = ""
        else:
            n = str(int(row["n"]))
            prevalence = f"{float(row['positive_prevalence']):.3f}"
        metrics = [f"{float(row[column]):.3f}" for column in columns[3:]]
        lines.append(",".join([year, n, prevalence, *metrics]))
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    result = evaluate(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_csv(result, args.output)
    print(result.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nWrote {args.output}")


if __name__ == "__main__":
    main()
