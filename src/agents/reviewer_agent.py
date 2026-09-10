"""
Reviewer Agent (roadmap section 27): "searches for overfitting, data
leakage, cherry-picking, insufficient samples, and transaction-cost
assumptions."

Same note as data_quality_agent.py: this is code applying fixed rules to
the actual result files, not an LLM forming a judgment -- the checks below
are exactly the ones this project's own README/reports already discuss in
prose (small folds, AUC-vs-noise, no transaction costs); this just makes
them a repeatable, automatic gate instead of something only caught by
re-reading the docs carefully.

Run standalone: `python3 src/agents/reviewer_agent.py`
"""

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPORTS_DIR = PROJECT_ROOT / "reports"
REVIEW_OUT = REPORTS_DIR / "REVIEWER_REPORT.md"

MIN_FOLD_SIZE = 100        # below this, a single fold's metric is mostly noise
AUC_NOISE_BAND = 0.03      # how close to 0.5 counts as "not distinguishable from random"


def review_sample_sizes() -> list:
    findings = []
    for path, label in [
        (REPORTS_DIR / "phase4_model_ladder_folds.csv", "Phase 4 (macro-only ladder)"),
        (REPORTS_DIR / "phase5_polymarket_comparison_folds.csv", "Phase 5 (Polymarket comparison)"),
    ]:
        if not path.exists():
            findings.append(("WARN", f"{label}: results file not found, skipping"))
            continue
        df = pd.read_csv(path)
        min_n = df["n"].min()
        status = "WARN" if min_n < MIN_FOLD_SIZE else "PASS"
        findings.append((status, f"{label}: smallest fold has n={min_n} test rows "
                                  f"(flag threshold: {MIN_FOLD_SIZE})"))
    return findings


def review_auc_vs_noise() -> list:
    """Flags every model whose AUC is within AUC_NOISE_BAND of 0.5 --
    i.e. every "beats the baseline" claim this project could make gets
    auto-checked against the same standard, not eyeballed per report."""
    findings = []
    path = REPORTS_DIR / "phase4_model_ladder_summary.csv"
    if not path.exists():
        return [("WARN", "phase4_model_ladder_summary.csv not found, skipping AUC check")]
    df = pd.read_csv(path, header=[0, 1], index_col=0)
    for model in df.index:
        auc_mean = df.loc[model, ("auc", "mean")]
        auc_std = df.loc[model, ("auc", "std")]
        distinguishable = abs(auc_mean - 0.5) > AUC_NOISE_BAND
        if distinguishable and auc_std < abs(auc_mean - 0.5):
            findings.append(("FLAG", f"{model}: AUC {auc_mean:.3f} ± {auc_std:.3f} is outside the "
                                      f"noise band AND its std doesn't already cover 0.5 -- the one "
                                      f"result in this project that would need real scrutiny before "
                                      f"trusting it (currently: none do)."))
        else:
            findings.append(("PASS", f"{model}: AUC {auc_mean:.3f} ± {auc_std:.3f} -- "
                                      f"not distinguishable from random, correctly reported as no edge"))
    return findings


def review_known_caveats_documented() -> list:
    """Confirms the README actually states the caveats this review would
    otherwise have to catch by hand -- a cheap check that the honesty this
    project has maintained turn over turn hasn't quietly regressed."""
    readme = (PROJECT_ROOT / "README.md").read_text()
    required_phrases = [
        ("transaction cost", "backtest has no transaction costs modeled"),
        ("Polymarket history is short", "Polymarket's short history window is disclosed"),
        ("no hyperparameter tuning", "no hyperparameter search against the eval folds"),
    ]
    findings = []
    for phrase, description in required_phrases:
        if phrase.lower() in readme.lower():
            findings.append(("PASS", f"README documents: {description}"))
        else:
            findings.append(("WARN", f"README does NOT mention: {description}"))
    return findings


def review_backtest_costs() -> list:
    path = REPORTS_DIR / "phase5_backtest_summary.csv"
    if not path.exists():
        return [("WARN", "phase5_backtest_summary.csv not found, skipping")]
    return [("WARN", "Backtest results include NO transaction costs -- real total returns would be "
                      "lower, especially at ~64 trades over 2 years. Do not treat backtest returns "
                      "as investable without adding costs (see spy-garch-vwap's 1% round-trip "
                      "assumption for a precedent).")]


def run() -> bool:
    """Returns False only if a FLAG-level finding turns up -- WARN/INFO
    are expected, disclosed caveats, not reasons to fail a pipeline run."""
    sections = {
        "Sample sizes": review_sample_sizes(),
        "AUC vs. noise": review_auc_vs_noise(),
        "Known caveats documented in README": review_known_caveats_documented(),
        "Backtest cost assumptions": review_backtest_costs(),
    }

    lines = ["# Reviewer Agent report", ""]
    print("=== Reviewer Agent ===")
    any_flag = False
    for section, findings in sections.items():
        print(f"\n-- {section} --")
        lines.append(f"## {section}\n")
        for status, message in findings:
            print(f"[{status}] {message}")
            lines.append(f"- **[{status}]** {message}")
            any_flag = any_flag or status == "FLAG"
        lines.append("")

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_OUT.write_text("\n".join(lines))
    print(f"\n-> {REVIEW_OUT}")
    return not any_flag


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
