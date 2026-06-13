"""
Описательные статистики RecruitView + средние по полу.

Источник пола: recruitview_full.csv (все 2011 строк, as-is).
Кодировка пола: 0 = женщина, 1 = мужчина.

Запуск (из любой папки):
    python src/descriptive_stats.py
Результат: results/tables/descriptive_statistics.txt
"""

import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "recruitview_full.csv"
OUT_FILE = ROOT / "outputs" / "stage1" / "descriptive_statistics.txt"

# Кодировка пола (по решению от 2026-06-09): 0 = женщина, 1 = мужчина
GENDER_LABELS = {0: "Женщины", 1: "Мужчины"}

BIG_FIVE = [
    "openness",
    "conscientiousness",
    "extraversion",
    "agreeableness",
    "neuroticism",
    "overall_personality",
]
PERFORMANCE = [
    "interview_score",
    "answer_score",
    "speaking_skills",
    "confidence_score",
    "facial_expression",
    "overall_performance",
]
CONTINUOUS = BIG_FIVE + PERFORMANCE

SEP = "=" * 80


def describe_series(s: pd.Series) -> dict:
    """Сводка по одной непрерывной переменной."""
    s = pd.to_numeric(s, errors="coerce")
    valid = s.dropna()
    return {
        "n": int(valid.shape[0]),
        "mean": valid.mean(),
        "sd": valid.std(),
        "min": valid.min(),
        "q1": valid.quantile(0.25),
        "median": valid.median(),
        "q3": valid.quantile(0.75),
        "max": valid.max(),
        "missing": int(s.isna().sum()),
    }


def stats_table(df: pd.DataFrame, cols: list, labels: dict) -> list:
    """Таблица describe для набора колонок -> список строк отчёта."""
    header = (
        f"{'Variable':<22}{'N':>6}{'Mean':>8}{'SD':>8}{'Min':>9}"
        f"{'Q1':>8}{'Median':>8}{'Q3':>8}{'Max':>8}{'Missing':>9}"
    )
    lines = [header, "-" * len(header)]
    for col in cols:
        d = describe_series(df[col])
        lines.append(
            f"{labels[col]:<22}{d['n']:>6,}{d['mean']:>8.3f}{d['sd']:>8.3f}"
            f"{d['min']:>9.3f}{d['q1']:>8.3f}{d['median']:>8.3f}"
            f"{d['q3']:>8.3f}{d['max']:>8.3f}{d['missing']:>9,}"
        )
    return lines


def cohens_d(a: pd.Series, b: pd.Series) -> float:
    """Cohen's d (pooled SD) для разницы (mean_a - mean_b)."""
    a, b = a.dropna(), b.dropna()
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan")
    pooled = np.sqrt(
        ((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2)
    )
    if pooled == 0:
        return float("nan")
    return (a.mean() - b.mean()) / pooled


def gender_means_table(df: pd.DataFrame, cols: list, labels: dict) -> list:
    """Средние по полу: Ж и М рядом, разница (М−Ж) и Cohen's d."""
    fem = df[df["gender"] == 0]
    mal = df[df["gender"] == 1]
    n_fem, n_mal = len(fem), len(mal)

    header = (
        f"{'Variable':<22}"
        f"{'Ж Mean':>9}{'Ж SD':>8}"
        f"{'М Mean':>9}{'М SD':>8}"
        f"{'Δ(М−Ж)':>9}{'Cohen d':>9}"
    )
    lines = [
        f"Женщины (gender=0): n = {n_fem:,}    Мужчины (gender=1): n = {n_mal:,}",
        "Δ(М−Ж) = среднее у мужчин минус среднее у женщин (в единицах z-score)",
        "Cohen d > 0 => у мужчин выше; |d| 0.2 малый / 0.5 средний / 0.8 большой эффект",
        "",
        header,
        "-" * len(header),
    ]
    for col in cols:
        mf = pd.to_numeric(fem[col], errors="coerce")
        mm = pd.to_numeric(mal[col], errors="coerce")
        diff = mm.mean() - mf.mean()
        d = cohens_d(mm, mf)
        lines.append(
            f"{labels[col]:<22}"
            f"{mf.mean():>9.3f}{mf.std():>8.3f}"
            f"{mm.mean():>9.3f}{mm.std():>8.3f}"
            f"{diff:>9.3f}{d:>9.3f}"
        )
    return lines


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    label_map = {
        "openness": "Openness",
        "conscientiousness": "Conscientiousness",
        "extraversion": "Extraversion",
        "agreeableness": "Agreeableness",
        "neuroticism": "Neuroticism",
        "overall_personality": "Overall Personality",
        "interview_score": "Interview Score",
        "answer_score": "Answer Score",
        "speaking_skills": "Speaking Skills",
        "confidence_score": "Confidence Score",
        "facial_expression": "Facial Expression",
        "overall_performance": "Overall Performance",
    }

    out = []
    out += [SEP, "ОПИСАТЕЛЬНЫЕ СТАТИСТИКИ — RecruitView Dataset", SEP, ""]
    out += [f"Дата: {dt.date.today().isoformat()}",
            f"Файл: {DATA_FILE.relative_to(ROOT)}", ""]
    out += [f"Всего наблюдений: {len(df):,}",
            f"Уникальных участников: {df['user_no'].nunique()}", ""]

    # 1. Демография
    out += [SEP, "1. ДЕМОГРАФИЯ (gender: 0 = женщина, 1 = мужчина)", SEP, ""]
    n_total = len(df)
    vc = df["gender"].value_counts(dropna=False)
    missing = int(df["gender"].isna().sum())
    out += ["По ответам:"]
    for code in (0, 1):
        cnt = int(vc.get(code, 0))
        out.append(f"  {code} ({GENDER_LABELS[code]}): {cnt:>5,} ({cnt / n_total * 100:5.1f}%)")
    out.append(f"  Не размечено: {missing:>5,} ({missing / n_total * 100:5.1f}%)")

    part = df.dropna(subset=["gender"]).groupby("user_no")["gender"].first()
    n_part = len(part)
    out += ["", f"По участникам (N = {n_part}):"]
    pvc = part.value_counts()
    for code in (0, 1):
        cnt = int(pvc.get(code, 0))
        pct = cnt / n_part * 100 if n_part else 0
        out.append(f"  {code} ({GENDER_LABELS[code]}): {cnt:>5,} ({pct:5.1f}%)")
    out.append("")

    # 2. Big Five
    out += [SEP, "2. BIG FIVE PERSONALITY TRAITS (z-scores)", SEP, ""]
    out += stats_table(df, BIG_FIVE, label_map)
    out.append("")

    # 3. Performance
    out += [SEP, "3. PERFORMANCE METRICS (z-scores)", SEP, ""]
    out += stats_table(df, PERFORMANCE, label_map)
    out.append("")

    # 4. Средние по полу (новая секция)
    out += [SEP, "4. СРЕДНИЕ ПО ПОЛУ (z-scores)", SEP, ""]
    out += gender_means_table(df, CONTINUOUS, label_map)
    out.append("")

    out += [SEP, "КОНЕЦ ОТЧЕТА", SEP]

    report = "\n".join(out) + "\n"
    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)

    print(report)
    print(f"\n✓ Сохранено: {OUT_FILE}")


if __name__ == "__main__":
    main()
