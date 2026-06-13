"""
Матрица покрытия «участник × вопрос».

Строки  = user_no (участники)
Столбцы = question_id (вопросы интервью)
Ячейка  = число ответов участника на этот вопрос (0 = не отвечал, >1 = повтор)

Запуск (из любой папки):
    python src/build_coverage_matrix.py
Выход:
    results/tables/user_question_matrix.csv
"""

from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_FILE = ROOT / "data" / "recruitview_full.csv"
CSV_OUT = ROOT / "outputs" / "stage1" / "user_question_matrix.csv"


def main() -> None:
    df = pd.read_csv(DATA_FILE)

    # числовая сортировка id
    df["user_no"] = pd.to_numeric(df["user_no"], errors="coerce").astype("Int64")
    df["question_id"] = pd.to_numeric(df["question_id"], errors="coerce").astype("Int64")

    # матрица: число ответов в каждой ячейке (user × question)
    mat = (
        df.pivot_table(index="user_no", columns="question_id",
                       values="response_id", aggfunc="count", fill_value=0)
        .sort_index(axis=0)
        .sort_index(axis=1)
        .astype(int)
    )

    CSV_OUT.parent.mkdir(parents=True, exist_ok=True)
    mat.to_csv(CSV_OUT)

    # --- сводка ---
    q_per_user = (mat > 0).sum(axis=1)
    u_per_q = (mat > 0).sum(axis=0)
    n_dupe_cells = int((mat > 1).sum().sum())
    print(f"Матрица: {mat.shape[0]} участников × {mat.shape[1]} вопросов")
    print(f"Вопросов на участника: mean={q_per_user.mean():.2f} "
          f"min={q_per_user.min()} max={q_per_user.max()}")
    print(f"Участников на вопрос:  mean={u_per_q.mean():.2f} "
          f"min={u_per_q.min()} max={u_per_q.max()}")
    print(f"Ячеек с повтором (>1 ответа): {n_dupe_cells}")
    print(f"Самые частые вопросы:\n{u_per_q.sort_values(ascending=False).head(5).to_string()}")
    print(f"\n✓ CSV: {CSV_OUT}")


if __name__ == "__main__":
    main()
