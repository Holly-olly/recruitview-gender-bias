"""
Подготовка полной базы данных RecruitView
Все переменные + баллы по шкалам + гендер (без транскриптов)
"""

from pathlib import Path

from datasets import load_from_disk
import pandas as pd
import os

ROOT = Path(__file__).resolve().parent.parent

print("="*80)
print("ПОДГОТОВКА ПОЛНОЙ БАЗЫ ДАННЫХ")
print("="*80)

# Загрузка оригинального датасета
print("\n1. Загрузка датасета RecruitView...")
ds = load_from_disk(str(ROOT / 'recruitview_train'))
df = ds.to_pandas()
print(f"   Загружено записей: {len(df):,}")

# Создаём response_id
df['response_id'] = range(1, len(df) + 1)

# Выбираем нужные колонки (без video и transcript)
columns_to_keep = [
    'response_id',
    'id',                      # unique entry identifier
    'user_no',                 # participant ID
    'question_id',             # question ID
    'question',                # question text
    'video_quality',           # High/Low
    'duration',                # short/medium/long
    # 12 target metrics (Big Five + Performance)
    'openness',
    'conscientiousness',
    'extraversion',
    'agreeableness',
    'neuroticism',
    'overall_personality',
    'interview_score',
    'answer_score',
    'speaking_skills',
    'confidence_score',
    'facial_expression',
    'overall_performance'
]

# Проверяем какие колонки существуют
available_cols = [col for col in columns_to_keep if col in df.columns]
missing_cols = [col for col in columns_to_keep if col not in df.columns]

if missing_cols:
    print(f"\n   ⚠️  Отсутствующие колонки: {missing_cols}")

df_clean = df[available_cols].copy()

print(f"\n2. Базовая таблица подготовлена")
print(f"   Колонок: {len(df_clean.columns)}")
print(f"   Записей: {len(df_clean):,}")

# Загрузка гендера: канонический файл — один пол на участника (0=женщина, 1=мужчина).
# Происхождение: 386 ответов размечены вручную (recruitview_gender.csv), пол распространён
# на всех участников; конфликты/пропуски ручного лога разрешены в gender_by_user.csv.
gender_file = str(ROOT / 'data' / 'gender_by_user.csv')

if os.path.exists(gender_file):
    print(f"\n3. Загрузка гендера по участникам из {gender_file}...")
    gender_df = pd.read_csv(gender_file)
    gender_by_user = gender_df.dropna(subset=['gender']).set_index('user_no')['gender']
    print(f"   Участников с полом: {gender_by_user.shape[0]}")

    # Распространяем пол на все ответы участника по user_no
    df_clean['gender'] = df_clean['user_no'].map(gender_by_user)

    coded = int(df_clean['gender'].notna().sum())
    vc = df_clean['gender'].value_counts()
    print(f"\n   ✓ Гендер распространён по user_no: {coded} / {len(df_clean)} ответов")
    print(f"     Женщин (0): {int(vc.get(0.0, 0))}  Мужчин (1): {int(vc.get(1.0, 0))}")

else:
    print(f"\n3. ⚠️  Файл {gender_file} не найден")
    print(f"   Создаём колонку gender с пустыми значениями")
    df_clean['gender'] = None

# Переупорядочиваем колонки для удобства
column_order = [
    'response_id',
    'id',
    'user_no',
    'gender',              # Гендер сразу после user_no
    'question_id',
    'question',
    'video_quality',
    'duration',
    # Big Five
    'openness',
    'conscientiousness',
    'extraversion',
    'agreeableness',
    'neuroticism',
    'overall_personality',
    # Performance
    'interview_score',
    'answer_score',
    'speaking_skills',
    'confidence_score',
    'facial_expression',
    'overall_performance'
]

# Используем только те колонки, которые есть
final_columns = [col for col in column_order if col in df_clean.columns]
df_final = df_clean[final_columns]

# Сохранение
output_file = str(ROOT / 'data' / 'recruitview_full.csv')
df_final.to_csv(output_file, index=False)

print(f"\n{'='*80}")
print("РЕЗУЛЬТАТ")
print(f"{'='*80}")
print(f"\n✓ Файл сохранён: {output_file}")
print(f"\nСтруктура таблицы:")
print(f"  Записей: {len(df_final):,}")
print(f"  Колонок: {len(df_final.columns)}")

print(f"\nКолонки:")
for i, col in enumerate(df_final.columns, 1):
    dtype = df_final[col].dtype
    missing = df_final[col].isna().sum()
    missing_pct = missing / len(df_final) * 100
    print(f"  {i:2d}. {col:25s} {str(dtype):10s} (пропусков: {missing:4d}, {missing_pct:5.1f}%)")

# Статистика по гендеру
if 'gender' in df_final.columns:
    print(f"\n{'='*80}")
    print("СТАТИСТИКА ПО ГЕНДЕРУ")
    print(f"{'='*80}")

    gender_counts = df_final['gender'].value_counts()
    gender_missing = df_final['gender'].isna().sum()

    print(f"\nРаспределение:")
    print(f"  Женщин (0): {gender_counts.get(0.0, 0)}")
    print(f"  Мужчин (1): {gender_counts.get(1.0, 0)}")
    print(f"  Не размечено: {gender_missing}")
    print(f"\nПрогресс разметки: {(1 - gender_missing/len(df_final))*100:.1f}%")

# Примеры данных
print(f"\n{'='*80}")
print("ПРИМЕРЫ ДАННЫХ")
print(f"{'='*80}")

print(f"\nПервые 3 записи:")
print(df_final.head(3)[['response_id', 'user_no', 'gender', 'question_id', 'openness', 'interview_score']].to_string(index=False))

print(f"\n{'='*80}")
print("ГОТОВО")
print(f"{'='*80}")
print(f"\nФайл: {output_file}")
print(f"Размер: {os.path.getsize(output_file) / 1024:.1f} KB")
