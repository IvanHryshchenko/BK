import pandas as pd
import numpy as np
import random

# =========================================================
# Большой качественный датасет для обучения нейросети
# =========================================================

np.random.seed(42)
random.seed(42)

operations = [
    "CUTTING",
    "TUBE CUTTING",
    "EU- LEAD - PREP",
    "LEAD - PREP",
    "FINAL ASSEMBLY",
    "FA TEST",
    "FINAL ASSEMBLY PACKING"
]

# Количество строк
ROWS = 120000

data = []

for i in range(ROWS):

    op = random.choice(operations)

    # =====================================================
    # Реалистичные диапазоны GCSD
    # =====================================================

    if op == "CUTTING":
        gcsd = np.random.normal(4.0, 0.7)

    elif op == "TUBE CUTTING":
        gcsd = np.random.normal(1.5, 0.35)

    elif op == "EU- LEAD - PREP":
        gcsd = np.random.normal(24.0, 3.5)

    elif op == "LEAD - PREP":
        gcsd = np.random.normal(20.0, 2.8)

    elif op == "FINAL ASSEMBLY":
        gcsd = np.random.normal(58.0, 6.0)

    elif op == "FA TEST":
        gcsd = np.random.normal(7.5, 1.0)

    elif op == "FINAL ASSEMBLY PACKING":
        gcsd = np.random.normal(3.2, 0.5)

    # Защита от отрицательных значений
    gcsd = max(0.5, round(gcsd, 4))

    # =====================================================
    # ADJ зависит от операции
    # =====================================================

    if "ASSEMBLY" in op:
        adj = np.random.normal(1.28, 0.07)

    elif "TEST" in op:
        adj = np.random.normal(1.18, 0.05)

    elif "LEAD" in op:
        adj = np.random.normal(1.33, 0.06)

    else:
        adj = np.random.normal(1.12, 0.04)

    adj = round(min(max(adj, 1.05), 1.45), 3)

    # =====================================================
    # Реалистичный шум
    # =====================================================

    # Основная формула
    base = gcsd * adj

    # Шум зависит от размера операции
    noise_percent = np.random.normal(1.0, 0.035)

    # Редкие аномалии (2%)
    if random.random() < 0.02:
        noise_percent *= random.uniform(0.88, 1.15)

    plant_standard = round(base * noise_percent, 4)

    # =====================================================
    # PART NUMBER
    # =====================================================

    prefixes = [
        "A", "B", "C", "D",
        "EU", "US", "JP",
        "X", "Z"
    ]

    suffix = random.randint(1000, 999999)

    if i % 9 == 0:
        part_number = op
    else:
        part_number = f"{random.choice(prefixes)}-{op}-{suffix}"

    # =====================================================
    # Добавление строки
    # =====================================================

    data.append({
        'PART NUMBER': part_number,
        'GCSD': round(gcsd, 4),
        'ADJ.': round(adj, 3),
        'PLANT STANDARD': plant_standard
    })

# =========================================================
# DataFrame
# =========================================================

df = pd.DataFrame(data)

# Перемешивание строк
df = df.sample(frac=1).reset_index(drop=True)

# TOTAL строка
total_row = pd.DataFrame([{
    'PART NUMBER': 'TOTAL',
    'GCSD': round(df['GCSD'].sum(), 4),
    'ADJ.': '',
    'PLANT STANDARD': round(df['PLANT STANDARD'].sum(), 4)
}])

df = pd.concat([df, total_row], ignore_index=True)

# =========================================================
# Сохранение
# =========================================================

OUTPUT_FILE = 'K1_massive.xlsx'

df.to_excel(OUTPUT_FILE, index=False)

print("======================================")
print("✅ Огромный датасет успешно создан")
print("======================================")
print(f"Строк: {len(df)}")
print(f"Файл: {OUTPUT_FILE}")
print("======================================")