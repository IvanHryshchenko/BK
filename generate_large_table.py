import pandas as pd
import numpy as np
import random

# =============================================
# Генерация большой таблицы точно как у тебя
# =============================================

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

data = []

for i in range(1520):  # ~1500 строк + оригинальные
    op = random.choice(operations)
    
    if op == "CUTTING":
        gcsd = round(random.uniform(2.5, 5.5), 4)
    elif op == "TUBE CUTTING":
        gcsd = round(random.uniform(0.8, 2.2), 4)
    elif "LEAD" in op:
        gcsd = round(random.uniform(15, 28), 4)
    elif op == "FINAL ASSEMBLY":
        gcsd = round(random.uniform(45, 72), 4)
    elif op == "FA TEST":
        gcsd = round(random.uniform(5.5, 9.5), 4)
    else:
        gcsd = round(random.uniform(2.0, 4.5), 4)
    
    adj = round(random.uniform(1.05, 1.45), 3)
    plant_standard = round(gcsd * adj * random.uniform(0.96, 1.09), 4)
    
    part_number = op if i % 8 == 0 else f"{op} {i}"
    
    data.append({
        'PART NUMBER': part_number,
        'GCSD': gcsd,
        'ADJ.': adj,
        'PLANT STANDARD': plant_standard
    })

# Создаём DataFrame
df = pd.DataFrame(data)

# Добавляем строку TOTAL в конце (как у тебя)
total_row = pd.DataFrame([{
    'PART NUMBER': 'TOTAL',
    'GCSD': round(df['GCSD'].sum(), 4),
    'ADJ.': '',
    'PLANT STANDARD': round(df['PLANT STANDARD'].sum(), 4)
}])

df = pd.concat([df, total_row], ignore_index=True)

# Сохраняем
df.to_excel('K1_large.xlsx', index=False)

print("✅ Файл успешно создан!")
print(f"Количество строк: {len(df)}")
print("Файл сохранён как: K1_large.xlsx")
print("\nТеперь просто переименуй его в K1.xlsx и запускай train.py")