import pandas as pd

# Коэффициенты из линейной регрессии (train.py)
w1 = 1.2875177919128709   # для GCSD
w2 = 17.39800771745418    # для ADJ.
b = -21.83017563915144

print("Загружаем K1_large.xlsx...")

# Загружаем файл
df = pd.read_excel("K1_large.xlsx")

print(f"Загружено строк: {len(df)}")
print("Колонки:", df.columns.tolist())

# Очистка и выбор колонок
df = df[['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD']].copy()

# Убираем строки TOTAL и пустые
df = df[df['GCSD'] > 0.1].copy()
df = df[~df['PART NUMBER'].astype(str).str.contains('TOTAL|SUMMARY', na=False)]

print(f"Строк после очистки: {len(df)}")

# Предсказание
df['PLANT_STANDARD_AI'] = w1 * df['GCSD'] + w2 * df['ADJ.'] + b

# Разница
df['Разница'] = df['PLANT STANDARD'] - df['PLANT_STANDARD_AI']
df['Абсолютная_ошибка'] = abs(df['Разница'])

# Результаты
print("\nПервые 10 строк результата:")
print(df[['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD', 'PLANT_STANDARD_AI', 'Разница']].round(4).head(10))

print(f"\nСредняя абсолютная ошибка: ±{df['Абсолютная_ошибка'].mean():.3f} минут")

# Сохраняем
df.to_excel("RESULT_WITH_AI.xlsx", index=False)
print("\n✅ Результат успешно сохранён в файл: RESULT_WITH_AI.xlsx")