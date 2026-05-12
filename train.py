import pandas as pd
from sklearn.linear_model import LinearRegression

# загрузка
df = pd.read_excel("K1_large.xlsx")

print("Колонки:", df.columns)

# оставляем только нужные
df = df[['GCSD', 'ADJ.', 'PLANT STANDARD']].dropna()

# X — входы
X = df[['GCSD', 'ADJ.']]

# y — что предсказываем
y = df['PLANT STANDARD']

# модель
model = LinearRegression()
model.fit(X, y)

print("\nКоэффициенты:")
print("GCSD =", model.coef_[0])
print("ADJ =", model.coef_[1])
print("Свободный член =", model.intercept_)