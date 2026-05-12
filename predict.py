import torch
import pandas as pd
import pickle
from data_preparation import load_data

class HarnessNet(torch.nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(input_size, 512),
            torch.nn.BatchNorm1d(512),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.25),
            torch.nn.Linear(512, 256),
            torch.nn.BatchNorm1d(256),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Загрузка
model = HarnessNet(input_size=11)  # количество фич
model.load_state_dict(torch.load('harness_model_improved.pth', weights_only=True))
model.eval()

with open('scaler_improved.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('feature_cols.pkl', 'rb') as f:
    feature_cols = pickle.load(f)

print("✅ Улучшенная модель загружена!")

# Предсказание
df = load_data('K1_large.xlsx')

# Создаём те же фичи
df = df.copy()  # чтобы не модифицировать оригинал
from data_preparation import create_features
df = create_features(df)

X = df[feature_cols].values
X_scaled = scaler.transform(X)

with torch.no_grad():
    pred = model(torch.FloatTensor(X_scaled)).numpy().flatten()

df['PLANT_STANDARD_AI'] = pred.round(4)
df['Разница'] = (df['PLANT STANDARD'] - df['PLANT_STANDARD_AI']).round(4)
df['Абсолютная_ошибка'] = abs(df['Разница'])

print(f"\nСредняя абсолютная ошибка: ±{df['Абсолютная_ошибка'].mean():.3f} минут")
print(f"Медианная ошибка: ±{df['Абсолютная_ошибка'].median():.3f} минут")

print("\nПервые 10 строк:")
print(df[['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD', 'PLANT_STANDARD_AI', 'Разница']].head(10))

df.to_excel("RESULT_WITH_AI_IMPROVED.xlsx", index=False)
print("\n✅ Результат сохранён в RESULT_WITH_AI_IMPROVED.xlsx")