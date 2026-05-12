from data_preparation import load_data
import torch
import torch.nn as nn
import pickle
import pandas as pd

# ====================== ЗАГРУЗКА МОДЕЛИ ======================
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        return self.model(x)

model = SimpleNet()
model.load_state_dict(torch.load('harness_model.pth'))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("Модель загружена!\n")

# ====================== ПРЕДСКАЗАНИЕ ======================
df = load_data('K1.xlsx')

# Фильтруем только нормальные строки (убираем TOTAL и пустые)
df = df[~df['PART NUMBER'].astype(str).str.contains('TOTAL', na=False)]
df = df[df['GCSD'] > 0]

df['Operation_encoded'] = le.transform(df['Operation_Type'])
X = df[['GCSD', 'ADJ.', 'Operation_encoded']].values
X_scaled = scaler.transform(X)

X_tensor = torch.FloatTensor(X_scaled)

with torch.no_grad():
    predictions = model(X_tensor).numpy().flatten()

df['PLANT_STANDARD_AI'] = predictions.round(4)
df['Разница'] = (df['PLANT STANDARD'] - df['PLANT_STANDARD_AI']).round(3)
df['Абсолютная_ошибка'] = df['Разница'].abs()

# Сохраняем
df.to_excel('RESULT_WITH_AI.xlsx', index=False)

print(f"✅ Готово! Обработано строк: {len(df)}")
print("\nРезультат:")
print(df[['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD', 'PLANT_STANDARD_AI', 'Разница']].round(3))
print(f"\nСредняя ошибка: ±{df['Абсолютная_ошибка'].mean():.3f} минут")