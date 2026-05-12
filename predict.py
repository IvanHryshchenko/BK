import torch
import pandas as pd
import pickle
from data_preparation import load_data

# Определяем класс модели (обязательно!)
class HarnessNet(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Sequential(
            torch.nn.Linear(3, 128),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.2),
            torch.nn.Linear(128, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.15),
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

# Загрузка модели и инструментов
model = HarnessNet()
model.load_state_dict(torch.load('harness_model.pth', weights_only=True))
model.eval()

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('label_encoder.pkl', 'rb') as f:
    le = pickle.load(f)

print("✅ Модель нейронной сети загружена!")

# Загрузка данных
df = load_data('K1_large.xlsx')

# Подготовка
df['Operation_encoded'] = le.transform(df['PART NUMBER'].astype(str))
X = df[['GCSD', 'ADJ.', 'Operation_encoded']].values
X_scaled = scaler.transform(X)

# Предсказание
with torch.no_grad():
    pred = model(torch.FloatTensor(X_scaled)).numpy().flatten()

df['PLANT_STANDARD_AI'] = pred.round(4)
df['Разница'] = (df['PLANT STANDARD'] - df['PLANT_STANDARD_AI']).round(4)
df['Абсолютная_ошибка'] = abs(df['Разница'])

# Результат
print(f"\nСредняя абсолютная ошибка: ±{df['Абсолютная_ошибка'].mean():.3f} минут")
print("\nПервые 10 строк:")
print(df[['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD', 'PLANT_STANDARD_AI', 'Разница']].head(10).round(4))

df.to_excel("RESULT_WITH_AI.xlsx", index=False)
print("\n✅ Результат сохранён в RESULT_WITH_AI.xlsx")