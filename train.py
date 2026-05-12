from data_preparation import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle

# Загружаем данные
df = load_data('K1.xlsx')

# === ДОПОЛНИТЕЛЬНАЯ ОЧИСТКА ===
print("До очистки:", len(df))
df = df.dropna(subset=['GCSD', 'ADJ.', 'PLANT STANDARD'])  # убираем все строки с NaN
df = df[df['GCSD'] > 0]                                   # убираем нулевые и отрицательные
print("После очистки:", len(df))

le = LabelEncoder()
df['Operation_encoded'] = le.fit_transform(df['Operation_Type'])

features = ['GCSD', 'ADJ.', 'Operation_encoded']
target = 'PLANT STANDARD'

X = df[features].values
y = df[target].values

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train).unsqueeze(1)

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
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("🔄 Обучение модели...")
for epoch in range(800):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    
    if epoch % 200 == 0:
        print(f"Эпоха {epoch} | Loss: {loss.item():.4f}")

# Сохранение
torch.save(model.state_dict(), 'harness_model.pth')
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(le, f)

print("✅ Обучение завершено!")
print(f"Количество строк, на которых обучалась модель: {len(df)}")