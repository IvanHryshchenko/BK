import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from data_preparation import load_data

df = load_data('K1_large.xlsx')

le = LabelEncoder()
df['Operation_encoded'] = le.fit_transform(df['PART NUMBER'].astype(str))

X = df[['GCSD', 'ADJ.', 'Operation_encoded']].values
y = df['PLANT STANDARD'].values.reshape(-1, 1)

# Разделяем на train / validation
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

class HarnessNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model = HarnessNet()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

best_loss = float('inf')
patience = 800
counter = 0

print("🚀 Обучение с валидацией...")

for epoch in range(15000):      # можно очень много
    # Train
    model.train()
    optimizer.zero_grad()
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    loss.backward()
    optimizer.step()

    # Validation
    model.eval()
    with torch.no_grad():
        val_outputs = model(X_val)
        val_loss = criterion(val_outputs, y_val).item()

    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model.state_dict().copy()
        counter = 0
    else:
        counter += 1

    if epoch % 500 == 0:
        print(f"Эпоха {epoch:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Best Val: {best_loss:.4f}")

    if counter >= patience:
        print(f"Ранняя остановка на эпохе {epoch}")
        break

model.load_state_dict(best_state)
torch.save(model.state_dict(), 'harness_model.pth')
with open('scaler.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('label_encoder.pkl', 'wb') as f: pickle.dump(le, f)

print(f"\n✅ Обучение завершено! Лучший Val Loss: {best_loss:.4f}")