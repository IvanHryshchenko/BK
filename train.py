import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import pickle
from data_preparation import load_data

# ====================== ЗАГРУЗКА ДАННЫХ ======================
df = load_data('K1_large.xlsx')

# Выбираем фичи
feature_cols = ['GCSD', 'ADJ.', 'GCSD_log', 'ADJ_log', 'GCSD_x_ADJ', 'GCSD_div_ADJ',
                'PART_LEN', 'PART_DIGITS', 'PART_HAS_LETTER', 'PART_TARGET_MEAN', 'PART_FREQ']

X = df[feature_cols].values
y = df['PLANT STANDARD'].values.reshape(-1, 1)

# Разделение
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

X_train = torch.FloatTensor(X_train)
y_train = torch.FloatTensor(y_train)
X_val = torch.FloatTensor(X_val)
y_val = torch.FloatTensor(y_val)

# ====================== МОДЕЛЬ ======================
class HarnessNet(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Dropout(0.25),
            
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.15),
            
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    
    def forward(self, x):
        return self.model(x)

model = HarnessNet(input_size=len(feature_cols))
criterion = nn.MSELoss()
optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=400, factor=0.5, verbose=True)

# ====================== ОБУЧЕНИЕ ======================
best_loss = float('inf')
patience = 1200
counter = 0
best_state = None

print("🚀 Запуск улучшенного обучения...")

for epoch in range(20000):
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
    
    scheduler.step(val_loss)
    
    if val_loss < best_loss:
        best_loss = val_loss
        best_state = model.state_dict().copy()
        counter = 0
    else:
        counter += 1
    
    if epoch % 500 == 0:
        print(f"Эпоха {epoch:5d} | Train Loss: {loss.item():.4f} | Val Loss: {val_loss:.4f} | Best: {best_loss:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
    
    if counter >= patience:
        print(f"Ранняя остановка на эпохе {epoch}")
        break

model.load_state_dict(best_state)
torch.save(model.state_dict(), 'harness_model_improved.pth')
with open('scaler_improved.pkl', 'wb') as f: pickle.dump(scaler, f)
with open('feature_cols.pkl', 'wb') as f: pickle.dump(feature_cols, f)

print(f"\n✅ ОБУЧЕНИЕ ЗАВЕРШЕНО! Лучший Val Loss: {best_loss:.4f}")