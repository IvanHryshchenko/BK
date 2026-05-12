import pandas as pd
import numpy as np

def load_data(file_path='K1.xlsx'):
    print("Реальные названия колонок:")
    
    # Пробуем разные варианты чтения
    df = pd.read_excel(file_path, header=0)
    print(df.columns.tolist())
    
    # Если колонок мало или есть Unnamed — пробуем пропустить строки
    if len(df.columns) < 4 or any('Unnamed' in str(col) for col in df.columns):
        df = pd.read_excel(file_path, header=1)  # пробуем вторую строку
    
    real_cols = df.columns.tolist()
    print("Реальные названия колонок:", real_cols)
    
    # Гибкое сопоставление колонок
    col_map = {}
    for col in real_cols:
        col_str = str(col).strip().upper()
        if 'PART NUMBER' in col_str or 'PART' in col_str:
            col_map['PART NUMBER'] = col
        elif 'GCSD' in col_str:
            col_map['GCSD'] = col
        elif 'ADJ' in col_str:
            col_map['ADJ.'] = col
        elif 'PLANT STANDARD' in col_str or 'PLANT' in col_str:
            col_map['PLANT STANDARD'] = col
    
    print("Найденные колонки:", col_map)
    
    if len(col_map) < 4:
        raise KeyError("Не удалось найти все необходимые колонки!")
    
    # Берем только нужные колонки
    df = df[[col_map['PART NUMBER'], 
             col_map['GCSD'], 
             col_map['ADJ.'], 
             col_map['PLANT STANDARD']]].copy()
    
    # Переименовываем для удобства
    df.columns = ['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD']
    
    # Очистка
    df = df[df['GCSD'] > 0.1]
    df = df[~df['PART NUMBER'].astype(str).str.contains('TOTAL|SUMMARY', na=False, case=False)]
    
    print(f"✅ Успешно загружено {len(df)} строк для работы")
    return df