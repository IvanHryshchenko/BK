import pandas as pd
import numpy as np

def load_data(file_path='K1.xlsx'):
    df = pd.read_excel(file_path, sheet_name='TOTAL values GCSD', header=2)
    
    # Показываем реальные названия колонок (для отладки)
    print("Реальные названия колонок:", df.columns.tolist())
    
    # Очищаем названия колонок от лишних пробелов
    df.columns = [str(col).strip() for col in df.columns]
    
    # Ищем нужные колонки (гибкий поиск)
    col_map = {}
    for col in df.columns:
        if 'PART NUMBER' in col.upper():
            col_map['PART NUMBER'] = col
        elif 'GCSD' in col.upper():
            col_map['GCSD'] = col
        elif 'ADJ' in col.upper():
            col_map['ADJ.'] = col
        elif 'PLANT STANDARD' in col.upper():
            col_map['PLANT STANDARD'] = col
    
    print("Найденные колонки:", col_map)
    
    # Берем только нужные
    df = df[[col_map['PART NUMBER'], 
             col_map['GCSD'], 
             col_map['ADJ.'], 
             col_map['PLANT STANDARD']]].copy()
    
    # Переименовываем для удобства
    df.columns = ['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD']
    
    # Приводим к числам
    for col in ['GCSD', 'ADJ.', 'PLANT STANDARD']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Тип операции
    df['Operation_Type'] = df['PART NUMBER'].astype(str).apply(lambda x: 
        'CUTTING' if 'CUT' in x.upper() else 
        'LEAD' if any(w in x.upper() for w in ['LEAD','PREP','R2']) else 
        'ASSEMBLY' if 'ASSEMBLY' in x.upper() or 'FINAL' in x.upper() else 'OTHER')
    
    df = df.dropna(subset=['GCSD', 'PLANT STANDARD']).reset_index(drop=True)
    
    print(f"✅ Успешно загружено {len(df)} строк для обучения")
    return df