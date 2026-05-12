import pandas as pd
import numpy as np

def load_data(file_path='K1_large.xlsx'):
    print(f"Загружаем {file_path}...")
    df = pd.read_excel(file_path, header=0)
    df.columns = [str(col).strip() for col in df.columns]
    
    # Гибкое сопоставление колонок
    col_map = {}
    for col in df.columns:
        col_upper = col.upper()
        if 'PART' in col_upper:
            col_map['PART NUMBER'] = col
        elif 'GCSD' in col_upper:
            col_map['GCSD'] = col
        elif 'ADJ' in col_upper:
            col_map['ADJ.'] = col
        elif 'PLANT' in col_upper or 'STANDARD' in col_upper:
            col_map['PLANT STANDARD'] = col
    
    print("Найденные колонки:", col_map)
    
    df = df[[col_map['PART NUMBER'], col_map['GCSD'], col_map['ADJ.'], col_map['PLANT STANDARD']]].copy()
    df.columns = ['PART NUMBER', 'GCSD', 'ADJ.', 'PLANT STANDARD']
    
    # Очистка
    df = df[df['GCSD'] > 0.1].copy()
    df = df[~df['PART NUMBER'].astype(str).str.contains('TOTAL|SUMMARY', na=False, case=False)]
    
    # === НОВЫЕ ФИЧИ ===
    df = create_features(df)
    
    print(f"✅ Успешно загружено {len(df)} строк")
    return df


def create_features(df):
    df = df.copy()
    
    # Базовые
    df['GCSD_log'] = np.log1p(df['GCSD'])
    df['ADJ_log'] = np.log1p(df['ADJ.'])
    df['GCSD_x_ADJ'] = df['GCSD'] * df['ADJ.']
    df['GCSD_div_ADJ'] = df['GCSD'] / (df['ADJ.'] + 1e-8)
    
    # Статистики по PART NUMBER
    df['PART_LEN'] = df['PART NUMBER'].astype(str).str.len()
    df['PART_DIGITS'] = df['PART NUMBER'].astype(str).str.count(r'\d')
    df['PART_HAS_LETTER'] = df['PART NUMBER'].astype(str).str.contains(r'[A-Za-z]').astype(int)
    
    # Target Encoding (среднее по операции) - очень мощная фича
    target_mean = df.groupby('PART NUMBER')['PLANT STANDARD'].mean()
    df['PART_TARGET_MEAN'] = df['PART NUMBER'].map(target_mean)
    
    # Частота операции
    freq = df['PART NUMBER'].value_counts()
    df['PART_FREQ'] = df['PART NUMBER'].map(freq)
    
    print(f"Создано {len(df.columns)-4} фич")
    return df