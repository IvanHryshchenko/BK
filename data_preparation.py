import pandas as pd

def load_data(file_path='K1_large.xlsx'):
    print(f"Загружаем {file_path}...")
    df = pd.read_excel(file_path, header=0)
    
    # Приводим названия колонок
    df.columns = [str(col).strip() for col in df.columns]
    
    # Гибкое сопоставление
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
    df = df[~df['PART NUMBER'].astype(str).str.contains('TOTAL|SUMMARY', na=False)]
    
    print(f"✅ Успешно загружено {len(df)} строк")
    return df