import pandas as pd
import os

def load_data(filepath='Taxi Price 1.csv'):
    """
    Загрузка датасета из CSV-файла с разделителем ';'.
    Возвращает DataFrame с обработанными типами данных.
    """

    # чтение CSV
    df = pd.read_csv(filepath, sep=';', decimal=',', encoding='utf-8-sig')

    print("\nобщая информация о датасете")
    df.info()

    print("\nИзначальная статистика числовых признаков")
    print(df.describe())

    print("\nПервые 5 строк")
    print(df.head())

    print("\nПропущенные значения")
    print(df.isnull().sum())

    print(f"Датасет загружен: {df.shape[0]} строк, {df.shape[1]} столбцов")

    return df