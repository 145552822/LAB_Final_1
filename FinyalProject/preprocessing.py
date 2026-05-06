import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler


def preprocess(df):
    """
    предобработка данных:
        очистка названий столбцов
        заполнение пропусков
        кодирование категориальных признаков
        нормализация числовых признаков
    """

    # Очистка пробелов в названиях столбцов и строковых значениях
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    print(f"\nСтолбцы после очистки: {df.columns.tolist()}")
    print()

    # проверка на пропуски до обработки
    print(f"\nПропуски до обработки: {df.isnull().sum()}")

    # удаление бесполезного хлама.
    features_to_drop = [
        'Процент на телефоне',
        'IOS/ Android'
    ]

    for feature in features_to_drop:
        if feature in df.columns:
            print(f"Удалён признак: {feature}")
            df = df.drop(columns=[feature])

    # преобразование столбца 'Время заказа' в числовой признак (час дня)
    if 'Время заказа' in df.columns:
        df['Час заказа'] = pd.to_datetime(df['Время заказа'], format='%H:%M', errors='coerce').dt.hour
        df.drop(columns=['Время заказа'], inplace=True)

    # заполнение числовых пропусков медианой
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isnull().any():
            median_val = df[col].median()
            df[col].fillna(median_val, inplace=True)
            print(f"Заполнены пропуски в '{col}' медианой = {median_val}")

    # заполнение категориальных пропусков модой
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isnull().any():
            mode_val = df[col].mode()[0]
            df[col].fillna(mode_val, inplace=True)
            print(f"Заполнены пропуски в '{col}' модой = {mode_val}")

    # проверка пропусков после заполнения
    print(f"\nПропуски после заполнения: {df.isnull().sum()}")

    # дополнительная проверка и удаление оставшихся NaN (если есть)
    if df.isnull().any().any():
        before_len = len(df)
        df = df.dropna()
        after_len = len(df)
        print(f"Удалено {before_len - after_len} строк с пропусками")

    # кодирование категориальных признаков с помощью LabelEncoder
    le = LabelEncoder()
    encoded_cols = []
    for col in cat_cols:
        df[col] = le.fit_transform(df[col])
        encoded_cols.append(col)

    print(f"\nЗакодированы категориальные столбцы: {encoded_cols}")

    # целевая переменная - 'Цена'
    target_col = 'Цена'
    if target_col not in df.columns:
        raise ValueError(f"Целевой столбец '{target_col}' не найден в данных!")

    # разделение на признаки и целевую переменную
    X = df.drop(columns=[target_col])
    y = df[target_col]

    print(f"\nФинальная проверка:")
    print(f"NaN в X: {X.isnull().any().any()}")
    print(f"NaN в y: {y.isnull().any()}")

    if X.isnull().any().any():
        nan_rows = X.isnull().any(axis=1)
        X = X[~nan_rows]
        y = y[~nan_rows]
        print(f"Удалено {nan_rows.sum()} строк. Осталось {len(X)} строк")

    # нормализация числовых признаков (StandardScaler)
    feature_names = X.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=feature_names)

    print(f"\nПризнаки для обучения ({len(feature_names)}): {feature_names}")
    print(f"Целевая переменная: '{target_col}', строк: {len(y)}")
    print(f"Размер X_scaled: {X_scaled.shape}")
    print(f"Размер y: {y.shape}")

    return X_scaled, y, feature_names