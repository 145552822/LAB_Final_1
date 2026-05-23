"""
predictor.py — Интерактивное предсказание цены такси после обучения модели.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


# Известные значения из датасета

KNOWN_COMPANIES = ['Yandex Go', 'InDrive', 'Navi Taxi']
KNOWN_TARIFFS   = ['Эконом', 'Комфорт', 'Комфорт +']
KNOWN_WEATHER   = ['Ясно', 'Прохладно', 'Пасмурно', 'Дождливый', 'Холодно', 'Жарко']
KNOWN_DEMAND    = ['да', 'нет', '50/50']


def _label_encode(value: str, known_values: list) -> int:
    """LabelEncoder без sklearn — воспроизводит алфавитный порядок fit."""
    mapping = {v: i for i, v in enumerate(sorted(set(known_values)))}
    return mapping.get(value, 0)


def _get_scaler_and_encoders(df_raw: pd.DataFrame, feature_names: list):
    """
    Создаёт и обучает StandardScaler и LabelEncoders на основе исходных данных.
    Возвращает (scaler, encoders_dict)
    """
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # Удаляем бесполезные признаки
    for col in ['Процент на телефоне', 'IOS/ Android', 'Время заказа']:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)

    # Преобразуем 'Час заказа' если нужно
    if 'Час заказа' not in df.columns and 'Время заказа' in df_raw.columns:
        df['Час заказа'] = pd.to_datetime(
            df_raw['Время заказа'], format='%H:%M', errors='coerce'
        ).dt.hour

    # Пересчитываем 'сом за км'
    if 'Расстояние' in df.columns and 'Цена' in df.columns:
        df['сом за км'] = df['Цена'] / df['Расстояние']

    # Кодируем категориальные признаки
    encoders = {}
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Оставляем только нужные признаки
    df = df[feature_names]

    # Масштабируем
    scaler = StandardScaler()
    scaler.fit(df)

    return scaler, encoders


def predict_price(model, feature_names: list, df_raw: pd.DataFrame,
                  user_input: dict) -> float:
    """
    Предсказывает цену по частичным данным от пользователя.
    """
    # Получаем обученные scaler и encoders
    scaler, encoders = _get_scaler_and_encoders(df_raw, feature_names)

    # Собираем строку с признаками
    row = {}

    # Категориальные признаки (кодируем теми же encoder)
    cat_mapping = {
        'Компания': KNOWN_COMPANIES,
        'Эконом': KNOWN_TARIFFS,
        'Погода': KNOWN_WEATHER,
        'Повышенный спрос': KNOWN_DEMAND
    }

    for col in feature_names:
        if col in cat_mapping:
            # Категориальный признак
            if col in user_input:
                # Пытаемся найти соответствие в encoder
                try:
                    row[col] = encoders[col].transform([user_input[col]])[0]
                except:
                    # Если значения нет в обученном encoder, используем первое значение
                    row[col] = 0
            else:
                # Используем моду (наиболее частотное значение)
                row[col] = 0  # default
        else:
            # Числовой признак
            if col in user_input:
                row[col] = float(user_input[col])
            else:
                # Используем медиану из датасета
                if col == 'Расстояние':
                    row[col] = df_raw['Расстояние'].median()
                elif col == 'сом за км':
                    # Пересчитываем медиану из цены/расстояние
                    median_price = df_raw['Цена'].median()
                    median_distance = df_raw['Расстояние'].median()
                    row[col] = median_price / median_distance if median_distance > 0 else 45
                elif col == 'Время поездки/мин':
                    row[col] = df_raw['Время поездки/мин'].median()
                elif col == 'Час заказа':
                    # Извлекаем час из времени заказа
                    if 'Время заказа' in df_raw.columns:
                        hours = pd.to_datetime(df_raw['Время заказа'], format='%H:%M', errors='coerce').dt.hour
                        row[col] = hours.median()
                    else:
                        row[col] = 14
                else:
                    row[col] = 0

    # Создаём DataFrame в правильном порядке признаков
    df_row = pd.DataFrame([row])[feature_names]

    # Масштабируем
    df_row_scaled = scaler.transform(df_row)

    # Предсказываем
    price = model.predict(df_row_scaled)[0]
    return round(float(price), 1)


# Интерактивный режим

def _ask(prompt: str, valid_options: list = None, value_type=str,
         required: bool = False):
    """Запрашивает значение у пользователя."""
    if valid_options:
        options_str = ' / '.join([f"{i+1}.{v}" for i, v in enumerate(valid_options)])
        full_prompt = f"{prompt} [{options_str}] (Enter=пропустить): "
    else:
        full_prompt = f"{prompt} (Enter=пропустить): "

    while True:
        raw = input(full_prompt).strip()

        if raw == '':
            if required:
                print("Это поле обязательно для заполнения.")
                continue
            return None

        if valid_options and raw.isdigit():
            idx = int(raw) - 1
            if 0 <= idx < len(valid_options):
                return valid_options[idx]
            else:
                print(f"Введите число от 1 до {len(valid_options)}")
                continue

        if valid_options:
            match = next((v for v in valid_options if v.lower() == raw.lower()), None)
            if match:
                return match
            print(f"Допустимые значения: {valid_options}")
            continue

        try:
            return value_type(raw)
        except ValueError:
            print(f"Ожидается {value_type.__name__}, попробуйте ещё раз.")


def predict_interactive(model, feature_names: list, df_raw: pd.DataFrame):
    """Интерактивный цикл предсказания цены такси."""
    print("\n" + "="*60)
    print("ПРЕДСКАЗАНИЕ ЦЕНЫ ТАКСИ")
    print("Заполните известные поля. Пустые поля — нажмите Enter.")
    print("="*60)

    while True:
        print()
        user_input = {}

        # Категориальные признаки
        val = _ask("Компания", KNOWN_COMPANIES)
        if val: user_input['Компания'] = val

        val = _ask("Тариф", KNOWN_TARIFFS)
        if val: user_input['Эконом'] = val

        val = _ask("Погода", KNOWN_WEATHER)
        if val: user_input['Погода'] = val

        val = _ask("Повышенный спрос", KNOWN_DEMAND)
        if val: user_input['Повышенный спрос'] = val

        # Числовые признаки
        val = _ask("Расстояние (км), например 8.5", value_type=float)
        if val is not None: user_input['Расстояние'] = val

        val = _ask("Время поездки (мин), например 25", value_type=float)
        if val is not None: user_input['Время поездки/мин'] = val

        val = _ask("Час заказа (0-23), например 18", value_type=int)
        if val is not None: user_input['Час заказа'] = val

        val = _ask("Цена за км (сом/км), например 45", value_type=float)
        if val is not None: user_input['сом за км'] = val

        # Предсказание
        print()
        if not user_input:
            print("Вы не ввели ни одного значения. Будут использованы средние по датасету.")

        print("Введённые данные:")
        for k, v in user_input.items():
            print(f"    {k}: {v}")

        all_fields = ['Компания', 'Эконом', 'Погода', 'Повышенный спрос', 'Расстояние', 'Время поездки/мин', 'Час заказа', 'сом за км']
        skipped = [f for f in all_fields if f not in user_input]
        if skipped:
            print(f"Пропущенные поля (заполнятся медианой/модой): {', '.join(skipped)}")

        try:
            price = predict_price(model, feature_names, df_raw, user_input)
            print(f"\nПредсказанная цена: {price:.0f} сом")
        except Exception as e:
            print(f"\nОшибка при предсказании: {e}")
            import traceback
            traceback.print_exc()

        # Ещё раз
        print()
        again = input("Предсказать ещё раз? (y / Enter=выход): ").strip().lower()
        if again not in ('y', 'д', 'да', 'yes'):
            print("Выход из режима предсказания.\n")
            break
