import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_and_evaluate(X, y, feature_names):
    """
    Обучение минимум 3 моделей из разных семейств:
        1 LinearRegression - линейная модель (базовая)
        2 RandomForest - ансамбль деревьев (бэггинг)
        3 GradientBoosting - ансамбль деревьев (бустинг)
    Возвращает таблицу метрик, лучшую модель, важность признаков.
    """

    print("\nФинальная проверка данных перед обучением:")
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    print(f"NaN в X: {X.isnull().any().any()}")
    print(f"NaN в y: {y.isnull().any()}")
    print(f"Inf в X: {np.isinf(X).any().any()}")

    if X.isnull().any().any() or np.isinf(X).any().any():
        mask = ~(X.isnull().any(axis=1) | np.isinf(X).any(axis=1))
        X = X[mask]
        y = y[mask]

    # разделение данных на основную и тестовую выборки (80/20)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"\nРазмер обучающей выборки: {X_train.shape[0]} строк")
    print(f"Размер тестовой выборки:  {X_test.shape[0]} строк")

    # определение моделей
    models = {
        'LinearRegression': LinearRegression(),
        'RandomForest': RandomForestRegressor(n_estimators=100,random_state=42),
        'GradientBoosting': GradientBoostingRegressor(n_estimators=100,random_state=42,learning_rate=0.1)
    }

    results = []  # список для хранения метрик каждой модели
    trained = {}  # словарь обученных моделей

    # обучение и оценка каждой модели
    for name, model in models.items():
        print(f"Обучение модели: {name}")

        try:
            # обучение модели на тренировочных данных
            model.fit(X_train, y_train)

            # предсказание на тестовых данных
            y_pred = model.predict(X_test)

            # расчёт метрик качества регрессии
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)

            print(f"MAE = {mae:.4f}  (Средняя Абсолютная Ошибка)")
            print(f"RMSE = {rmse:.4f} (Корень из Средней Квадратичной Ошибки)")
            print(f"R² = {r2:.4f}  (коэффициент детерминации)")

            results.append({
                'Модель': name,
                'MAE': round(mae, 4),
                'RMSE': round(rmse, 4),
                'R²': round(r2, 4)
            })
            trained[name] = model

        except Exception as e:
            print(f"Ошибка при обучении {name}: {e}")
            results.append({
                'Модель': name,
                'MAE': np.nan,
                'RMSE': np.nan,
                'R²': np.nan
            })

    # сводная таблица сравнения моделей
    results_df = pd.DataFrame(results)
    print("\n\n\n   Сравнение Моделей")
    print(results_df.to_string(index=False))

    # выбор лучшей модели по наименьшему RMSE
    valid_results = results_df.dropna(subset=['RMSE'])
    if len(valid_results) == 0:
        raise ValueError("Ни одна модель не была успешно обучена")

    best_name = valid_results.loc[valid_results['RMSE'].idxmin(), 'Модель']
    best_model = trained[best_name]
    print(f"\nЛучшая модель: {best_name} (минимальный RMSE)")

    # Кросс-валидация финальной (лучшей) модели с cv=5
    # это проверяет стабильность модели на разных частях датасета
    print(f"\nКросс-валидация для '{best_name}' (cv=5):")
    try:
        cv_scores = cross_val_score(best_model, X, y,cv=5, scoring='r2')
        print(f"R² по фолдам: {np.round(cv_scores, 4)}")
        print(f"Среднее R²  : {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    except Exception as e:
        print(f"Кросс-валидация не выполнена: {e}")

    # подготовка данных о важности признаков для исследователя
    # для лесов используется 'feature_importances_', для линейных - 'coef_'
    if hasattr(best_model, 'feature_importances_'):
        importance = best_model.feature_importances_
    elif hasattr(best_model, 'coef_'):
        importance = np.abs(best_model.coef_)
    else:
        importance = np.zeros(len(feature_names))

    feature_importance_df = pd.DataFrame({
        'Признак': feature_names,
        'Важность': np.round(importance, 6)
    }).sort_values('Важность', ascending=False)

    print("\nВажность признаков:")
    print(feature_importance_df.to_string(index=False))

    return (
        best_model,
        best_name,
        X_train, X_test, y_train, y_test,
        results_df,
        feature_importance_df
    )