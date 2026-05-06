import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve
import os

def evaluate_best_model(model, model_name, X_train, X_test, y_train, y_test, feature_importance_df):
    """
    Финальная оценка лучшей модели:
        Метрики на тестовой выборке
        График: предсказанные против фактических
        График: важность признаков
        График: кривая обучения
    """

    os.makedirs('plots', exist_ok=True)

    # финальное предсказание лучшей модели на тестовой выборке
    y_pred = model.predict(X_test)

    # расчёт итоговых метрик качества
    mae  = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2   = r2_score(y_test, y_pred)

    print(f"Финальная оценка лучшей модели: {model_name}")
    print(f"MAE = {mae:.4f}")
    print(f"RMSE = {rmse:.4f}")
    print(f"R² = {r2:.4f}")

    # сравнение предсказаний с реальностью
    plt.figure(figsize=(7, 6))
    plt.scatter(y_test, y_pred, alpha=0.7, color='mediumseagreen', edgecolors='white', s=70, label='Предсказания')
    # идеальная прямая (y = x)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Идеальная прямая')
    plt.xlabel('Реальная цена (сом)')
    plt.ylabel('Предсказанная цена (сом)')
    plt.title(f'Predicted vs Actual — {model_name}', fontsize=13)
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/07_predicted_vs_actual.png', dpi=150, bbox_inches='tight')
    plt.close()

    # важность признаков лучшей модели
    if feature_importance_df is not None and len(feature_importance_df) > 0:
        plt.figure(figsize=(9, max(5, len(feature_importance_df) * 0.45)))
        colors = sns.color_palette("viridis", len(feature_importance_df))
        plt.barh(feature_importance_df['Признак'], feature_importance_df['Важность'], color=colors)
        plt.xlabel('Важность признака')
        plt.title(f'Важность признаков — {model_name}', fontsize=13)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig('plots/08_feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # кривая обучения лучшей модели
    # Показывает, как изменяется качество при увеличении объёма данных
    import pandas as pd
    X_all = pd.concat([X_train, X_test])
    y_all = pd.concat([y_train, y_test]) if hasattr(y_train, 'reset_index') else np.concatenate([y_train, y_test])

    try:
        train_sizes, train_scores, val_scores = learning_curve(model, X_all, y_all, cv=5, scoring='r2', train_sizes=np.linspace(0.2, 1.0, 6), n_jobs=-1)

        train_mean = train_scores.mean(axis=1)
        val_mean   = val_scores.mean(axis=1)
        train_std  = train_scores.std(axis=1)
        val_std    = val_scores.std(axis=1)

        plt.figure(figsize=(8, 5))
        plt.plot(train_sizes, train_mean, 'o-', color='steelblue', label='Обучение (train)')
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color='steelblue')
        plt.plot(train_sizes, val_mean, 's-', color='darkorange', label='Валидация (CV)')
        plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color='darkorange')
        plt.xlabel('Размер обучающей выборки')
        plt.ylabel('R²')
        plt.title(f'Кривая обучения — {model_name}', fontsize=13)
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/09_learning_curve.png', dpi=150, bbox_inches='tight')
        plt.close()
    except Exception as e:
        print(f"Learning curve не построена: {e}")

    print("\nФинальная оценка завершена. Графики сохранены в папку plots/")