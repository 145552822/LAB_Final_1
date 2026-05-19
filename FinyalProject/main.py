# main.py запускает всё. Ничего по отдельности не включается.

import os

# устанавливаем рабочую папку
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from data_loader   import load_data
from eda           import run_eda
from preprocessing import preprocess
from models        import train_and_evaluate
from evaluate      import evaluate_best_model
from predictor     import predict_interactive

if __name__ == '__main__':

    # загрузка датасета
    df = load_data('Taxi Price 1.csv')

    # разведочный анализ данных (EDA). графики сохраняются в plots/
    run_eda(df)

    # предобработка данных (очистка, кодирование, нормализация, выделение признаков и цели)
    X, y, feature_names = preprocess(df)

    # обучение и сравнение моделей
    (best_model, best_name,
     X_train, X_test, y_train, y_test,
     results_df, feature_importance_df) = train_and_evaluate(X, y, feature_names)

    # финальная оценка лучшей модели (predicted vs actual, feature importance, learning curve)
    evaluate_best_model(
        best_model, best_name,
        X_train, X_test, y_train, y_test,
        feature_importance_df
    )

    print(f"\nЛучшая модель: {best_name}")

    # интерактивное предсказание цены по введённым факторам
    predict_interactive(best_model, feature_names, df)