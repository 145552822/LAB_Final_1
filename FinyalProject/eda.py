import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

def run_eda(df_raw):
    """
    Разведочный анализ данных (EDA).
    Строит минимум 5 графиков и сохраняет их в папку plots/.
    """

    # создание папки для сохранения графиков
    os.makedirs('plots', exist_ok=True)

    # предобработка для EDA: очистка пробелов
    df = df_raw.copy()
    df.columns = df.columns.str.strip()
    df = df.map(lambda x: x.strip() if isinstance(x, str) else x)

    # преобразование процентов в числа
    if 'Процент на телефоне' in df.columns:
        df['Процент на телефоне'] = (df['Процент на телефоне'].astype(str).str.replace('%', '', regex=False).str.replace(',', '.', regex=False).astype(float))

    # выбор числовых столбцов для анализа
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

    # **гистограмма** распределения целевой переменной 'Цена'
    plt.figure(figsize=(8, 5))
    sns.histplot(df['Цена'], bins=20, kde=True, color='steelblue')
    plt.title('Распределение целевой переменной: Цена поездки', fontsize=14)
    plt.xlabel('Цена (сом)')
    plt.ylabel('Частота')
    plt.tight_layout()
    plt.savefig('plots/01_hist_price.png', dpi=150, bbox_inches='tight')
    plt.close()

    # гистограммы всех числовых признаков
    if len(num_cols) > 1:
        fig, axes = plt.subplots(nrows=(len(num_cols) + 2) // 3, ncols=3, figsize=(15, max(4, 3 * ((len(num_cols) + 2) // 3))))
        axes = axes.flatten()
        for i, col in enumerate(num_cols):
            axes[i].hist(df[col].dropna(), bins=15, color='teal', edgecolor='white')
            axes[i].set_title(col, fontsize=10)
            axes[i].set_xlabel('')
        for j in range(i + 1, len(axes)):
            axes[j].set_visible(False)
        plt.suptitle('Распределение числовых признаков', fontsize=14, y=1.01)
        plt.tight_layout()
        plt.savefig('plots/02_hist_features.png', dpi=150, bbox_inches='tight')
        plt.close()

    # **Boxplot** - выявление выбросов по ключевым признакам
    box_cols = [c for c in ['Цена', 'Расстояние', 'Время поездки/мин', 'сом за км']
                if c in df.columns]
    if box_cols:
        plt.figure(figsize=(10, 6))
        df[box_cols].boxplot(
            patch_artist=True,
            boxprops=dict(facecolor='lightblue', color='navy'),
            medianprops=dict(color='red', linewidth=2)
        )
        plt.title('Диаграммы размаха (выявление выбросов)', fontsize=14)
        plt.xticks(rotation=15)
        plt.ylabel('Значение')
        plt.tight_layout()
        plt.savefig('plots/03_boxplot_outliers.png', dpi=150, bbox_inches='tight')
        plt.close()

    # **тепловая карта** корреляций между признаками
    if len(num_cols) >= 2:
        corr_matrix = df[num_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix,annot=True, fmt='.2f',cmap='coolwarm', center=0,linewidths=0.5, square=True,annot_kws={'size': 9})
        plt.title('Тепловая карта корреляций между числовыми признаками', fontsize=13)
        plt.tight_layout()
        plt.savefig('plots/04_heatmap_corr.png', dpi=150, bbox_inches='tight')
        plt.close()

    # **диаграмма** рассеяния - Цена vs Расстояние
    if 'Расстояние' in df.columns and 'Цена' in df.columns:
        plt.figure(figsize=(8, 5))
        plt.scatter(df['Расстояние'], df['Цена'],alpha=0.6, color='darkorange', edgecolors='white', s=60)
        # линия тренда
        z = np.polyfit(df['Расстояние'].dropna(), df['Цена'].dropna(), 1)
        p = np.poly1d(z)
        x_line = np.linspace(df['Расстояние'].min(), df['Расстояние'].max(), 100)
        plt.plot(x_line, p(x_line), 'r--', linewidth=2, label='Линия тренда')
        plt.title('Зависимость цены от расстояния', fontsize=14)
        plt.xlabel('Расстояние (км)')
        plt.ylabel('Цена (сом)')
        plt.legend()
        plt.tight_layout()
        plt.savefig('plots/05_scatter_price_distance.png', dpi=150, bbox_inches='tight')
        plt.close()

    # **Boxplot** цены по компаниям (категориальный анализ)
    if 'Компания' in df.columns and 'Цена' in df.columns:
        plt.figure(figsize=(9, 5))
        companies = df['Компания'].unique()
        data_by_company = [df[df['Компания'] == c]['Цена'].values for c in companies]
        plt.boxplot(data_by_company, labels=companies, patch_artist=True, boxprops=dict(facecolor='lightyellow', color='darkgreen'), medianprops=dict(color='red', linewidth=2))
        plt.title('Распределение цен по компаниям', fontsize=14)
        plt.xlabel('Компания')
        plt.ylabel('Цена (сом)')
        plt.xticks(rotation=15)
        plt.tight_layout()
        plt.savefig('plots/06_boxplot_by_company.png', dpi=150, bbox_inches='tight')
        plt.close()

    print("\nEDA завершён. Все графики сохранены в папку plots/")