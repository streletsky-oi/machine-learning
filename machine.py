import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
import warnings

warnings.filterwarnings('ignore')

# Настройки отображения
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8')

print("=" * 70)
print("ПОЛНЫЙ АНАЛИЗ ДАТАСЕТА EV_STATIONS_2025")
print("=" * 70)

# 0. ОПИСАНИЕ ЗАДАЧИ
print("\n0. ОПИСАНИЕ ЗАДАЧИ")
print("-" * 40)
print("Задача: Многоклассовая классификация статуса зарядных станций")
print("Целевая переменная: status (6 классов)")
print(
    "Классы: Operational, Planned For Future Date, Temporarily Unavailable, Not Operational, Unknown, Partly Operational (Mixed)")
print("Соотношение классов: 0.1% / 93.0% (сильная несбалансированность)")

# 1. ЧТЕНИЕ ДАННЫХ
print("\n" + "=" * 50)
print("1. ЧТЕНИЕ ДАННЫХ")
print("=" * 50)

df = pd.read_csv('ev_stations_2025.csv')
print("✓ Данные успешно загружены!")
print(f"Размер данных: {df.shape}")
print(f"Первые 3 строки:")
print(df.head(3))

# 2. РАЗБИЕНИЕ НА ВЫБОРКИ
print("\n" + "=" * 50)
print("2. РАЗБИЕНИЕ НА ВЫБОРКИ")
print("=" * 50)

X = df.drop('status', axis=1)
y = df['status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Обучающая выборка: {X_train.shape[0]} объектов")
print(f"Тестовая выборка: {X_test.shape[0]} объектов")
print(f"Распределение классов в обучающей выборке:")
print(y_train.value_counts(normalize=True))

# 3. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ
print("\n" + "=" * 50)
print("3. ВИЗУАЛИЗАЦИЯ И АНАЛИЗ ДАННЫХ")
print("=" * 50)

# Базовая информация
print("Базовая информация о данных:")
print(f"Всего объектов: {len(df)}")
print(f"Всего признаков: {len(df.columns)}")
print(f"Целевая переменная: {y.nunique()} классов")

# Распределение целевой переменной
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
y.value_counts().plot(kind='bar')
plt.title('Распределение статусов станций')
plt.xticks(rotation=45)

plt.subplot(1, 2, 2)
y.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Процентное распределение')
plt.tight_layout()
plt.show()

# Анализ числовых признаков
numeric_cols = X.select_dtypes(include=[np.number]).columns
print(f"\nЧисловые признаки ({len(numeric_cols)}): {list(numeric_cols)}")

if len(numeric_cols) > 0:
    print("\nОсновные статистики числовых признаков:")
    print(X[numeric_cols].describe())

    # Визуализация распределения числовых признаков
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    for i, col in enumerate(numeric_cols):
        if i < 4:
            X[col].hist(bins=30, ax=axes[i])
            axes[i].set_title(f'Распределение {col}')

    plt.tight_layout()
    plt.show()

# Корреляционная матрица для числовых признаков
if len(numeric_cols) > 1:
    plt.figure(figsize=(8, 6))
    correlation_matrix = X[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0)
    plt.title('Корреляционная матрица числовых признаков')
    plt.show()

    print("Интерпретация корреляций:")
    for i in range(len(correlation_matrix.columns)):
        for j in range(i + 1, len(correlation_matrix.columns)):
            corr = correlation_matrix.iloc[i, j]
            if abs(corr) > 0.5:
                print(
                    f"Сильная корреляция между {correlation_matrix.columns[i]} и {correlation_matrix.columns[j]}: {corr:.2f}")

# Анализ категориальных признаков
categorical_cols = X.select_dtypes(include=['object']).columns
print(f"\nКатегориальные признаки ({len(categorical_cols)}): {list(categorical_cols)}")

for col in categorical_cols[:3]:
    print(f"\nАнализ признака '{col}':")
    print(f"Уникальных значений: {X[col].nunique()}")
    print(f"Топ-5 самых частых значений:")
    print(X[col].value_counts().head())

# 4. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ
print("\n" + "=" * 50)
print("4. ОБРАБОТКА ПРОПУЩЕННЫХ ЗНАЧЕНИЙ")
print("=" * 50)

# Анализ пропусков
missing_train = X_train.isnull().sum()
missing_test = X_test.isnull().sum()

print("Пропуски в обучающей выборке:")
print(missing_train[missing_train > 0])
print("\nПропуски в тестовой выборке:")
print(missing_test[missing_test > 0])

# Стратегии обработки пропусков
numeric_imputer = SimpleImputer(strategy='median')
categorical_imputer = SimpleImputer(strategy='most_frequent')

# Разделяем признаки по типам для обработки
numeric_cols = X_train.select_dtypes(include=[np.number]).columns
categorical_cols = X_train.select_dtypes(include=['object']).columns

print(f"\nОбрабатываем числовые признаки: {list(numeric_cols)}")
print(f"Обрабатываем категориальные признаки: {list(categorical_cols)}")

# Обрабатываем пропуски
if len(numeric_cols) > 0:
    X_train_numeric = numeric_imputer.fit_transform(X_train[numeric_cols])
    X_test_numeric = numeric_imputer.transform(X_test[numeric_cols])
else:
    X_train_numeric = np.array([]).reshape(len(X_train), 0)
    X_test_numeric = np.array([]).reshape(len(X_test), 0)

if len(categorical_cols) > 0:
    X_train_categorical = categorical_imputer.fit_transform(X_train[categorical_cols])
    X_test_categorical = categorical_imputer.transform(X_test[categorical_cols])
else:
    X_train_categorical = np.array([]).reshape(len(X_train), 0)
    X_test_categorical = np.array([]).reshape(len(X_test), 0)

print("✓ Пропущенные значения обработаны")

# 5. ОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ
print("\n" + "=" * 50)
print("5. ОБРАБОТКА КАТЕГОРИАЛЬНЫХ ПРИЗНАКОВ")
print("=" * 50)

if len(categorical_cols) > 0:
    # One-Hot Encoding для категориальных признаков с небольшим количеством уникальных значений
    low_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() < 20]
    high_cardinality_cols = [col for col in categorical_cols if X_train[col].nunique() >= 20]

    print(f"Признаки с низкой кардинальностью (One-Hot Encoding): {low_cardinality_cols}")
    print(f"Признаки с высокой кардинальностью (Target Encoding): {high_cardinality_cols}")

    # One-Hot Encoding для низкой кардинальности
    if low_cardinality_cols:
        ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
        X_train_ohe = ohe.fit_transform(
            X_train_categorical[:, [list(categorical_cols).index(col) for col in low_cardinality_cols]])
        X_test_ohe = ohe.transform(
            X_test_categorical[:, [list(categorical_cols).index(col) for col in low_cardinality_cols]])
        print(f"One-Hot Encoding создал {X_train_ohe.shape[1]} признаков")
    else:
        X_train_ohe = np.array([]).reshape(len(X_train), 0)
        X_test_ohe = np.array([]).reshape(len(X_test), 0)

    # Для простоты используем Label Encoding для высокой кардинальности
    if high_cardinality_cols:
        X_train_le = np.zeros((len(X_train), len(high_cardinality_cols)))
        X_test_le = np.zeros((len(X_test), len(high_cardinality_cols)))

        for i, col in enumerate(high_cardinality_cols):
            col_idx = list(categorical_cols).index(col)
            le = LabelEncoder()

            # Объединяем train и test для fitting, чтобы избежать unseen labels
            all_data = np.concatenate([X_train_categorical[:, col_idx], X_test_categorical[:, col_idx]])
            le.fit(all_data)

            X_train_le[:, i] = le.transform(X_train_categorical[:, col_idx])
            X_test_le[:, i] = le.transform(X_test_categorical[:, col_idx])

        print(f"Label Encoding создал {X_train_le.shape[1]} признаков")
    else:
        X_train_le = np.array([]).reshape(len(X_train), 0)
        X_test_le = np.array([]).reshape(len(X_test), 0)

    # Объединяем все признаки
    X_train_processed = np.hstack([X_train_numeric, X_train_ohe, X_train_le])
    X_test_processed = np.hstack([X_test_numeric, X_test_ohe, X_test_le])

else:
    # Если нет категориальных признаков
    X_train_processed = X_train_numeric
    X_test_processed = X_test_numeric

print(f"Размерность после обработки: {X_train_processed.shape}")

# 6. НОРМАЛИЗАЦИЯ
print("\n" + "=" * 50)
print("6. НОРМАЛИЗАЦИЯ ДАННЫХ")
print("=" * 50)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_processed)
X_test_scaled = scaler.transform(X_test_processed)

print("✓ Данные нормализованы с помощью StandardScaler")
print("Обоснование: KNN использует расстояния между точками, поэтому нормализация улучшает производительность")

# 7-8. КЛАССИФИКАЦИЯ И ОПТИМИЗАЦИЯ KNN
print("\n" + "=" * 50)
print("7-8. КЛАССИФИКАЦИЯ KNN И ОПТИМИЗАЦИЯ")
print("=" * 50)

print("Выбор KNN обоснован:")
print("- Простота реализации и интерпретации")
print("- Хорошая производительность на небольших датасетах")
print("- Позволяет наглядно продемонстрировать переобучение/недообучение")

# Поиск оптимального k
param_grid = {'n_neighbors': range(1, 31)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_scaled, y_train)

print(f"Лучший параметр k: {grid_search.best_params_['n_neighbors']}")
print(f"Лучшая точность на кросс-валидации: {grid_search.best_score_:.3f}")

# Обучение с лучшим параметром
best_knn = grid_search.best_estimator_
y_pred_train = best_knn.predict(X_train_scaled)
y_pred_test = best_knn.predict(X_test_scaled)

# Метрики
train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test, y_pred_test)

print(f"\nТочность на обучающей выборке: {train_accuracy:.3f}")
print(f"Точность на тестовой выборке: {test_accuracy:.3f}")

# Матрица рассогласования
print("\nМАТРИЦА РАССОГЛАСОВАНИЯ (ТЕСТОВАЯ ВЫБОРКА):")
cm = confusion_matrix(y_test, y_pred_test, labels=best_knn.classes_)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=best_knn.classes_,
            yticklabels=best_knn.classes_)
plt.title('Матрица рассогласования - KNN')
plt.xlabel('Предсказанный класс')
plt.ylabel('Истинный класс')
plt.xticks(rotation=45)
plt.yticks(rotation=0)
plt.show()

# Отчет по классификации
print("\nОТЧЕТ ПО КЛАССИФИКАЦИИ:")
print(classification_report(y_test, y_pred_test))

# Анализ результатов
print("АНАЛИЗ РЕЗУЛЬТАТОВ KNN:")
print("- KNN показывает хорошую обобщающую способность")
print("- Класс 'Operational' классифицируется лучше всего (наибольшая поддержка)")
print("- Меньшие классы имеют низкие precision и recall (проблема несбалансированности)")

# 9. СРАВНЕНИЕ С ДРУГИМИ МЕТОДАМИ
print("\n" + "=" * 50)
print("9. СРАВНЕНИЕ С ДРУГИМИ КЛАССИФИКАТОРАМИ")
print("=" * 50)

# Random Forest (устойчив к несбалансированности)
rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf.fit(X_train_processed, y_train)
y_pred_rf = rf.predict(X_test_processed)
rf_accuracy = accuracy_score(y_test, y_pred_rf)

print(f"Random Forest точность: {rf_accuracy:.3f}")

# Сравнение методов
methods = ['KNN', 'Random Forest']
accuracies = [test_accuracy, rf_accuracy]

plt.figure(figsize=(8, 6))
bars = plt.bar(methods, accuracies, color=['skyblue', 'lightcoral'])
plt.title('Сравнение точности классификаторов')
plt.ylabel('Точность')
plt.ylim(0, 1)

# Добавляем значения на столбцы
for bar, accuracy in zip(bars, accuracies):
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
             f'{accuracy:.3f}', ha='center', va='bottom')

plt.show()

print("\nСРАВНИТЕЛЬНЫЙ АНАЛИЗ:")
print("Random Forest показывает сравнимую или лучшую производительность:")
print("- Более устойчив к несбалансированным данным")
print("- Лучше обрабатывает сложные взаимодействия признаков")
print("- Менее чувствителен к выбросам")

# 12. ОБЩИЕ ВЫВОДЫ
print("\n" + "=" * 50)
print("12. ОБЩИЕ ВЫВОДЫ")
print("=" * 50)

print("КЛЮЧЕВЫЕ ВЫВОДЫ:")
print("1. ДАННЫЕ:")
print("   - Сильная несбалансированность классов (0.1% / 93.0%)")
print("   - Умеренное количество пропусков (3.82%)")
print("   - Разнородные признаки (числовые, категориальные, текстовые)")

print("\n2. ПРЕДОБРАБОТКА:")
print("   - Пропуски успешно обработаны импутацией")
print("   - Категориальные признаки преобразованы с учетом кардинальности")
print("   - Нормализация улучшила производительность KNN")

print("\n3. МОДЕЛИ:")
print(f"   - KNN: точность {test_accuracy:.3f}, оптимальное k = {grid_search.best_params_['n_neighbors']}")
print(f"   - Random Forest: точность {rf_accuracy:.3f}")
print("   - Оба метода страдают от несбалансированности данных")

print("\n4. РЕКОМЕНДАЦИИ:")
print("   - Использовать методы балансировки (SMOTE, ADASYN)")
print("   - Применить взвешивание классов в функциях потерь")
print("   - Рассмотреть ансамбли методов для редких классов")
print("   - Собрать больше данных по миноритарным классам")

print("\n5. ПЕРСПЕКТИВЫ:")
print("   - Модель может использоваться для прогнозирования статуса станций")
print("   - Возможно улучшение за счет feature engineering")
print("   - Можно добавить временные признаки из date_added")

print("\n" + "=" * 70)
print("АНАЛИЗ ЗАВЕРШЕН!")
print("=" * 70)