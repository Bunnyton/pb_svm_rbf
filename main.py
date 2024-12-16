#!/bin/python3

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Определение RBF Kernel в виде PyTorch-функции
def rbf_kernel(x1, x2, gamma=0.1):
    """RBF Kernel на PyTorch."""
    return torch.exp(-gamma * torch.norm(x1 - x2, dim=1) ** 2)

# Реализация обучения SVM с ядром RBF на PyTorch
class SVM_RBF(nn.Module):
    def __init__(self, n_features, C=0.1, gamma=0.1):
        super(SVM_RBF, self).__init__()
        self.C = C
        self.gamma = gamma
        self.alpha = nn.Parameter(torch.zeros(n_features, requires_grad=True))
        self.b = nn.Parameter(torch.tensor(0.0, requires_grad=True))

    def forward(self, X, y):
        """Вычисляет функцию решения."""
        kernel_matrix = torch.exp(-self.gamma * torch.cdist(X, X) ** 2)
        decision = torch.sum(self.alpha * y * kernel_matrix, dim=1) + self.b
        return decision

    def loss(self, X, y):
        """Функция потерь для SVM."""
        decision = self.forward(X, y)
        hinge_loss = torch.clamp(1 - y * decision, min=0)
        return torch.mean(hinge_loss) + 0.5 * torch.sum(self.alpha ** 2)

# Функция для обучения одной SVM модели на GPU
def train_svm_rbf(X, y, C=0.1, gamma=0.1, tol=1e-5, max_iter=200):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)

    model = SVM_RBF(X.shape[0], C=C, gamma=gamma).to(device)
    optimizer = optim.Adam([model.alpha, model.b], lr=0.01)

    for iter_count in range(max_iter):
        optimizer.zero_grad()
        loss = model.loss(X, y)
        loss.backward()
        optimizer.step()

        # Проверка на сходимость
        if loss.item() < tol:
            print(f"Convergence reached after {iter_count} iterations.")
            break

        if iter_count % 10 == 0:
            print(f"Iteration {iter_count}, Loss: {loss.item():.6f}")

    w = (model.alpha * y) @ X  # Вычисление весов
    b = model.b.item()
    return w.cpu().detach().numpy(), b

# Функция для сохранения весов в файл
def save_weights(cls, w, b, directory="weights"):
    """Сохранение весов и смещения в файл."""
    os.makedirs(directory, exist_ok=True)
    file_path = os.path.join(directory, f"class_{cls}_weights.txt")

    with open(file_path, "w") as f:
        f.write(f"Weights: {w.tolist()}\n")
        f.write(f"Bias: {b}\n")

    print(f"Weights and bias for class {cls} saved to {file_path}")

# Функция для чтения весов из файла
def load_weights(cls, directory="weights"):
    """Загрузка весов и смещения из файла."""
    file_path = os.path.join(directory, f"class_{cls}_weights.txt")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")

    with open(file_path, "r") as f:
        lines = f.readlines()
        w = np.array(eval(lines[0].split(": ")[1]))
        b = float(lines[1].split(": ")[1])

    print(f"Weights and bias for class {cls} loaded from {file_path}")
    return w, b

# Обучение модели SVM для одного класса
def train_single_model(cls, X, y, C, gamma, tol, max_iter):
    """Обучаем один SVM-классификатор для конкретного класса."""
    print(f"Training SVM for class {cls}")
    y_binary = np.where(y == cls, 1, -1)

    try:
        w, b = load_weights(cls)
        print(f"Continuing training for class {cls}")
    except FileNotFoundError:
        w, b = None, None

    w, b = train_svm_rbf(X, y_binary, C, gamma, tol, max_iter)
    save_weights(cls, w, b)  # Сохраняем веса после тренировки
    return cls, (w, b)

# Реализация стратегии One-vs-All
def one_vs_all_train(X, y, C=0.1, gamma=0.1, tol=1e-5, max_iter=5000, directory="weights"):
    """
    Реализация стратегии One-vs-All (OvA): загружаем веса, если они существуют,
    и продолжаем тренировать или начинаем заново.
    """
    classes = np.unique(y)
    models = {}

    print(f"Training models for classes: {classes}")
    for cls in classes:
        print(f"Currently training class: {cls}")
        cls, model = train_single_model(cls, X, y, C, gamma, tol, max_iter)
        models[cls] = model

    return models, classes  # Возвращаем все модели и список классов

# Предсказание классов
def one_vs_all_predict(X, models, classes):
    """
    Предсказание классов на основе обученных моделей One-vs-All.
    """
    scores = {cls: np.dot(X, w) + b for cls, (w, b) in models.items()}
    scores = np.vstack([scores[cls] for cls in classes]).T
    indices = np.argmax(scores, axis=1)
    return classes[indices]  # Возвращаем соответствующие классы

# Загрузка моделей из файлов
def load_models(classes, directory="weights"):
    """Загрузка всех моделей для One-vs-All."""
    models = {}
    for cls in classes:
        w, b = load_weights(cls, directory)
        models[cls] = (w, b)
    return models

# Загрузка данных
wine = pd.read_csv('WineQT.csv')
print(wine)

# Определите X и y
X = wine.iloc[:, :-1].values  # Все столбцы кроме последнего
y = wine.iloc[:, -1].values   # Последний столбец

# Нормализация данных
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Разделение данных на тренировочную и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Обучение модели
print("Starting training...")
models, classes = one_vs_all_train(X_train, y_train, C=0.5, gamma=0.5)

# Предсказание
print("Making predictions on the test set...")
y_pred = one_vs_all_predict(X_test, models, classes)
test_accuracy = np.mean(y_pred == y_test)

# Результаты
print(f"Точность на тестовой выборке (custom SVM): {test_accuracy:.2f}")

# Загрузка моделей из файлов и предсказание
print("Loading models from files...")
loaded_models = load_models(classes)
print("Making predictions with loaded models...")
y_pred_loaded = one_vs_all_predict(X_test, loaded_models, classes)
loaded_accuracy = np.mean(y_pred_loaded == y_test)
print(f"Точность на тестовой выборке (loaded models): {loaded_accuracy:.2f}")

