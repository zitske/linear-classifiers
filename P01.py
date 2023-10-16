# Importar as bibliotecas necessárias
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Carregar o dataset Iris
iris = datasets.load_iris()
X = iris.data[:, :2]
y = iris.target

# Dividir o dataset em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Plotar o dataset em um gráfico
plt.figure(figsize=(10, 6))
plt.scatter(X[y == 0][:, 0], X[y == 0][:, 1], label="Classe 0", c="r", marker="o")
plt.scatter(X[y == 1][:, 0], X[y == 1][:, 1], label="Classe 1", c="g", marker="s")
plt.scatter(X[y == 2][:, 0], X[y == 2][:, 1], label="Classe 2", c="b", marker="x")
plt.xlabel("Característica 1")
plt.ylabel("Característica 2")
plt.legend()
plt.title("Dataset Iris")
plt.show()

# Treinar um classificador linear (Regresão Logística)
classifier = LogisticRegression(max_iter=10000, multi_class="auto")
classifier.fit(X_train, y_train)

# Fazer previsões no conjunto de teste
y_pred = classifier.predict(X_test)

# Calcular as métricas de avaliação
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted")
recall = recall_score(y_test, y_pred, average="weighted")
f1 = f1_score(y_test, y_pred, average="weighted")

# Imprimir as métricas de avaliação
print("Acurácia:", accuracy)
print("Precisão:", precision)
print("Recall:", recall)
print("F1 Score:", f1)
