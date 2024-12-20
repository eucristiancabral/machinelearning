import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import seaborn as sns
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Carregar o dataset
penguins_df = pd.read_csv("C:/Users/Cliente/OneDrive/Documentos/Cristian/Programação/provapratica-cristian/clusterização/penguins.csv")

# Exibindo as primeiras linhas
print(penguins_df.head())

# Remover linhas com valores ausentes em qualquer coluna
penguins_df = penguins_df.dropna()

# Seleção de variáveis
# Vamos assumir que a variável alvo é 'sex' (binária, 0 para FEMALE e 1 para MALE)
penguins_df['sex'] = penguins_df['sex'].map({'FEMALE': 0, 'MALE': 1})

# Remover linhas onde 'sex' é NaN (em caso de valores ausentes na coluna alvo)
penguins_df = penguins_df.dropna(subset=['sex'])

X = penguins_df.drop(columns=['sex'])  # Remover a coluna alvo 'sex'
y = penguins_df['sex']  # Variável alvo binária

# Convertendo variáveis categóricas para variáveis numéricas (One-Hot Encoding)
X = pd.get_dummies(X, drop_first=True)

# Dividir os dados em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar as variáveis
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Criar e treinar o modelo de Regressão Logística
logreg = LogisticRegression(random_state=42)
logreg.fit(X_train_scaled, y_train)

# Fazer previsões
y_pred = logreg.predict(X_test_scaled)
y_prob = logreg.predict_proba(X_test_scaled)[:, 1]  # Probabilidade da classe 1

# Avaliar o modelo
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Exibir os resultados
print(f"Acurácia: {accuracy:.4f}")
print("Matriz de Confusão:")
print(conf_matrix)
print("Relatório de Classificação:")
print(class_report)

# Gráfico da Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curva ROC')
plt.legend(loc='lower right')
plt.show()

# Gráfico da Matriz de Confusão
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Predito: 0', 'Predito: 1'], yticklabels=['Real: 0', 'Real: 1'])
plt.ylabel('Real')
plt.xlabel('Predito')
plt.title('Matriz de Confusão')
plt.show()

# Gráfico de Distribuição das Previsões
plt.figure(figsize=(8, 6))
plt.hist(y_pred, bins=2, alpha=0.7, color='blue', edgecolor='black')
plt.title('Distribuição das Previsões')
plt.xlabel('Classe Predita')
plt.ylabel('Frequência')
plt.xticks([0, 1], ['Classe 0', 'Classe 1'])
plt.show()
