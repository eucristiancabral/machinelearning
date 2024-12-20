import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

# Leitura do arquivo
penguins_df = pd.read_csv("C:/Users/Cliente/OneDrive/Documentos/Cristian/Programação/provapratica-cristian/clusterização/penguins.csv")

# Exibindo as primeiras linhas
print(penguins_df.head())

# Boxplot para verificar os dados
penguins_df.boxplot()
plt.show()

# Removendo valores ausentes
penguins_df = penguins_df.dropna()

# Filtrando valores extremos da coluna 'flipper_length_mm'
penguins_df = penguins_df[(penguins_df["flipper_length_mm"] > 0) & (penguins_df["flipper_length_mm"] < 4000)]

# Verificando se os índices 9 e 14 existem no DataFrame
indices_to_drop = [9, 14]
existing_indices = [idx for idx in indices_to_drop if idx in penguins_df.index]

# Removendo apenas os índices que existem
penguins_clean = penguins_df.drop(existing_indices)

# Exibindo as primeiras linhas após limpeza
print(penguins_clean.head())

# Aplicando One-Hot Encoding
df = pd.get_dummies(penguins_clean).drop("sex_.", axis=1)
print(df.head())

# Normalizando os dados
scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)

# PCA para redução de dimensionalidade
pca = PCA(n_components=None)
dfx_pca = pca.fit(df_scaled)
print(dfx_pca.explained_variance_ratio_)

# Selecionando o número de componentes com base na variância explicada
n_components = sum(dfx_pca.explained_variance_ratio_ > 0.1)
pca = PCA(n_components=n_components)
penguins_PCA = pca.fit_transform(df_scaled)
print(f"Number of components: {n_components}")

# Método do cotovelo para escolher o número de clusters
inertia = []
for k in range(1, 10):
    kmeans = KMeans(n_clusters=k, random_state=42).fit(penguins_PCA)
    inertia.append(kmeans.inertia_)
plt.plot(range(1, 10), inertia, marker="o")
plt.xlabel("Number of clusters")
plt.ylabel("Inertia")
plt.title("Elbow Method")
plt.show()

# Definindo o número de clusters a ser usado
n_clusters = 4

# Aplicando K-means
kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(penguins_PCA)

# Visualizando os clusters
plt.scatter(penguins_PCA[:, 0], penguins_PCA[:, 1], c=kmeans.labels_, cmap="viridis")
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title(f"K-means Clustering (K={n_clusters})")
plt.show()
