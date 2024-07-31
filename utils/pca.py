import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Charger les données à partir du fichier texte
file_path = 'output.txt'
data = pd.read_csv(file_path, sep=" ", header=None)

# Séparer les caractéristiques (X) et la variable cible (y)

# print(data)
# exit()

data = data.iloc[:, :-1]

X = data.iloc[:, :-1]


y = data.iloc[:, -1]



# Standardiser les données (centrer et réduire)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Effectuer la PCA
pca = PCA(n_components=3)  # Vous pouvez choisir le nombre de composantes principales
X_pca = pca.fit_transform(X) #pca.fit_transform(X_scaled)

# Afficher la variance expliquée par chaque composante principale
explained_variance = pca.explained_variance_ratio_
print("Variance expliquée par chaque composante principale:", explained_variance)

# Visualiser les résultats de la PCA
plt.figure(figsize=(8,6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', edgecolor='k', s=100)
plt.xlabel('Première composante principale')
plt.ylabel('Deuxième composante principale')
plt.title('PCA des données de trade')
plt.colorbar(label='Succès du trade (1=réussi, 0=non réussi)')
plt.savefig('pca_trades.png')  # Vous pouvez spécifier le chemin et le nom de fichier que vous souhaitez
plt.close()
