import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
url = "https://raw.githubusercontent.com/tirthajyoti/Machine-Learning-with-Python/master/Datasets/Mall_Customers.csv"
try:
    df = pd.read_csv(url)
    df.columns = df.columns.str.strip()
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading data: {e}")
    raise
features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
missing_features = [feature for feature in features if feature not in df.columns]
if missing_features:
    print(f"Missing required columns: {', '.join(missing_features)}")
    raise SystemExit(1)
X = df[features]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
kmeans = KMeans(n_clusters=3, init='k-means++', n_init='auto', random_state=42)
clusters = kmeans.fit_predict(X_scaled)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=clusters, palette='viridis', s=100, edgecolor='black')
plt.title('Customer Segmentation (PCA + K-Means)')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()
df['Cluster'] = clusters
print("\n--- Cluster Profiles ---")
print(df.groupby('Cluster')[features].mean())
