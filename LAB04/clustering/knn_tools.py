from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def calculate_wcss(X, max_k=10):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    wcss = []
    for i in range(1, max_k + 1):
        kmeans = KMeans(n_clusters=i, n_init=10, random_state=42)
        kmeans.fit(X_scaled)
        wcss.append(kmeans.inertia_)
    return wcss