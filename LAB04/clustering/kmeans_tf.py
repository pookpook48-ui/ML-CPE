from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def train_kmeans(X, n_clusters=3):
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
    kmeans.fit(X_scaled)
    return kmeans, scaler, X_scaled