import math
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import silhouette_score
from sklearn.metrics import davies_bouldin_score
import random

def euclidean_distance(p1, p2):
    '''
    Computes the Euclidean distance between two points
    Inputs: p1,p2 coordinates about two points
    Outputs: Euclidean distance between p1 and p2
    '''
    return math.sqrt(sum((float(x) - float(y)) ** 2 for x, y in zip(p1, p2)))


def closest_centroid(point, centroids):
    '''
        Find the nearest centroids to a given point
        Inputs:
            point: coordinates about a point
            centroids: list of coordinates of centroids
        Outputs: the closest centroid for the given point
        '''
    distances = [euclidean_distance(point, centroid) for centroid in centroids]
    return distances.index(min(distances))


def kmeans(data, k, max_iterations=100, tolerance=1e-4):
    """
    Implementazione dell'algoritmo K-Means usando PySpark.

    Args:
        data: RDD di punti (ogni punto è una lista o tupla di coordinate).
        k: Numero di cluster.
        max_iterations: Numero massimo di iterazioni.
        tolerance: Soglia per la convergenza dei centroidi.

    Returns:
        centroidi_finali: Lista dei centroidi finali.
        assegnazioni: RDD contenente i cluster di appartenenza di ogni punto.
    """
    # Step 1: Inizializza i centroidi scegliendo casualmente k punti dal dataset
    centroids = data.takeSample(False, k)

    for i in range(max_iterations):
        # Step 2: Assegna ogni punto al centroide più vicino
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Step 3: Calcola i nuovi centroidi
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
            .collect()
        )

        # Converte i nuovi centroidi in una lista ordinata
        new_centroids = sorted(new_centroids, key=lambda x: x[0])
        new_centroids = [centroid[1] for centroid in new_centroids]

        # Step 4: Controlla la convergenza
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        centroids = new_centroids  # Aggiorna i centroidi

    # Ritorna i centroidi finali e le assegnazioni
    assignments = data.map(lambda point: (closest_centroid(point, centroids), point))
    return centroids, assignments


def kmeans_plus_plus(data, k, max_iterations=100, tolerance=1e-3):
    """
    Implement K-Means++ using PySpark's RDD.

    Args:
        data: RDD of points (each point is a list or tuple of coordinates).
        k: Number of clusters.
        max_iterations: Maximum number of iterations.
        tolerance: Convergence tolerance for centroid change.

    Returns:
        centroids: Final list of centroids.
        assignments: RDD containing the cluster assignments of each point.
    """

    # Step 1: Randomly pick the first centroid
    centroids = [data.takeSample(False, 1)[0]]

    # Step 2: Select the remaining k-1 centroids using a distance-based probability distribution
    for _ in range(k - 1):
        distances = data.map(lambda point: min([euclidean_distance(point, c) for c in centroids]))
        total_distance = distances.reduce(lambda a, b: a + b)
        probabilities = distances.map(lambda d: d ** 2 / total_distance)

        # Choose the next centroid based on these probabilities
        r = random.random()
        cumulative_prob = 0.0
        for point, prob in zip(data.collect(), probabilities.collect()):
            cumulative_prob += prob
            if cumulative_prob >= r:
                centroids.append(point)
                break

    # Step 5: Proceed with standard K-means algorithm (assignment, update centroids)
    for i in range(max_iterations):
        # Step 6: Assign points to the nearest centroid
        clusters = data.map(lambda point: (closest_centroid(point, centroids), (point, 1)))

        # Step 7: Recompute the centroids
        new_centroids = (
            clusters
            .reduceByKey(lambda x, y: ([a + b for a, b in zip(x[0], y[0])], x[1] + y[1]))
            .mapValues(lambda x: [a / x[1] for a in x[0]])
            .collect()
        )

        # Convert the new centroids into a list sorted by the cluster index
        new_centroids = sorted(new_centroids, key=lambda x: x[0])
        new_centroids = [centroid[1] for centroid in new_centroids]

        # Step 8: Check for convergence (if centroids don't change significantly)
        if all(euclidean_distance(c1, c2) < tolerance for c1, c2 in zip(centroids, new_centroids)):
            break

        centroids = new_centroids  # Update centroids

    # Step 9: Return final centroids and the assignments
    assignments = data.map(lambda point: (closest_centroid(point, centroids), point))
    return centroids, assignments

def plot_kmeans_clusters(centroids, assignments):
    """
    Crea uno scatterplot per visualizzare i cluster e i centroidi.

    Args:
        centroids: Lista dei centroidi finali.
        assignments: RDD contenente i punti con i rispettivi cluster.
    """
    # Estrai i punti e le assegnazioni dal driver
    points = assignments.map(lambda x: x[1]).collect()  # Raccogli i punti nel driver
    labels = assignments.map(lambda x: x[0]).collect()  # Raccogli le etichette nel driver

    # Converti i punti in un array NumPy per facilitarne la manipolazione
    points = np.array(points)

    # Creazione dello scatterplot
    plt.figure(figsize=(8, 6))

    # Aggiungi ogni punto con un colore in base al cluster
    scatter = plt.scatter(points[:, 0], points[:, 1], c=labels, cmap='viridis', s=50)

    # Aggiungi i centroidi al grafico
    centroids = np.array(centroids)
    plt.scatter(centroids[:, 0], centroids[:, 1], c='red', marker='x', s=200, label='Centroidi')

    # Aggiungi una legenda e titoli
    plt.title('K-Means Clustering')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend()

    # Mostra il grafico
    plt.colorbar(scatter)
    plt.show()


def calculate_ari(y_true, y_pred):
    """
    Calculate the Adjusted Rand Index (ARI) between the true labels and predicted labels.

    Parameters:
    - y_true: True cluster labels (array-like)
    - y_pred: Predicted cluster labels from K-means (array-like)

    Returns:
    - ARI score (float)
    """
    ari_score = adjusted_rand_score(y_true, y_pred)
    return ari_score


def calculate_silhouette_score(X, y_pred):
    """
    Calculate the Silhouette Score for the clustering results.

    Parameters:
    - X: Feature matrix (array-like or DataFrame)
    - y_pred: Predicted cluster labels from K-means (array-like)

    Returns:
    - Silhouette Score (float)
    """
    silhouette_avg = silhouette_score(X, y_pred)
    return silhouette_avg


def calculate_davies_bouldin_score(X, y_pred):
    """
    Calculate the Davies-Bouldin Index (DB) for the clustering results.

    Parameters:
    - X: Feature matrix (array-like or DataFrame)
    - y_pred: Predicted cluster labels from K-means (array-like)

    Returns:
    - Davies-Bouldin Index (float)
    """
    db_score = davies_bouldin_score(X, y_pred)
    return db_score

