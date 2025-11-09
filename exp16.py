import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering

# Load the dataset
try:
    iris = pd.read_csv("IRIS.csv")
except FileNotFoundError:
    print("Error: IRIS.csv not found.")
    print("Please make sure 'IRIS.csv' is in the same folder as this script.")
    exit()


x = iris.iloc[:,:-1].values
y = iris.iloc[:, -1].values

# --- Note: train_test_split is not used in this script ---
# This is okay, as clustering is often performed on the whole dataset
# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2)

# --- Hierarchical Clustering ---
# You correctly create the model and fit it to the data
hc = AgglomerativeClustering(n_clusters = 3)
y_pred = hc.fit_predict(x) # 'y_pred' holds the cluster labels (0, 1, or 2) for each point

# --- FIGURE 1: Scatter Plot of Clusters ---
plt.figure(1) # Create the first figure window
plt.title("Hierarchical Clusters (Sepal Dimensions)")

# --- FIX ---
# The code below was using 'kmeans', but your variable is 'y_pred'
# I have replaced 'kmeans' with 'y_pred'
plt.scatter(x[y_pred == 0, 0], x[y_pred == 0, 1], s=50, c='red', label='Cluster 1 (Setosa?)')
plt.scatter(x[y_pred == 1, 0], x[y_pred == 1, 1], s=50, c='green', label='Cluster 2 (Versicolor?)')
plt.scatter(x[y_pred == 2, 0], x[y_pred == 2, 1], s=50, c='blue', label='Cluster 3 (Verginica?)')

# --- FIX ---
# This line was removed because 'AgglomerativeClustering' (your 'hc' model)
# does not have 'cluster_centers_' like KMeans does.
# plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s=150, c='yellow', label='Centroids')

plt.xlabel("Sepal Length")
plt.ylabel("Sepal Width")
plt.legend()


# --- FIGURE 2: Dendrogram ---
plt.figure(2) # Create a second, separate figure window
plt.title("Iris Dendrogram")
dendrogram(linkage(x, method='ward')) # 'ward' is a common method for linkage
plt.xlabel("Data Points")
plt.ylabel("Euclidean Distance")

# Show both plots
plt.show()