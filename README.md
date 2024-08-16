# Customer Segmentation & Clustering

---

## Project Overview

This project aims to perform customer segmentation using the K-Means clustering algorithm. The goal is to group mall customers into different segments based on their characteristics like age, income, and spending habits. These segments can then be used to tailor marketing strategies to target each customer group more effectively.

## Steps Involved in the Project

### 1. **Data Preparation**
   - **Data Loading**: The dataset, which includes details such as `CustomerID`, `Gender`, `Age`, `Annual Income (k$)`, and `Spending Score (1-100)`, is loaded from a CSV file.
   - **Data Cleaning**: Basic cleaning operations are performed to ensure that the data is in the correct format for analysis.

### 2. **Exploratory Data Analysis (EDA)**
   - **Statistical Summary**: Calculating basic statistics like mean, median, and standard deviation to understand the central tendencies and dispersion of the data.
   - **Distribution Plots**: Visualizing the distribution of key variables such as `Age`, `Annual Income`, and `Spending Score` to detect patterns, trends, or outliers.
   - **Univariate Analysis**: Analyzing each variable individually using distribution plots and summary statistics.
   - **Bivariate Analysis**: Exploring the relationships between two variables (e.g., `Annual Income` vs `Spending Score`) to understand how they interact.
   - **Multivariate Analysis**: Analyzing the interactions between multiple variables to reveal more complex patterns in customer behavior.

### 3. **Clustering Analysis**
   - **K-Means Clustering**: Implementing K-Means clustering on different sets of features (univariate, bivariate, multivariate) to identify customer segments.
   - **Elbow Method**: Used to determine the optimal number of clusters by plotting the inertia against the number of clusters.
   - **Silhouette Score**: Calculating and visualizing the silhouette score to evaluate the quality of clustering.

### 4. **Interpretation & Visualization**
   - **Cluster Visualization**: Visualizing the clusters using scatter plots, with centroids marked to represent the center of each cluster.
   - **Cluster Summary**: Summarizing the characteristics of each cluster to understand the profile of customers within each segment.
   - **Gender Distribution**: Analyzing the distribution of gender within each cluster using crosstabs.

### 5. **Final Evaluation**
   - **Silhouette Score Analysis**: A silhouette plot is used to visualize how well each point fits into its assigned cluster.
   - **Inertia and Silhouette Scores**: Experimenting with different numbers of clusters and evaluating them using both inertia and silhouette scores to determine the optimal clustering configuration.

## Detailed Steps and Code

### 1. Data Loading
```python
import pandas as pd
df = pd.read_csv("Mall_Customers.csv")
print(df.head())
```

### 2. Univariate Analysis
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Visualize distributions of key numerical features
columns = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)']
for col in columns:
    plt.figure()
    sns.distplot(df[col], kde=True, hist=True)
    plt.title(f'Distribution of {col}')
    plt.show()
```

### 3. Bivariate Analysis
```python
# Scatter plot to analyze the relationship between 'Annual Income' and 'Spending Score'
sns.scatterplot(data=df, x='Annual Income (k$)', y='Spending Score (1-100)', hue='Gender')
plt.title('Income vs Spending Score by Gender')
plt.show()
```

### 4. Clustering (Univariate)
```python
from sklearn.cluster import KMeans

# Initialize and fit KMeans for univariate clustering based on 'Annual Income'
clustering1 = KMeans(n_clusters=3)
clustering1.fit(df[['Annual Income (k$)']])
df['Income Cluster'] = clustering1.labels_
```

### 5. Elbow Method for Optimal Clusters
```python
# Inertia plot to determine the optimal number of clusters
inertia_scores = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(df[['Annual Income (k$)']])
    inertia_scores.append(kmeans.inertia_)

plt.figure()
plt.plot(range(1, 11), inertia_scores, marker='o')
plt.title('Elbow Method for Optimal Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.grid(True)
plt.show()
```

### 6. Clustering (Bivariate)
```python
clustering2 = KMeans(n_clusters=5)
clustering2.fit(df[['Annual Income (k$)', 'Spending Score (1-100)']])
df['Spending and Income Cluster'] = clustering2.labels_
```

### 7. Clustering (Multivariate)
```python
from sklearn.preprocessing import StandardScaler

# Standardize features before multivariate clustering
scale = StandardScaler()
dff = pd.get_dummies(df.drop(['CustomerID'], axis=1), drop_first=True)
scaled_features = scale.fit_transform(dff)
dff_scaled = pd.DataFrame(scaled_features, columns=dff.columns)

# Clustering on scaled data
kmeans3 = KMeans(n_clusters=5)
kmeans3.fit(dff_scaled)
dff['Multi Clusters'] = kmeans3.labels_

# Group by the new clusters and calculate the mean for each feature
cluster_summary = dff.groupby('Multi Clusters').mean()
print(cluster_summary)
```

### 8. Silhouette Score Analysis
```python
from sklearn.metrics import silhouette_score

# Calculate the silhouette score to evaluate the clustering performance
silhouette_avg = silhouette_score(dff_scaled, kmeans3.labels_)
print(f'Silhouette Score: {silhouette_avg}')
```

## Results and Analysis
- **Cluster Characteristics**: The clusters were identified based on similar characteristics, providing insights into the customer base. For instance, one cluster may represent high-income, high-spending customers, while another may represent low-income, low-spending customers.
- **Optimal Clusters**: The Elbow Method and Silhouette Score helped determine the optimal number of clusters, which was found to be around 5 for this dataset.

## Conclusion
This project successfully segmented customers into different groups, each with unique characteristics. The clusters can be used by the marketing team to tailor their strategies and improve customer targeting, potentially leading to increased sales and customer satisfaction.
