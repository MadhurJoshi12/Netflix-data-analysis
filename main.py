import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 📂 Load data
df = pd.read_csv('Netflix dataset.csv', encoding="utf-8")

print("Original Shape:", df.shape)

# 🎯 Select only numeric columns
X = df.select_dtypes(include=[np.number])

# 🧹 Handle missing values (keep only complete rows)
mask = X.notna().all(axis=1)
X_clean = X[mask]
# mask cleans data for model
# .dropna() cleans data for plotting

print("Clean Shape:", X_clean.shape)

# ⚖️ Scale data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_clean)

# # 📉 Elbow Method
# wcss = []

# for k in range(1, 15):
#     kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
#     # n_init = how many times K-Means will run with different random starting centroids
#     kmeans.fit(X_scaled)
#     wcss.append(kmeans.inertia_) 
#     # inertia_ = Total WCSS (Within-Cluster Sum of Squares)

# # 📊 Plot Elbow Graph
# plt.figure(figsize=(8,5))
# plt.plot(range(1,15), wcss, marker='o')
# plt.xlabel('Number of Clusters (K)')
# plt.ylabel('WCSS')
# plt.title('Elbow Method')
# plt.grid(True)
# plt.show()

# 🎯 Choose K (change this after observing elbow)
best_k = 5

# 🤖 Train final model
model = KMeans(n_clusters=best_k, random_state=42, n_init=10)
# 10 is not special—it’s just a practical balance

# 🏷️ Assign clusters ONLY to valid rows
df.loc[mask, 'Group'] = model.fit_predict(X_scaled)
'''👉 It is:
NOT predicting
NOT guessing future values
👉 It is saying:
These points look similar → put them in same group
'''
# 📊 Visualization (choose 2 real numeric columns)
cols = X_clean.columns

if len(cols) >= 2:
    col1, col2 = cols[0], cols[1]
    # If at least 2 columns exist, pick first two for plotting
    plt.figure(figsize=(6,5))
    for group in df['Group'].dropna().unique():
        group_data = df[df['Group'] == group]
        # Give me only rows belonging to this group
        plt.scatter(group_data[col1], group_data[col2], label=f'Group {int(group)}')

    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.title("K-Means Clustering")
    plt.legend()
    plt.grid(True)
    plt.show()

# 🖨️ Final output
# print(df[['Group']].head(20)) >> For row maybe can be either clustered or non clustered rows
print(df[df['Group'].notna()])  #>>For clustered rows
