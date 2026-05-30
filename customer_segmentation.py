import pandas as pd
from sklearn.cluster import KMeans

data = {
    'Age':[22,45,35,25,50,28,42,31,23,48],
    'Income':[25,70,60,30,90,40,85,55,35,95],
    'Spending':[80,20,60,75,15,65,25,55,85,10]
}

df = pd.DataFrame(data)

kmeans = KMeans(n_clusters=3, random_state=42)
df['Cluster'] = kmeans.fit_predict(df[['Income','Spending']])

print(df)
