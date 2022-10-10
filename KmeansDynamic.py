import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np
#read data from facebook_combined.xlsx
df = pd.read_csv("TOPSIS.R2.csv")
print(df.tail())
G = nx.read_edgelist("facebook_combined.txt",create_using=nx.Graph(), nodetype = int)
list_of_nodes=np.array(list(G.nodes()))
df['nodes']=list_of_nodes
#Preprocessing using min max scaler
scaler = MinMaxScaler()
scaler.fit(df[['C']])
df['C'] = scaler.transform(df[['C']])

scaler.fit(df[['nodes']])
df['nodes'] = scaler.transform(df[['nodes']])


#draw a scatter for visualize our data
plt.scatter(df.nodes,df.C)
plt.xlabel('nodes')
plt.ylabel('C')
plt.show()


#what the perfect number of scatter for our data 
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df[['nodes','C']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

#####
#prapare dataset
data=df[['nodes','C']]
# using Kmeans
model = KMeans()
# know number of clusters
visualizer = KElbowVisualizer(model, k=(1,12)).fit(data)
print(visualizer.elbow_value_)
visualizer.show()


#clustering our data and print the clusters id
km = KMeans(n_clusters=visualizer.elbow_value_)
y_predicted = km.fit_predict(df[['nodes','C']])
print(y_predicted)

#add a columns for clusters id and print te centroid coordinates
df['cluster']=y_predicted
print(df.tail())
print(km.cluster_centers_)
## print(visualizer.elbow_value_)
#draw a scatter plot who represent the different clusters with different colors and the centroid of each cluster
for i in range(visualizer.elbow_value_):
    plt.scatter(df[df.cluster==i].nodes,df[df.cluster==i].C,label = i)
plt.scatter(km.cluster_centers_[:,0],km.cluster_centers_[:,1],color='purple',marker='*',label='centroid')
plt.xlabel('nodes')
plt.ylabel('C')
plt.legend()
plt.show()


options = {
      'cmap'       : plt.get_cmap('jet'), 
      'node_color' : y_predicted,
      'node_size'  : 35,
      'edge_color' : 'tab:grey',
      'with_labels': False
    }
plt.figure()
pos = nx.spring_layout(G)
nx.draw(G,**options,pos=pos,edgecolors='black')
plt.show()