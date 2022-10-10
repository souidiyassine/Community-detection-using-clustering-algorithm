import pandas as pd
from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
from sklearn.preprocessing import MinMaxScaler
from matplotlib import pyplot as plt
import networkx as nx
import numpy as np

g = nx.karate_club_graph()
list_of_nodes=np.array(list(g.nodes()))
Dic = {
  'BC':nx.betweenness_centrality(g),
  'CC':nx.closeness_centrality(g),
  'DC':nx.degree_centrality(g),
  'EC':nx.eigenvector_centrality(g, max_iter=5000)}
df=pd.DataFrame(data=Dic)
print(df.head())
# Normalize
N=[]
for j in range(len(df.columns)):
        S=0
        for i in range(len(df)):
                S+=np.square(df.iloc[i,j])
        N.append(np.sqrt(S))
        df.iloc[:,j]/=N[j]

print(df.head())
#weight
df['BC']*=0.3
df['CC']*=0.3
df['DC']*=0.2
df['EC']*=0.2
print(df.head())
# Calcul v+ et v-
Max=df.max()
Min=df.min()
VP=np.array(Max)
VN=np.array(Min)
#Calcul S
SP=[]
SN=[]
C=[]
for i in range(len(df)):
        S_Pos=0
        S_Neg=0
        for j in range(len(df.columns)):
               S_Pos +=np.square(VP[j] - df.iloc[i,j])
               S_Neg +=np.square(VN[j] - df.iloc[i,j])
        SP.append(np.sqrt(S_Pos))
        SN.append(np.sqrt(S_Neg))
        Aux=SN[i]/(SN[i]+SP[i])
        C.append(Aux)
df['S+']=SP
df['S-']=SN
df['C']=C
print(df.head())
df2 = pd.DataFrame({'nodes' : list_of_nodes,'C' : C})
df2.sort_values(by=['C'],ascending=False, inplace=True)
K = int(input("Entrer le nombre K de noeuds plus influant que vous voulez voir : "))
for i in range(0,K,1):
    print(df2.iloc[i,0])

#Preprocessing using min max scaler
scaler = MinMaxScaler()
scaler.fit(df2[['C']])
df['C'] = scaler.transform(df2[['C']])

scaler.fit(df2[['nodes']])
df['nodes'] = scaler.transform(df2[['nodes']])


#draw a scatter for visualize our data
plt.scatter(df2.nodes,df2.C)
plt.xlabel('nodes')
plt.ylabel('C')
plt.show()


#what the perfect number of scatter for our data 
sse = []
k_rng = range(1,10)
for k in k_rng:
    km = KMeans(n_clusters=k)
    km.fit(df2[['nodes','C']])
    sse.append(km.inertia_)
plt.xlabel('K')
plt.ylabel('Sum of squared error')
plt.plot(k_rng,sse)
plt.show()

# using Kmeans
model = KMeans()
# know number of clusters
visualizer = KElbowVisualizer(model, k=(1,12)).fit(df2)
print(visualizer.elbow_value_)
visualizer.show()


#clustering our data and print the clusters id
km = KMeans(n_clusters=visualizer.elbow_value_)
y_predicted = km.fit_predict(df2)
print(y_predicted)

#add a columns for clusters id and print te centroid coordinates
df2['cluster']=y_predicted
print(df.tail())
print(km.cluster_centers_)
## print(visualizer.elbow_value_)
#draw a scatter plot who represent the different clusters with different colors and the centroid of each cluster
for i in range(visualizer.elbow_value_):
    plt.scatter(df2[df2.cluster==i].nodes,df2[df2.cluster==i].C,label = i)
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
pos = nx.spring_layout(g)
nx.draw(g,**options,pos=pos,edgecolors='black')
plt.show()