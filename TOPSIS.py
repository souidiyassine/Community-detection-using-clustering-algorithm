from matplotlib.pyplot import prism
import networkx as nx
import pandas as pd
import numpy as np
g = nx.read_edgelist("facebook_combined.txt",create_using=nx.Graph(), nodetype = int)
# Creates pandas DataFrame.  
list_of_nodes=np.array(list(g.nodes()))
Dic = {
  'BC':nx.betweenness_centrality(g),
  'CC':nx.closeness_centrality(g),
  'DC':nx.degree_centrality(g),
  'EC':nx.eigenvector_centrality(g, max_iter=5000)}

df=pd.DataFrame(data=Dic,index=list_of_nodes)
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
df.to_csv('TOPSIS.R2.csv')
df2 = pd.DataFrame({'nodes' : list_of_nodes,'C' : C})
df2.sort_values(by=['C'],ascending=False, inplace=True)
K = int(input("Entrer le nombre K de noeuds plus influant que vous voulez voir : "))
for i in range(0,K,1):
    print(df2.iloc[i,0])