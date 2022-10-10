from networkx.algorithms import community
import networkx as nx
import matplotlib.pyplot as plt
G = nx.karate_club_graph()
#Nombre de partitions trouvées 
partition = community.greedy_modularity_communities(G)
print(len(partition))

#Affichage de la composition des communautés
for i in range(len(partition)):
        community = list(partition[i])
        print("Communauté ",str(i+1))
        print([ i for i in community ])

#Visualisation par contruction d’un vecteur de couleurs :
couleurs_num = [0] * G.number_of_nodes()
for i in range(len(partition)):
  for j in partition[i]:
    couleurs_num[j] = i

options = {
      'cmap'       : plt.get_cmap('jet'), 
      'node_color' : couleurs_num,
      'node_size'  : 35,
      'edge_color' : 'tab:grey',
      'with_labels': False
    }
plt.figure()
pos = nx.spring_layout(G)
nx.draw(G,**options,pos=pos,edgecolors='black')
plt.show()