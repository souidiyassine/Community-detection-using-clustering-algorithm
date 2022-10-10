import networkx as nx
import EoN
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
G = nx.read_edgelist("facebook_combined.txt",create_using=nx.Graph(), nodetype = int)
df=pd.read_csv('TOPSIS.R2.csv')
df.sort_values(by=['C'],ascending=False, inplace=True)
K = int(input("Entrer le nombre K de noeuds plus influant que vous voulez voir : "))

#sorted by C
list=[]
for i in range(0,K,1):
   list.append(df.iloc[i,0])
print(list)

#sorted by BC 
df.sort_values(by=['BC'],ascending=False, inplace=True)
list_BC=[]
for i in range(0,K,1):
   list_BC.append(df.iloc[i,0])

#sorted by CC
df.sort_values(by=['CC'],ascending=False, inplace=True)
list_CC=[]
for i in range(0,K,1):
   list_CC.append(df.iloc[i,0])

#sorted by DC
df.sort_values(by=['DC'],ascending=False, inplace=True)
list_DC=[]
for i in range(0,K,1):
   list_DC.append(df.iloc[i,0])

#sorted by EC
df.sort_values(by=['EC'],ascending=False, inplace=True)
list_EC=[]
for i in range(0,K,1):
   list_EC.append(df.iloc[i,0])


gamma = 1.
tau = 0.3
fig, axs = plt.subplots(2, 2)

#TOPSIS VS BC
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list)
axs[0,0].plot(t,np.cumsum(I),  label='TOPSIS')
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list_BC)
axs[0,0].plot(t,np.cumsum(I), label='BC')
axs[0,0].legend()
axs[0,0].set_title("TOPSIS VS BC")

#TOPSIS VS CC
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list)
axs[0,1].plot(t,np.cumsum(I),  label='TOPSIS')
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list_CC)
axs[0,1].plot(t,np.cumsum(I), label='CC')
axs[0,1].legend()
axs[0,1].set_title("TOPSIS VS CC")

#TOPSIS VS DC
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list)
axs[1,0].plot(t,np.cumsum(I),  label='TOPSIS')
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list_DC)
axs[1,0].plot(t,np.cumsum(I), label='DC')
axs[1,0].legend()
axs[1,0].set_title("TOPSIS VS DC")

#TOPSIS VS EC 
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list)
axs[1,1].plot(t,np.cumsum(I),  label='TOPSIS')
t, S, I, R = EoN.fast_SIR(G, tau, gamma,tmax=10,initial_infecteds = list_EC)
axs[1,1].plot(t,np.cumsum(I), label='EC')
axs[1,1].legend()
axs[1,1].set_title("TOPSIS VS EC")
plt.show()