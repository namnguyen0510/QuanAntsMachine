import numpy as np 
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.feature_selection import *
import tqdm
import networkx as nx
import matplotlib.pyplot as plt
import torch.nn as nn
import torch
import seaborn as sns
import stringdb
import requests
import os

try:
	os.mkdir('US')
	os.mkdir('UB')
	os.mkdir('US_csv')
	os.mkdir('UB_csv')
except:
	pass


#############################
############UTILS############
#############################

def norm_adj(G):
	A = nx.to_numpy_matrix(G)
	# 1/MI can result in infty if MI small, -> 
	A[A == np.inf] = -1
	A[A == -1] = np.max(A)
	# Normalize
	A = (A-A.min())/(A.max()-A.min())
	return A


def get_nodes(G, A, num_nodes = 256, save = True):
	nodes = G.nodes
	scr = A.sum(0).tolist()
	scr = scr[0]
	df = pd.DataFrame([])
	df['Gene Name'] = G.nodes
	df['scr'] = scr
	df = df.sort_values(by = 'scr', ascending = False)
	df = df[:num_nodes]
	if save:
		df.to_csv('MI.csv', index = False)
	s_genes = df['Gene Name'].tolist()
	return G.subgraph(s_genes)



def get_string_graph(gs):
	# Source: https://towardsdatascience.com/visualizing-protein-networks-in-python-58a9b51be9d5
	proteins = '%0d'.join(gs)
	url = 'https://string-db.org/api/tsv/network?identifiers=' + proteins + '&species=9606' + '&add_node=50'
	r = requests.get(url)
	lines = r.text.split('\n') # pull the text from the response object and split based on new lines
	data = [l.split('\t') for l in lines] # split each line into its components based on tabs
	# convert to dataframe using the first row as the column names; drop empty, final row
	df = pd.DataFrame(data[1:-1], columns = data[0]) 
	# dataframe with the preferred names of the two proteins and the score of the interaction
	interactions = df[['preferredName_A', 'preferredName_B', 'score']]  
	#print(interactions)
	G=nx.Graph(name='Protein Interaction Graph')
	interactions = np.array(interactions)
	for i in range(len(interactions)):
	    interaction = interactions[i]
	    w = float(interaction[2]) # score as weighted edge where high scores = low weight
	    a = interaction[0] # protein a node
	    b = interaction[1] # protein b node
	    #if w > 0.5:
	    G.add_weighted_edges_from([(a,b,w)]) # add weighted edge to graph
	return G

#############################
###########GETMAP############
#############################
# MI GRAPH
G = nx.read_gpickle('RRGraph_RAS.gpickle')
G.remove_node('RAS')
A = norm_adj(G)
As = norm_adj(G)
gs = G.nodes
# STRING GRAPH
Gb = get_string_graph(G.nodes)
gb = Gb.nodes
Ab = norm_adj(Gb)
inters = set(gs) & set(gb)
Gs = G.subgraph(sorted(inters))
Gb = Gb.subgraph(sorted(inters))

#Gs = get_nodes(Gs, norm_adj(Gs), num_nodes = 256, save = True)
#Gb = get_nodes(Gb, norm_adj(Gb), num_nodes = 256, save = True)
print(len(set(Gs.nodes) & set(Gb.nodes)))
print(Gs)
print(Gb)
As = norm_adj(Gs)
Ab = norm_adj(Gb)
print(Gs.nodes)
print(Gb.nodes)
print(As.shape)
print(Ab.shape)


df_As = pd.DataFrame(As)
df_As.columns = Gs.nodes
df_As = df_As.reindex(sorted(df_As.columns), axis=1)
df_Ab = pd.DataFrame(Ab)
df_Ab.columns = Gb.nodes
df_Ab = df_Ab.reindex(sorted(df_Ab.columns), axis=1)
print(df_As)
print(df_Ab)
sns.heatmap(df_As , cbar = False)
plt.savefig('AMI.jpg', dpi =600)
sns.heatmap(df_Ab, cbar = False)
plt.savefig('String.jpg', dpi =600)
df_As.to_csv('MI_map.csv')
df_Ab.to_csv('STRING_map.csv')

As = df_As.to_numpy()
Ab = df_Ab.to_numpy()

#print(As)
#print(Ab)

def Sampling(As, Ab, initiate = True, state_s = None, state_b = None, dt = 7, iters = None):
	# INITIATE IF FIRST STEP
	if initiate:
		state_s = np.zeros(len(As))
		state_s[0] = 1
		state_b = np.zeros(len(Ab))
		state_b[0] = 1


	# ELSE USE PREVIOUS STATE
	#print(state_s,state_b)
	# SPECTRAL DECOMPOSITION
	eval_As, evec_As = np.linalg.eig(As)
	eval_Ab, evec_Ab = np.linalg.eig(Ab)
	print(eval_As, evec_As)
	print(eval_Ab, evec_Ab)
	#eval_sum = eta*eval_As+(1-eta)*eval_Ab
	eval_sum = eval_As + eval_Ab
	D = np.diag(eval_sum*dt*1j)
	# COMPUTING UNITARY TRANSFORMATION (DUAL-TRANSITION)
	Us = np.matmul(np.matmul(evec_As.conj().T, D), evec_As)
	Ub = np.matmul(np.matmul(evec_Ab.conj().T, D), evec_Ab)
	print(Us.conj().T*Us)
	print(Ub.conj().T*Ub)


	Us = np.abs(Us)
	Ub = np.abs(Ub)
	#print(evec_As.shape)
	#print(evec_Ab.shape)

	df_US = pd.DataFrame(Us)
	df_US.columns = inters
	df_US.to_csv('US_csv/Us_{}.csv'.format(iters),index = True)
	df_UB = pd.DataFrame(Ub)
	df_UB.columns = inters
	df_UB.to_csv('UB_csv/Ub_{}.csv'.format(iters),index = True)


	# PLOT DUAL-TRANSITION
	sns.heatmap(Us , cbar = False)
	plt.savefig('US/Us_{}.jpg'.format(iters), dpi =600)
	sns.heatmap(Ub, cbar = False)
	plt.savefig('UB/Ub_{}.jpg'.format(iters), dpi =600)
	state_st = np.matmul(Us,state_s)
	state_bt = np.matmul(Ub,state_b)


	return Us, Ub, state_st, state_bt

from scipy.special import softmax

Us, Ub,state_st, state_bt = Sampling(As, Ab, initiate = True)
print(np.argmax(state_st))
print(np.argmax(state_bt))


for i in range(10):
	print("iter: {}".format(i))
	Us, Ub, state_st, state_bt = Sampling(Us, Ub, initiate = False, state_s=state_st, state_b=state_bt, iters = i)
	#print(state_st)
	#print(state_bt)
	ps = softmax(state_st)
	pb = softmax(state_bt)
	print(np.argmax(ps))
	print(np.argmax(pb))
	if (np.argmax(ps) == 0) and (np.argmax(pb) == 0):
		break


	


















#











#