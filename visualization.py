from dgi import DGI, Classifier
import torch as th
import torch.nn as nn
from gcn import Net
from graphSAGE import SAGE
# from graphSSL import LogisticRegression, Encoder
# from ssldegree import Predictor
from sklearn import metrics
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from utils import load_model, load_elliptic_data, load_ssl_degree_model, load_ssl_model, load_log_reg, \
    load_elliptic_data_SSL

g, features, _, feature_dim, _, test_ids, _, test_labels = load_elliptic_data(
    'dataset/ellipticGraph')


def get_embeddings(g, features):
    in_size = features.shape[1]
    out_size = 2
    model = load_model(Net, path='./model_artifacts/gcn.pth', in_size=in_size, out_size=out_size)
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        _, preds = th.max(logits, dim=1)
    return np.array(preds)


embeddings = get_embeddings(g, features)
print(embeddings)
G = nx.DiGraph()
G.add_nodes_from(range(g.num_nodes()))
for src, dst in zip(*g.edges()):
    G.add_edge(src, dst)

# Set the layout of the graph to be based on the node embeddings
pos = {i: embeddings[i] for i in range(g.num_nodes())}
print(pos)
# Draw the graph with node labels and embeddings as node colors
nx.draw(G, with_labels=True, node_color=embeddings, cmap=plt.cm.Set1)

# Display the graph visualization
plt.savefig('graph.png')





