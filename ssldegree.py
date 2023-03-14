import argparse
import random

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.nn as dglnn
from graphSSL import LogisticRegression


class Encoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.3)

    def forward(self, graph, h):
        if self.training: h = self.dropout(h)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                if self.training: h = self.dropout(h)
        return h


class Decoder(nn.Module):
    def __init__(self, input_dim, hid_dim, out_dim, num_layers):
        super().__init__()
        self.linears = nn.ModuleList()
        dims = [input_dim] + [hid_dim] * (num_layers-1) + [out_dim]
        for inp, out in zip(dims[:-1], dims[1:]):
            self.linears.append(nn.Linear(inp, out))
        self.activation = nn.ReLU()

    def forward(self, h):
        for layer in self.linears[:-1]:
            h = self.activation(layer(h))
        return self.linears[-1](h)

class Predictor(nn.Module):
    def __init__(self, graph, hid_dim):
        super(Predictor, self).__init__()
        self.graph = graph
        self.encoder = Encoder(
            in_size=self.graph.ndata['features'].size(1),
            hid_size=hid_dim,
            out_size=hid_dim
        )
        self.decoder = Decoder(
            input_dim=hid_dim,
            hid_dim=hid_dim,
            out_dim=1,
            num_layers=2
        )
        self.mse = nn.MSELoss()

    def forward(self):
        with self.graph.local_scope():
            feats = self.graph.ndata['features']
            h = self.encoder(self.graph, feats)
            h = self.decoder(h)
        return h

    def loss(self):
        with self.graph.local_scope():
            degrees = self.graph.ndata['degree']
            output = self.forward()
            return self.mse(output.squeeze(), degrees)

    def batch_loss(self, index):
        with self.graph.local_scope():
            degrees = self.graph.ndata['degree'][index].to(torch.float)
            output = self.forward()[index].squeeze()
            return self.mse(output, degrees)
        
def train_ssldegree_logistic_model(model, g, features, train_ids, train_labels, anomaly_ratio,num_class = 2, supervised=False):
    encoder = model.encoder
    encoder.eval()
    #device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        embeddings = encoder(g, features)
    _embeddings = embeddings.detach()
    if supervised == True:
        _embeddings = features
    emb_dim = _embeddings.shape[1]
    classifier = LogisticRegression(emb_dim, num_class,anomaly_ratio).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(100):
        classifier.train()
        logits, loss = classifier(_embeddings[train_ids].to(device), train_labels.to(device))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()  
    return classifier


if __name__ == "__main__":

    device = "cuda:2"

    graph = dgl.load_graphs('/data/zilu/files/elliptic/ellipticGraph_nodeDegree')[0][0].to(device)

    model = Predictor(graph, hid_dim=128).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    num_epochs = 200

    batch_size = 4096

    num_batches = graph.number_of_nodes() // batch_size

    node_index = list(range(graph.number_of_nodes()))

    model.train()

    for e in range(num_epochs):
        random.shuffle(node_index)
        epochloss = []
        for b in range(num_batches):
            batch_index = node_index[b*batch_size : (b+1)*batch_size]
            optimizer.zero_grad()
            loss = model.batch_loss(batch_index)
            loss.backward()
            optimizer.step()
            epochloss.append(loss.item())
        print('Epoch '+str(e), end=": ")
        print(sum(epochloss)/len(epochloss))
    '''
    model.train()
    for e in range(num_epochs):
        optimizer.zero_grad()
        loss = model.loss()
        loss.backward()
        optimizer.step()
        print(loss.item())
    '''
    torch.save(model.state_dict(), "trainedmodel/ssl_predictor_node_200.pt")

    print('finsihed ssl train')
