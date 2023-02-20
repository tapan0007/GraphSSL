import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_elliptic_data, evaluate,load_elliptic_data_SSL
import dgl.nn as dglnn
from dgl import AddSelfLoop


class GraphSSL(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)
        self.sim = Similarity(1)

    def forward(self, graph, x, shuffled_index):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        decoder = self.sim(h, h[shuffled_index])
        return (h, decoder)
        
class Similarity(nn.Module):
    def __init__(self, dim):
        super().__init__()
        cos=nn.CosineSimilarity(dim)

    def forward(self,a,b):
        return cos(a,b)
    
class LogisticRegression(nn.Module):

    def __init__(self, num_dim, num_class):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss


if __name__ == "__main__":
    print(f"Training with DGL built-in GraphSage module")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )
    g, features,pimg0,pimg1, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data_SSL(
                'dataset/ellipticGraph')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = 2
    model = GraphSSL(in_size, 100, out_size).to(device)

    # model training
    print("Training...")
    # define train/val samples, loss function and optimizer
    cos = nn.CosineSimilarity(dim=1, eps=1e-6)
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    
    # training loop
    for epoch in range(200):
        model.train()
        shuffled_index = torch.randperm(features.shape[0])
        struct_distances = cos(pimg0, pimg0[shuffled_index])
        h, decoder_sim = model(g, features,shuffled_index)
        loss = loss_mse(decoder_sim, struct_distances.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch {:05d} | Loss {:.4f}  ".format(
                epoch, loss.item()
            )
        )
