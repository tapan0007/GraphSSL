import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import evaluateSSL,load_elliptic_data_SSL, save_model
import dgl.nn as dglnn
from dgl import AddSelfLoop
import numpy as np
import gudhi
import json
import pickle 
class Encoder(nn.Module):
    def __init__(self, in_size, hid_size1, hid_size2, out_size, decoder_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size1, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size1, out_size, "gcn"))
        #self.layers.append(dglnn.SAGEConv(hid_size2, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)
        self.decoder = Decoder(out_size, decoder_size)

    def forward(self, graph, x, shuffled_index, sample_batch=None,mode="train"):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        if mode == "train":
            h_batch = h[sample_batch]
            h_copy = h_batch.unsqueeze(1).repeat(1, shuffled_index.shape[1] , 1)
            dec = self.decoder(h_copy, h[shuffled_index])
            return (h, dec)
        else:
            return h
        
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.cos=nn.CosineSimilarity(2)
        #self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,a,b):
        h_a = self.linear(a)
        h_b = self.linear(b)
        return self.cos(h_a,h_b)
    
class LogisticRegression(nn.Module):

    def __init__(self, num_dim, num_class,anomaly_ratio=None):
        super().__init__()
        self.linear = nn.Linear(num_dim, num_class)
        torch.nn.init.xavier_uniform_(self.linear.weight.data)
        self.linear.bias.data.fill_(0.0)
        self.cross_entropy = nn.CrossEntropyLoss(weight=torch.FloatTensor([1, (1/anomaly_ratio)]))

    def forward(self, x, y):
        logits = self.linear(x)
        loss = self.cross_entropy(logits, y)
        return logits, loss

def load_diagrams(pickle_file):
    with open(pickle_file, "rb") as fp:   #Pickling
        contents_H0 = pickle.load(fp)
    return contents_H0

contents_H0 = load_diagrams("contents_H0.pkl")
def compute_bottleneck_distance(sample_batch, node_batches):
    ret = []
    for i in range(len(sample_batch)):
        b_dis = []
        for j in range(len(node_batches[i])):
            f1 = sample_batch[i]
            f2 = node_batches[i][j]
            b_dis.append(gudhi.bottleneck_distance(contents_H0[f1], contents_H0[f2], 0.05))
        ret.append(b_dis)
    return torch.tensor(ret)

def train_ssl_model(g, features, pimg0):
    #device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)
    print(len(contents_H0))
    # create GraphSAGE model
    in_size = features.shape[1]
    model = Encoder(in_size, args.hidden_dim1, args.hidden_dim2, args.out_dim, args.decoder_out_dim).to(device)

    # model training
    print("Training SSL Model Component")
    # define train/val samples, loss function and optimizer
    ce = nn.CrossEntropyLoss()
    cos=nn.CosineSimilarity(2)
    loss_mse = nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    features = features.to(device)
    # training loop
    for epoch in range(args.epochs):
        model.train()
        sample_batch = torch.randint(0, features.shape[0], (args.batch_size,))
        batch_features = features[sample_batch]
        #batch_features = features
        node_batches = torch.randint(0, features.shape[0], (batch_features.shape[0], args.node_pair_size))
        node_batches = node_batches.to(device)
        pimg0_batch = pimg0[sample_batch]
        #struct_distances = cos(pimg0_batch.unsqueeze(1).repeat(1, args.node_pair_size , 1), pimg0[node_batches])
        
        struct_distances = compute_bottleneck_distance(sample_batch, node_batches)
        
        h, decoder_sim = model(g, features.to(device),node_batches.to(device), sample_batch.to(device))
        loss = loss_mse(decoder_sim, struct_distances.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch {:05d} | Loss {:.4f} ".format(
                epoch, loss.item()
            )
        )
    return model

def train_ssl_logistic_model(model, g, features, train_ids, train_labels, anomaly_ratio,num_class = 2, supervised=False):
    model.eval()
    #device = "cpu"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        embeddings = model(g, features, None, None, "test")
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
    print(f"Training with DGL built-in GraphSage module")
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train SSL method",
    )

    parser.add_argument(
        "--hidden_dim1",
        type=int,
        default=64,
        help="Dimensions of the encoder hidden layer"
    )


    parser.add_argument(
        "--hidden_dim2",
        type=int,
        default=32,
        help="Dimensions of the encoder hidden layer"
    )

    parser.add_argument(
        "--out_dim",
        type=int,
        default=16,
        help="Dimensions of the encoder output"
    )

    parser.add_argument(
        "--decoder_out_dim",
        type=int,
        default=8,
        help="Dimensions of the decoder output"
    )

    parser.add_argument(
        "--node_pair_size",
        type=int,
        default=20,
        help="Numbers of pairs for the SSL loss per each node"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="batch size"
    )

    args = parser.parse_args()
    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )
    g, features,pimg0,pimg1, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data_SSL(
                'dataset/ellipticGraph')
    model = train_ssl_model(g, features, pimg0)
    save_model(model, "./model_artifacts/ssl_model.pth")
    anomaly_ratio = np.count_nonzero(test_labels == 1) / len(test_labels)
    classifier = train_ssl_logistic_model(model, g, features, train_ids, train_labels,anomaly_ratio)
    save_model(classifier, "./model_artifacts/ssl_classifier.pth")
    evaluateSSL(model, classifier, features, test_ids, test_labels)
