import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import evaluateSSL,load_elliptic_data_SSL, save_model
import dgl.nn as dglnn
from dgl import AddSelfLoop
import numpy as np

class Encoder(nn.Module):
    def __init__(self, in_size, hid_size, out_size, decoder_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)
        self.decoder = Decoder(out_size, decoder_size, 1)

    def forward(self, graph, x, shuffled_index):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        dec = self.decoder(h, h[shuffled_index])
        return (h, dec)
        
class Decoder(nn.Module):
    def __init__(self, input_dim, output_dim, cos_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.cos=nn.CosineSimilarity(cos_dim)
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self,a,b):
        h_a = self.linear(a)
        h_b = self.linear(b)
        return self.cross_entropy(h_a,h_b)
    
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
    
def train_ssl_model(g, features, pimg0):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = args.out_dim
    model = Encoder(in_size, args.hidden_dim, out_size, 2).to(device)

    # model training
    print("Training SSL Model Component")
    # define train/val samples, loss function and optimizer
    #cos=nn.CosineSimilarity(1)
    ce = nn.CrossEntropyLoss()
    loss_mse = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    
    # training loop
    for epoch in range(args.epochs):
        model.train()
        shuffled_index = torch.randperm(features.shape[0])
        struct_distances = ce(pimg0, pimg0[shuffled_index])
        h, decoder_sim = model(g, features,shuffled_index)
        loss = loss_mse(decoder_sim, struct_distances.detach())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        print(
            "Epoch {:05d} | Loss {:.4f} | Average Loss {:.4f}  ".format(
                epoch, loss.item(), loss.item() / features.shape[0]
            )
        )
    return model

def train_ssl_logistic_model(model, g, features, train_ids, train_labels, num_class = 2, supervised=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        embeddings, sim = model(g, features, torch.randperm(features.shape[0]))
    _embeddings = embeddings.detach()
    if supervised == True:
        _embeddings = features
    emb_dim = _embeddings.shape[1]
    classifier = LogisticRegression(emb_dim, num_class).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
    for _ in range(100):
        classifier.train()
        logits, loss = classifier(_embeddings[train_ids], train_labels)
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
        "--hidden_dim",
        type=int,
        default=64,
        help="Dimensions of the encoder hidden layer"
    )

    parser.add_argument(
        "--out_dim",
        type=int,
        default=16,
        help="Dimensions of the encoder output"
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
    classifier = train_ssl_logistic_model(model, g, features, train_ids, train_labels)
    save_model(classifier, "./model_artifacts/ssl_classifier.pth")
    supervised_features = torch.cat((features, pimg0, pimg1), 1)
    evaluateSSL(model, classifier, features, test_ids, test_labels)
    #evaluateSSL(model,features,supervised_features, train_ids, test_ids, train_labels, test_labels, 2, True)
