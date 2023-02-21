import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_elliptic_data, evaluateSSL,load_elliptic_data_SSL
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

def evaluateSSL(model,features,supervised_features,train_ids, test_ids, train_labels, test_labels, num_class = 2, supervised=False):
    model.eval()
    with torch.no_grad():
        embeddings, sim = model(g, features, torch.randperm(features.shape[0]))
    _embeddings = embeddings.detach()
    if supervised == True:
        _embeddings = features
    iters = 20
    test_accs = []
    emb_dim = _embeddings.shape[1]
    for i in range(iters):
        classifier = LogisticRegression(emb_dim, num_class).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=0.01, weight_decay=0.0)
        for _ in range(100):
            classifier.train()
            logits, loss = classifier(_embeddings[train_ids], train_labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        test_logits, _ = classifier(_embeddings[test_ids], test_labels)
        test_preds = torch.argmax(test_logits, dim=1)
        test_acc = (torch.sum(test_preds == test_labels).float() / test_labels.shape[0]).detach().cpu().numpy()
        test_accs.append(test_acc * 100)
        #print("Finished iteration {:02} of the logistic regression classifier.test accuracy {:.2f}".format(i + 1, test_acc))
    test_accs = np.stack(test_accs)
    test_acc, test_std = test_accs.mean(), test_accs.std()
    model_name = "SSL" if supervised == False else "supervised"
    print('Average test accuracy for {}: {:.2f} with std: {:.2f}'.format(model_name,test_acc, test_std))    
        
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
        default=20,
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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = args.out_dim
    model = Encoder(in_size, args.hidden_dim, out_size, 2).to(device)

    # model training
    print("Training...")
    # define train/val samples, loss function and optimizer
    cos=nn.CosineSimilarity(1)
    loss_mse = nn.MSELoss(reduction = 'sum')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)
    
    # training loop
    for epoch in range(args.epochs):
        model.train()
        shuffled_index = torch.randperm(features.shape[0])
        struct_distances = cos(pimg0, pimg0[shuffled_index])
        struct_distances = struct_distances.detach()
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
    supervised_features = torch.cat((features, pimg0, pimg1), 1)
    print("Training a logistic regression model to test both SSL and supervised model")
    evaluateSSL(model,features,supervised_features, train_ids, test_ids, train_labels, test_labels)
    #evaluateSSL(model,features,supervised_features, train_ids, test_ids, train_labels, test_labels, 2, True)
