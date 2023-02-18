import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import load_elliptic_data, evaluate
import dgl.nn as dglnn
from dgl import AddSelfLoop


class SAGE(nn.Module):
    def __init__(self, in_size, hid_size, out_size):
        super().__init__()
        self.layers = nn.ModuleList()
        # two-layer GraphSAGE-mean
        self.layers.append(dglnn.SAGEConv(in_size, hid_size, "gcn"))
        self.layers.append(dglnn.SAGEConv(hid_size, out_size, "gcn"))
        self.dropout = nn.Dropout(0.5)

    def forward(self, graph, x):
        h = self.dropout(x)
        for l, layer in enumerate(self.layers):
            h = layer(graph, h)
            if l != len(self.layers) - 1:
                h = F.relu(h)
                h = self.dropout(h)
        return h


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphSAGE")
    parser.add_argument(
        "--dataset",
        type=str,
        default="cora",
        help="Dataset name ('cora', 'citeseer', 'pubmed')",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GraphSage module")

    # load and preprocess dataset
    transform = (
        AddSelfLoop()
    )  # by default, it will first remove self-loops to prevent duplication
    # if args.dataset == "elliptic":
    #     g, features, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data(
    #         'dataset/ellipticGraph')
    # else:
    #     raise ValueError("Unknown dataset: {}".format(args.dataset))
    g, features, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data(
                'dataset/ellipticGraph')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    g = g.int().to(device)

    # create GraphSAGE model
    in_size = features.shape[1]
    out_size = 2
    model = SAGE(in_size, 100, out_size).to(device)

    # model training
    print("Training...")
    # define train/val samples, loss function and optimizer
    loss_fcn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

    # training loop
    for epoch in range(200):
        model.train()
        logits = model(g, features)
        loss = loss_fcn(logits[train_ids], train_labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        acc = evaluate(model, g, features, test_labels, test_ids)
        print(
            "Epoch {:05d} | Loss {:.4f} | Accuracy {:.4f} ".format(
                epoch, loss.item(), acc
            )
        )

