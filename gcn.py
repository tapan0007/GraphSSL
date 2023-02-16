import argparse
import dgl
import dgl.function as fn
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import time
import sklearn.model_selection as sk
import numpy as np

gcn_msg = fn.copy_u(u='h', out='m')
gcn_reduce = fn.sum(msg='m', out='h')


class GCNLayer(nn.Module):
    def __init__(self, in_feats, out_feats):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, g, feature):
        # Creating a local scope so that all the stored ndata and edata
        # (such as the `'h'` ndata below) are automatically popped out
        # when the scope exits.
        with g.local_scope():
            g.ndata['h'] = feature
            g.update_all(gcn_msg, gcn_reduce)
            h = g.ndata['h']
            return self.linear(h)


class Net(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(Net, self).__init__()
        self.layer1 = GCNLayer(in_dim, hidden_dim)
        self.layer2 = GCNLayer(hidden_dim, out_dim)

    def forward(self, g, features):
        x = F.relu(self.layer1(g, features))
        x = self.layer2(g, x)
        return x


def load_elliptic_data(path):
    dataset = dgl.load_graphs(path)
    g = dataset[0][0]
    feats = g.ndata['features']
    # Below can be removed if test metrics do not perform well
    graph_structure = dgl.load_graphs('dataset/ellipticGraph_pimg')
    pimg0 = graph_structure[0][0].ndata['pimgs0']
    pimg1 = graph_structure[0][0].ndata['pimgs1']
    feats = th.cat((feats, pimg0, pimg1), 1)
    # Till here
    n_nodes, feat_dim = feats.shape[0], feats.shape[1]
    known_ids = np.array(dataset[1]['ids'])
    node_labels = dataset[1]['labels']
    train_nid, test_nid, train_l, test_l = sk.train_test_split(known_ids, node_labels, test_size=0.6, random_state=42)
    return g, feats, n_nodes, feat_dim, train_nid, test_nid, train_l, test_l


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="GCN")
    parser.add_argument(
        "--dataset",
        type=str,
        default="elliptic",
        help="Dataset name ('ellipticGraph')",
    )
    args = parser.parse_args()
    print(f"Training with DGL built-in GCN module")
    g, features, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data(
        'dataset/ellipticGraph')
    net = Net(feature_dim, 100, 2)

    g.add_edges(g.nodes(), g.nodes())

    optimizer = th.optim.Adam(net.parameters(), lr=1e-2)
    dur = []
    for epoch in range(100):
        if epoch >= 3:
            t0 = time.time()

        net.train()
        logits = net(g, features)
        logp = F.log_softmax(logits, 1)
        loss = F.nll_loss(logp[train_ids], train_labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        acc = evaluate(net, g, features, test_labels, test_ids)
        print("Epoch {:05d} | Loss {:.4f} | Test Acc {:.4f} | Time(s) {:.4f}".format(
            epoch, loss.item(), acc, np.mean(dur)))
