import dgl
import sklearn.model_selection as sk
import numpy as np
import torch as th


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

# remove this function after testing
def load_elliptic_data_SSL(path):
    dataset = dgl.load_graphs(path)
    g = dataset[0][0]
    feats = g.ndata['features']
    # Below can be removed if test metrics do not perform well
    graph_structure = dgl.load_graphs('dataset/ellipticGraph_pimg')
    pimg0 = graph_structure[0][0].ndata['pimgs0']
    pimg1 = graph_structure[0][0].ndata['pimgs1']
    print(feats.shape)
    print(pimg0.shape)
    print(pimg1.shape)
    #feats = torch.cat((feats, pimg0, pimg1), 1)
    # Till here
    n_nodes, feat_dim = feats.shape[0], feats.shape[1]
    known_ids = np.array(dataset[1]['ids'])
    node_labels = dataset[1]['labels']
    train_nid, test_nid, train_l, test_l = sk.train_test_split(known_ids, node_labels, test_size=0.6, random_state=42)
    return g, feats,pimg0,pimg1, n_nodes, feat_dim, train_nid, test_nid, train_l, test_l


g, features, pimg0, pimg1,num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data_SSL(
                'dataset/ellipticGraph')

def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)