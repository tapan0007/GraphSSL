import dgl
import sklearn.model_selection as sk
import numpy as np
#from ssldegree import Predictor
#from graphSSL import LogisticRegression
import torch as th
import torch

def load_elliptic_data(path):
    dataset = dgl.load_graphs(path)
    g = dataset[0][0]
    feats = g.ndata['features']
    # Below can be removed if test metrics do not perform well
    graph_structure = dgl.load_graphs('dataset/ellipticGraph_pimg')
    # Ablation Study, only add pimg0 or only add pimg1 or add both
    pimg0 = graph_structure[0][0].ndata['pimgs0']
    # feats = th.cat((feats, pimg0), 1)
    pimg1 = graph_structure[0][0].ndata['pimgs1']
    # feats = th.cat((feats, pimg1), 1)
    feats = th.cat((feats, pimg0, pimg1), 1)
    # Till here
    n_nodes, feat_dim = feats.shape[0], feats.shape[1]
    known_ids = np.array(dataset[1]['ids'])
    node_labels = dataset[1]['labels']
    train_nid, test_nid, train_l, test_l = sk.train_test_split(known_ids, node_labels, test_size=0.95, random_state=42)
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
    #feats = torch.cat((feats, pimg0, pimg1), 1)
    # Till here
    n_nodes, feat_dim = feats.shape[0], feats.shape[1]
    known_ids = np.array(dataset[1]['ids'])
    node_labels = dataset[1]['labels']
    train_nid, test_nid, train_l, test_l = sk.train_test_split(known_ids, node_labels, test_size=0.95, random_state=42)
    return g, feats,pimg0,pimg1, n_nodes, feat_dim, train_nid, test_nid, train_l, test_l


def evaluate(model, g, features, labels, mask):
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, indices = th.max(logits, dim=1)
        correct = th.sum(indices == labels)
    return correct.item() * 1.0 / len(labels)

def evaluateSSL(model, g, classifier,features, test_ids, test_labels, supervised=False):
    model.eval()
    with torch.no_grad():
        embeddings, sim = model(g, features, torch.randperm(features.shape[0]))
    _embeddings = embeddings.detach()
    if supervised == True:
        _embeddings = features

    test_logits, _ = classifier(_embeddings[test_ids], test_labels)
    test_preds = torch.argmax(test_logits, dim=1)
    test_acc = (torch.sum(test_preds == test_labels).float() / test_labels.shape[0]).detach().cpu().numpy()

    print('Average test accuracy {}'.format(test_acc))    
    return test_acc   
        
    
def save_model(model, path):
    try:
        th.save(model.state_dict(), path)
        print("Model saved to {}".format(path))
    except Exception as e:
        print("Model not saved")
        print(e)

def load_model(model_class, path, in_size, out_size):
    model = model_class(in_size, 100, out_size)
    model.load_state_dict(th.load(path))
    return model.eval()

def load_ssl_model(model_class, in_size, hid_size1, hid_size2, out_size, decoder_size):
    path="./model_artifacts/ssl_model.pth"
    model = model_class(in_size, hid_size1, hid_size2, out_size, decoder_size)
    model.load_state_dict(th.load(path))
    return model.eval()

def load_ssl_degree_model(model_class, g):
    model = model_class(g, 128)
    path="./trainedmodel/ssl_predictor_node_200.pt"
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    return model.eval()

def load_log_reg(model_class, path, num_dim, num_class):
    model = model_class(num_dim, num_class)
    model.load_state_dict(th.load(path))
    return model.eval()
