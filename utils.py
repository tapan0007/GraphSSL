import dgl
import sklearn.model_selection as sk
import numpy as np
#from graphSSL import LogisticRegression
import torch as th
import torch

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

def evaluateSSL(model,features,supervised_features,train_ids, test_ids, train_labels, test_labels, num_class = 2, supervised=False):
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
