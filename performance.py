import torch as th
from gcn import Net
from graphSAGE import SAGE
from graphSSL import LogisticRegression, Encoder
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model, load_elliptic_data, load_ssl_model, load_log_reg, load_elliptic_data_SSL

def performance_metrics(model, g, features, labels, mask):
    """Calculates the AUC ROC, Geometric mean, and macro-f1 score to measure
    model performance.
    """
    model.eval()
    with th.no_grad():
        logits = model(g, features)
        logits = logits[mask]
        _, preds = th.max(logits, dim=1)

    f1 = metrics.f1_score(labels, preds, average="macro")
    auroc = metrics.roc_auc_score(labels, preds)
    g_mean = geometric_mean_score(labels, preds)
    return f1, g_mean, auroc

def performance_metrics_ssl(classifier, model, g, features, labels, mask):
    """Calculates the AUC ROC, Geometric mean, and macro-f1 score to measure
    model performance.
    """
    model.eval()
    with th.no_grad():
        embeddings, sim = model(g, features, th.randperm(features.shape[0]))
    _embeddings = embeddings.detach()

    logits, _ = classifier(_embeddings[mask], labels)
    preds = th.argmax(logits, dim=1)

    f1 = metrics.f1_score(labels, preds, average="macro")
    auroc = metrics.roc_auc_score(labels, preds)
    g_mean = geometric_mean_score(labels, preds)
    return f1, g_mean, auroc


if __name__ == '__main__':
    g, features, _, _, _, test_ids, _, test_labels = load_elliptic_data(
                'dataset/ellipticGraph')
    in_size = features.shape[1]
    out_size = 2
    # Load trained models
    gcn_model = load_model(Net, path="./model_artifacts/gcn.pth", in_size=in_size, out_size=out_size)
    sage_model = load_model(SAGE, path="./model_artifacts/sage.pth", in_size=in_size, out_size=out_size)
    ssl_model = load_ssl_model(Encoder, path="./model_artifacts/ssl_model.pth", in_size=166, hid_size=64, out_size=16, decoder_size=2)
    ssl_classifier = load_log_reg(LogisticRegression, path="./model_artifacts/ssl_classifier.pth", num_dim=16, num_class=2)

    gcn_f1, gcn_g_mean, gcn_auroc = performance_metrics(gcn_model, g, features, test_labels, test_ids)
    sage_f1, sage_g_mean, sage_auroc = performance_metrics(sage_model, g, features, test_labels, test_ids)

    g, features,_,_, _, _, _, test_ids, _, test_labels = load_elliptic_data_SSL(
                'dataset/ellipticGraph')
    ssl_f1, ssl_g_mean, ssl_auroc = performance_metrics_ssl(ssl_classifier, ssl_model, g, features, test_labels, test_ids)

    performance_data = pd.DataFrame({
        "macro_f1":[gcn_f1, sage_f1, ssl_f1],
        "g_mean": [gcn_g_mean, sage_g_mean, ssl_g_mean], 
        "AUROC": [gcn_auroc, sage_auroc, ssl_auroc]
                                     }, index=["GCN", "GraphSAGE", "GraphSSL"])
    performance_data.to_csv("./performance/results.csv")

    print("GCN Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
        gcn_f1, gcn_g_mean, gcn_auroc))
    print("GraphSAGE Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
        sage_f1, sage_g_mean, sage_auroc))
    print("GraphSSL Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
        ssl_f1, ssl_g_mean, ssl_auroc))
    print("Done running performance metrics")


