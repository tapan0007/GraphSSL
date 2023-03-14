#import numpy as np
import torch as th
from gcn import Net
from graphSAGE import SAGE
from graphSSL import LogisticRegression, Encoder
from ssldegree import Predictor, train_ssldegree_logistic_model
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import pandas as pd
from utils import load_model, load_elliptic_data, load_ssl_degree_model, load_ssl_model, load_log_reg, load_elliptic_data_SSL

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

def performance_metrics_ssldegree(encoder, classifier, g, features, labels, mask):
    encoder.eval()
    with th.no_grad():
        embeddings = encoder(g, features)
    _embeddings = embeddings.detach()

    logits, _ = classifier(_embeddings[test_ids], test_labels)
    preds = th.argmax(logits, dim=1)

    f1 = metrics.f1_score(test_labels, preds, average="macro")
    auroc = metrics.roc_auc_score(test_labels, preds)
    g_mean = geometric_mean_score(test_labels, preds)
    return f1, g_mean, auroc

def gcn_performance(g, features, test_labels, test_ids):
    in_size = features.shape[1]
    out_size = 2
    gcn_model = load_model(Net, path="./model_artifacts/gcn.pth", in_size=in_size, out_size=out_size)
    gcn_f1, gcn_g_mean, gcn_auroc = performance_metrics(gcn_model, g, features, test_labels, test_ids)
    return [gcn_f1, gcn_g_mean, gcn_auroc]

def sage_performance(g, features, test_labels, test_ids):
    in_size = features.shape[1]
    out_size = 2
    sage_model = load_model(SAGE, path="./model_artifacts/sage.pth", in_size=in_size, out_size=out_size)
    sage_f1, sage_g_mean, sage_auroc = performance_metrics(sage_model, g, features, test_labels, test_ids)
    return [sage_f1, sage_g_mean, sage_auroc]

def ssldegree_performance(g, features, train_ids, train_labels, test_labels, test_ids):
     model = load_ssl_degree_model(Predictor, g)
     anomaly_ratio = .097
     classifier = train_ssldegree_logistic_model(model, g, features, train_ids, train_labels, anomaly_ratio)
     encoder = model.encoder
     ssldegree_f1, ssldegree_g_mean, ssldegree_auroc = performance_metrics_ssldegree(encoder, classifier, g, features, test_labels, test_ids)
     return [ssldegree_f1, ssldegree_g_mean, ssldegree_auroc]



if __name__ == '__main__':
    # Get metrics for gcn and graphSAGE
    g, features, _, _, _, test_ids, _, test_labels = load_elliptic_data(
                'dataset/ellipticGraph')
    gcn_metrics = gcn_performance(g, features, test_labels, test_ids)
    sage_metrics = sage_performance(g, features, test_labels, test_ids)

    # Get metrics for ssldegree and graphSAGE
    g, features,pimg0,pimg1, num_nodes, feature_dim, train_ids, test_ids, train_labels, test_labels = load_elliptic_data_SSL(
         'dataset/ellipticGraph')
    ssldegree_metrics = ssldegree_performance(g, features, train_ids, train_labels, test_labels, test_ids)

    print(gcn_metrics)
    print(sage_metrics)
    print(ssldegree_metrics)

    data = [gcn_metrics, sage_metrics, ssldegree_metrics]
    metrics = pd.DataFrame(data, columns=["F1 Score", "Geo Mean", "AUROC"], index=["GCN", "GraphSAGE", "SSLDegree"])
    metrics.to_csv("./performance/results.csv")

    # g, features,_,_, _, _, _, test_ids, _, test_labels = load_elliptic_data_SSL(
    #             'dataset/ellipticGraph')
    #ssl_f1, ssl_g_mean, ssl_auroc = performance_metrics_ssl(ssl_classifier, ssl_model, g, features, test_labels, test_ids)

    # performance_data = pd.DataFrame({
    #     "macro_f1":[gcn_f1, sage_f1, ],
    #     "g_mean": [gcn_g_mean, sage_g_mean, ], 
    #     "AUROC": [gcn_auroc, sage_auroc, ]
    #                                  }, index=["GCN", "GraphSAGE", "GraphSSL"])
    # performance_data.to_csv("./performance/results.csv")

    # print("GCN Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
    #     gcn_f1, gcn_g_mean, gcn_auroc))
    # print("GraphSAGE Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
    #     sage_f1, sage_g_mean, sage_auroc))
    # #print("GraphSSL Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
    # #    ssl_f1, ssl_g_mean, ssl_auroc))
    # print("Done running performance metrics")


