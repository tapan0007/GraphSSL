import torch as th
from gcn import Net
from graphSAGE import SAGE
from sklearn import metrics
from imblearn.metrics import geometric_mean_score
import matplotlib.pyplot as plt
import pandas as pd
from utils import load_model, load_elliptic_data

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


if __name__ == '__main__':
    g, features, _, _, _, test_ids, _, test_labels = load_elliptic_data(
                'dataset/ellipticGraph')
    in_size = features.shape[1]
    out_size = 2
    # Load trained models
    gcn_model = load_model(Net, path="./model_artifacts/gcn.pth", in_size=in_size, out_size=out_size)
    sage_model = load_model(SAGE, path="./model_artifacts/sage.pth", in_size=in_size, out_size=out_size)

    gcn_f1, gcn_g_mean, gcn_auroc = performance_metrics(gcn_model, g, features, test_labels, test_ids)
    sage_f1, sage_g_mean, sage_auroc = performance_metrics(sage_model, g, features, test_labels, test_ids)

    performance_data = pd.DataFrame({
        "macro_f1":[gcn_f1, sage_f1],
        "g_mean": [gcn_g_mean, sage_g_mean], 
        "AUROC": [gcn_auroc, sage_auroc]
                                     }, index=["GCN", "GraphSAGE"])
    performance_data.to_csv("./performance/results.csv")

    print("GCN Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
        gcn_f1, gcn_g_mean, gcn_auroc))
    print("GraphSAGE Metrics: Macro F1-{}, Geometric Mean-{}, AUROC-{}".format(
        sage_f1, sage_g_mean, sage_auroc))
    print("Done running performance metrics")


