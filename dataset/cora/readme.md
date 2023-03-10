## Readme

*This file describes the cora dataset.*  

#### cora.dgl  
- access: dgl.load_graphs(path)[0][0]
- homogeneous, undirected graph contains nodal level features and labels
- DGL graph format (torch backend)
- ndata['feat'] $X\in R^{2708\times 1433}$ node level features
- ndata['label'] $Y\in Z^{2708}$, there are seven classes in total  
- ndata['train_mask'], train set masking (True / False), 140 train node samples (around 5%)
- ndata['val_mask'], validation set masking (True / False), 500 validation node samples 
- ndata['test_mask'], test set masking (True / False), 1000 test node samples
- Please note that the train-val-test split is a default split setting for cora dataset among the graph learning community, but it is okay to make another split manually. 

#### h0diagrams  
- json dict file, loaded with json.loads() 
- h0 dimension structural information
- dict key is the str(node_idx) 
- dict value is a persistent diagram with $N\times 2$ dimension in list of list format, please refer to elliptic for more details

#### h1diagrams  
- json dict file, loaded with json.loads() 
- h1 dimension structural information
- dict key is the str(node_idx) 
- dict value is a persistent diagram with $N\times 2$ dimension in list of list format, please refer to elliptic for more details 