## Readme

*This file describes the topological properties summarized from elliptic dataset.*  

#### Components  
- ellicpticGraph_pimg
  - access: dgl.load_graphs(path)[0][0]
  - homogeneous, undirected graph contains nodal level persistent images
  - DGL graph format (torch backend)
  - ndata['features'] $X\in R^{203769\times 166}$ nodel level features
  - ndata['pimgs0'] $H^{(0)}\in R^{203769\times 64}$ persistent images at homology dimension H0
  - ndata['pimgs1'] $H^{(1)}\in R^{203769\times 64}$ persistent images at homology dimension H1  
- labels 
  - access: dgl.load_graphs(path)[1]
  - key 'known_ids' and 'known_labs', the corresponding index and labels of nodes  
  - label 0: benign, label 1: anomaly 
  - labels of around three quarters of nodes are unknown, please ignore these for downstream task
- diagrams  
  - json dict file, loaded with json.loads() 
  - key 'H0': list of persistent diagrams at homology dimenstion H0 
  - key 'H1': list of persistent diagrams at homology dimenstion H1
  - each diagram summarizes a node neighborhood structrue, so the length of the list ('H0' or 'H1') is 203769
  - each diagram is an $N\times 2$ array, but please be careful about the variable length, *i.e.* $N$ is not a fixed constant
  - there may exist some emtpy diagrams 

#### Others
1. Persistent images (pimgs0/pimgs1) and persistent diagrams (diagrams['H0']/diagrams['H1'])  
   Persistent diagram is computed according to 3-hop neighborhood structure of the target node. It is a set of 2D points. 
   Persistent image is the vectorized representation of a diagram, and can be viewed as a fixed length vector. 
   Both diagram and image are the instances of topological structure summaries, so please be sure to use only one of them for supervision signals.  
   There are two dimensions of persistent summaries, and the more important one is dimension 0, but dimension 1 also provides additional topological information. For simplicity, we could start from using dimension 0 and then incorporate with dimension 1. 
2. Pariwise distance between diagrams 
   Since diagram is a set with variable length and it is difficult to directly predict a set, we can sample a pair of nodes randomly and then predict their pairwise distance between the diagrams.  
   Here we could use bottleneck distance between two diagrams, $d: Dgm \times Dgm \to R$, which takes input of two diagrams and returns a scalar distance. 
   One efficient approximation to do such calculation: [gudhi.hera.bottleneck_distance](https://gudhi.inria.fr/python/latest/bottleneck_distance_user.html).  
   But feel free to explore any distance map defined for sets, *e.g.* chamfer distance, wasserstein distance. 
3. Predicting persistent images
   I also provide another topology representation, persistent images. 
   For images, it is quite simple to develop SSL principles, since it is a fixed length vector.
