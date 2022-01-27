# Multi-view Clustering
A categorization and reproduction for Multi-view Clustering(MvC).

## Contents
* [Papers](##papers)
* [Performance](##performance)
<!-- * [Datasets](#datasets)
* [Papers](#papers)
* [Leaderboard](#leaderboard) -->
  
## Papers
### Consensus MvC for better performance
 - [x] CAN [Clustering and Projected Clustering with Adaptive Neighbors](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.707.56&rep=rep1&type=pdf) [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/CAN.py)
   - [x] MLAN [Multi-View Clustering and Semi-Supervised Classification with Adaptive Neighbours](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPDFInterstitial/14833/14423)   [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/MLAN.py)
 - [x] CLR [The constrained laplacian rank algorithm for graph-based clustering](https://ojs.aaai.org/index.php/AAAI/article/download/10302/10161)  [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/CLR.py)
   - [x] SwMC [Self-weighted Multiview Clustering with Multiple Graphs](https://www.ijcai.org/proceedings/2017/0357.pdf)  [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/PwSC.py)
   - [x] PwMC [Self-weighted Multiview Clustering with Multiple Graphs](https://www.ijcai.org/proceedings/2017/0357.pdf)  [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/PwSC.py)
 - [x] Co-reg [Co-regularized multi-view spectral clustering](http://www.abhishek.umiacs.io/coregspectral.nips11.pdf)    [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/Co-regularization.py)
 - [x] AMGL [Parameter-free auto-weighted multiple graph learning: a framework for multiview clustering and semi-supervised classification](https://www.ijcai.org/Proceedings/16/Papers/269.pdf)  [[code]](https://github.com/bjlfzs/multi-view-clustering/blob/main/reproduction/concensus/AMGL.py)
 - [ ] MCGC [Multiview Consensus Graph Clustering]


### MvC for large-scale data
 - [ ] MVSC [Large-scale multi-view spectral clustering via bipartite graph](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9641)
 - [ ] SMKMC [Multi-View K-Means Clustering on Big Data](https://www.researchgate.net/profile/Xiao-Cai-2/publication/258945832_Multi-View_K-Means_Clustering_on_Big_Data/links/0a85e5304c2a14700f000000/Multi-View-K-Means-Clustering-on-Big-Data.pdf)

### Incomplete MvC

 - [ ] PIC [Spectral Perturbation Meets Incomplete Multi-view Data](https://arxiv.org/pdf/1906.00098)
 - [ ] PVC [Partial multi-view clustering](https://ojs.aaai.org/index.php/AAAI/article/view/8973)

## Performance

### Consensus MvC

|          |        |  Mfeat |           |        | Caltech101-7 |           |
|----------|:------:|:------:|:---------:|:------:|:------------:|:---------:|
| Methods  | Acc    | NMI    | F-measure | Acc    | NMI          | F-measure |
| SC_best  | 0.829  | 0.843  | 0.802     | 0.636  | 0.518        | 0.282     |
| CLR_best | 0.832  | 0.838  | 0.802     | 0.712  | 0.526        | 0.267     |
| SwMC     | 0.858  | 0.888  | 0.838     | 0.651  | 0.582        | 0.319     |
| PwMC     | 0.860  | 0.885  | 0.839     | 0.652  | 0.548        | 0.320     |
| CAN_best | 0.876  | 0.904  | 0.844     | 0.723  | 0.435        | 0.248     |
| MLAN     | 0.952  | 0.913  | 0.954     | 0.816  | 0.550        | 0.379     |
| Co-reg   | 0.964  | 0.931  | 0.964     | 0.651  | 0.550        | 0.308     |
| AMGL     | 0.876  | 0.894  | 0.861     | 0.674  | 0.650        | 0.392     |
