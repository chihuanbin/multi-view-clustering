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
 - [x] CAN [Clustering and Projected Clustering with Adaptive Neighbors](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.707.56&rep=rep1&type=pdf) [code]()
 - [x] MLAN [Multi-View Clustering and Semi-Supervised Classification with Adaptive Neighbours](https://www.aaai.org/ocs/index.php/AAAI/AAAI17/paper/viewPDFInterstitial/14833/14423)   [code]()
 - [x] CLR [The constrained laplacian rank algorithm for graph-based clustering](https://ojs.aaai.org/index.php/AAAI/article/download/10302/10161)  [code]()
 - [x] SwMC [Self-weighted Multiview Clustering with Multiple Graphs](https://www.ijcai.org/proceedings/2017/0357.pdf)  [code]()
 - [x] PwMC [Self-weighted Multiview Clustering with Multiple Graphs](https://www.ijcai.org/proceedings/2017/0357.pdf)  [code]()
 - [x] Co-regularization [Co-regularized multi-view spectral clustering](http://www.abhishek.umiacs.io/coregspectral.nips11.pdf)    [code]()
 - [ ] AMGL [Parameter-free auto-weighted multiple graph learning: a framework for multiview clustering and semi-supervised classification](https://www.ijcai.org/Proceedings/16/Papers/269.pdf)
 - [ ] MCGC [Multiview Consensus Graph Clustering]


### MvC for large-scale data
 - [ ] MVSC [Large-scale multi-view spectral clustering via bipartite graph](https://www.aaai.org/ocs/index.php/AAAI/AAAI15/paper/viewPaper/9641)
 - [ ] SMKMC [Multi-View K-Means Clustering on Big Data](https://www.researchgate.net/profile/Xiao-Cai-2/publication/258945832_Multi-View_K-Means_Clustering_on_Big_Data/links/0a85e5304c2a14700f000000/Multi-View-K-Means-Clustering-on-Big-Data.pdf)

### Incomplete MvC

 - [ ] PIC [Spectral Perturbation Meets Incomplete Multi-view Data](https://arxiv.org/pdf/1906.00098)
 - [ ] PVC [Partial multi-view clustering](https://ojs.aaai.org/index.php/AAAI/article/view/8973)

## Performance

### Consensus MvC
|                   |        |  Mfeat |        |        | Caltech101-7 |        |
|-------------------|:------:|:------:|:------:|:------:|:------------:|:------:|
| Methods           | Acc    | NMI    | F      | Acc    | NMI          | F      |
| SC_best           | 0.8290 | 0.8430 | 0.8020 | 0.6357 | 0.5177       | 0.2823 |
| CLR_best          | 0.8315 | 0.8380 | 0.8020 | 0.7120 | 0.5260       | 0.2670 |
| SwMC              | 0.8575 | 0.8876 | 0.8376 | 0.6506 | 0.5820       | 0.3192 |
| PwMC              | 0.8595 | 0.8845 | 0.8388 | 0.6520 | 0.5483       | 0.3195 |
| CAN(best)         | 0.8760 | 0.9043 | 0.8443 | 0.7225 | 0.4347       | 0.2475 |
| MLAN              | 0.9515 | 0.9129 | 0.9538 | 0.8161 | 0.5501       | 0.3792 |
| Co-regularization | 0.9640 | 0.9311 | 0.9643 | 0.6509 | 0.5502       | 0.3077 |
