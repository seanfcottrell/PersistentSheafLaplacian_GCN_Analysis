import scanpy as sc
import numpy as np
import pandas as pd
import warnings
import os
import sys
import networkx as nx

rootPath = '/mnt/home/cottre61/GFP-GAT/STAGATE_pyG'
warnings.filterwarnings("ignore")

class GeneCoexpressionNetwork():
    def __init__(self, adata, threshold, n, cluster = None, column = None):
        '''
        adata: the adata object containing our gene expression and cluster labels
        threshold: the co expression threshold for constructing the gcn
        n: number of DEGs in the network
        cluster: which cell type / cluster we are interested in
        column: where in adata.obs the cluster labels are contained
        '''
        self.adata = adata
        self.n = n
        self.cluster = cluster
        self.column = column
        self.threshold = threshold

    def preprocessing(self):
        # Normalize counts per cell
        sc.pp.normalize_total(self.adata, target_sum=1e4)
        # Log transform
        sc.pp.log1p(self.adata)
    
    def DEGs(self):
        self.preprocessing()
        adata2 = self.adata.copy()
        adata2.obs[self.column] = adata2.obs[self.column].cat.rename_categories(lambda x: str(x))
        ### DEGs
        sc.tl.rank_genes_groups(adata2, groupby=self.column, groups=[self.cluster], reference='rest', method='wilcoxon')
        results = adata2.uns['rank_genes_groups']
        degs = pd.DataFrame({
            'names': results['names'][self.cluster],  
            'pvals': results['pvals'][self.cluster],
            'logfoldchanges': results['logfoldchanges'][self.cluster],
            'scores': results['scores'][self.cluster]
        })

        # Filter DEGs based on p-value
        degs_filtered = degs[degs['pvals'] < 0.05]
        degs_sorted = degs_filtered.sort_values(by='scores', ascending=False)
        top_n_degs = degs_sorted.head(self.n)['names']

        adata2 = adata2[adata2.obs[self.column] == self.cluster]
        adata2 = adata2[:, top_n_degs]
        return adata2
    
    def GeneCoexpression(self):
        adata = self.DEGs()
        if isinstance(adata.X, np.ndarray):
            expression_data = pd.DataFrame(adata.X, columns=adata.var_names)
        else:
            expression_data = pd.DataFrame(adata.X.toarray(), columns=adata.var_names)
        coexpression_matrix = expression_data.corr(method='pearson')
        return adata, coexpression_matrix
    
    def GCN(self):
        adata, C = self.GeneCoexpression()
        gene_names = adata.var_names
        node_mapping = {gene: idx for idx, gene in enumerate(gene_names)}
        # Reorder the coexpression matrix according to the node mapping
        mapped_indices = [node_mapping[gene] for gene in gene_names]
        coexpression_matrix = np.array(C)[np.ix_(mapped_indices, mapped_indices)]

        mask = np.abs(coexpression_matrix) >= self.threshold
        G = nx.Graph()
        G.add_nodes_from(node_mapping)
        for i in range(len(coexpression_matrix)):
            for j in range(i + 1, len(coexpression_matrix)):  
                if mask[i, j]:  
                    G.add_edge(i, j, weight=(1/np.abs(coexpression_matrix[i, j])))
       
        index_to_gene = {v: k for k, v in node_mapping.items()}
        G = nx.relabel_nodes(G, index_to_gene)
        return gene_names, G

