import scanpy as sc
import numpy as np
import pandas as pd
import warnings
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

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
        #self.preprocessing()
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
        degs_sorted = degs_filtered.sort_values(by='scores', key=lambda x: x.abs(), ascending=False)
        top_n_degs = degs_sorted.head(self.n)[['names', 'logfoldchanges']]

        adata2 = adata2[adata2.obs[self.column] == self.cluster]
        adata2 = adata2[:, top_n_degs['names']]
        adata2.var['logfoldchanges'] = top_n_degs.set_index('names')['logfoldchanges']
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
        logfc_series = adata.var['logfoldchanges']
        logfc_dict = logfc_series.to_dict()
        nx.set_node_attributes(G, logfc_dict, 'logFCs')
        return gene_names, G

############################################## Plotting #######################################################

def plot_network(G, stage, n_labels):
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    pos = nx.kamada_kawai_layout(G)
    LogFCs = nx.get_node_attributes(G, 'logFCs')
    sorted_genes = sorted(LogFCs.keys(), key=lambda x: abs(LogFCs[x]), reverse=True)
    top_genes = sorted_genes[:n_labels]
    node_sizes = [abs(LogFCs.get(node, 0)) * 400 for node in G.nodes()]
    node_colors = [LogFCs.get(node, 0) for node in G.nodes()]

    nx.draw_networkx_edges(
        G, pos,
        alpha=0.5,
        edge_color='gray',
        width=0.5,
        ax=ax
    )
    nodes = nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        cmap='coolwarm',  
        alpha=0.8,
        ax=ax,
        edgecolors='white'
    )
    
    norm = mcolors.Normalize(vmin=min(node_colors), vmax=max(node_colors))
    sm = plt.cm.ScalarMappable(cmap='coolwarm', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(
    sm, 
    ax=ax, 
    shrink=0.75, 
    orientation='horizontal',  
    pad=0.05                    
    )
    cbar.set_label('Log2FC', fontsize=25)
    cbar.ax.tick_params(labelsize=25)

    label_x = 1.05
    sorted_pos = sorted([(gene, pos[gene][1]) for gene in top_genes], key=lambda x: x[1])
    vertical_padding_factor = 1.2  
    min_y = min([pos[gene][1] for gene in top_genes])
    max_y = max([pos[gene][1] for gene in top_genes])
    y_range = (max_y - min_y) * vertical_padding_factor

    for idx, (gene, y) in enumerate(sorted_pos):
        y_label = np.linspace(min_y - (y_range - (max_y - min_y))/2, 
                            max_y + (y_range - (max_y - min_y))/2, 
                            n_labels)[idx]
        ax.text(
            label_x, y_label,
            gene,
            fontsize=25,          
            color='black',     
            ha='left',
            va='center'
        )
        ax.plot(
            [pos[gene][0], 1.0],  
            [pos[gene][1], y_label],  
            color='black',
            linewidth=0.75,
            linestyle='--',
            alpha=0.7
        )
    
    x_values, y_values = zip(*pos.values())
    ax.set_xlim(min(x_values)-0.1, 1.2)
    ax.set_ylim(min(y_values)-0.1, max(y_values)+0.1)
    plt.title(f'{stage}', fontsize=25)
    plt.axis('off')
    plt.show()

def DEG_plot(df, N):
    df['nlog10'] = -np.log10(df.pvals)
    df['Significance'] = np.where(df['nlog10'] > 5, 'Significant', 'Not Significant')
    df['Regulation'] = np.where(df['log2FoldChange'] > 0, 'Upregulated', 'Downregulated')
    conditions = [
        (df['nlog10'] > 5) & (df['log2FoldChange'] > 1),
        (df['nlog10'] > 5) & (df['log2FoldChange'] < -1),
        (df['nlog10'] >= 5)
    ]
    choices = ['Upregulated', 'Downregulated', 'Not Significant']
    df['Gene Regulation'] = np.select(conditions, choices, default='Not Significant')

    palette = {
        'Upregulated': '#DB3F3F',       
        'Downregulated': '#3B4CC0',    
        'Not Significant': 'grey',   
    }

    top_upregulated = df[df['Regulation'] == 'Upregulated'].nlargest(N, 'log2FoldChange')
    top_downregulated = df[df['Regulation'] == 'Downregulated'].nsmallest(N, 'log2FoldChange')
    top_genes = pd.concat([top_upregulated, top_downregulated])

    sns.set(style="whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams['font.size'] = 25
    plt.rcParams['axes.labelweight'] = 'normal'  
    plt.rcParams['axes.titleweight'] = 'normal'  
    plt.figure(figsize=(12, 8))
    ax = sns.scatterplot(
        data=df,
        x='log2FoldChange',
        y='nlog10',
        hue='Gene Regulation',
        palette=palette,
        s=100,
        edgecolor='white',
        linewidth=1,
        legend='full'
    )

    ax.axhline(5, color='k', linestyle='--', linewidth=2)
    ax.axvline(1, zorder=0, c='k', lw=2, ls='--')
    ax.axvline(-1, zorder=0, c='k', lw=2, ls='--')
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    x_text = x_max + (x_max - x_min) * 0.05  
    ax.set_xlim(x_min, x_text + (x_max - x_min) * 0.1) 
    labels_data = []

    for i in range(len(df)):
        if df.iloc[i].names in list(top_genes.names):
            y_point = df.iloc[i].nlog10
            x_point = df.iloc[i].log2FoldChange
            label = df.iloc[i].names
            labels_data.append((y_point, x_point, label))

    labels_data.sort(key=lambda x: x[0])
    n_labels = len(labels_data)
    y_positions = np.linspace(
        y_min + 0.01*(y_max - y_min) * 0.1,
        y_max - 3.5*(y_max - y_min) * 0.1,
        n_labels
    )

    for idx, (y_point, x_point, label) in enumerate(labels_data):
        y_text = y_positions[idx]
        ax.text(x_text+1.5, y_text, label, fontsize=25, va='center')
        ax.plot([x_point, x_text+1], [y_point, y_text], c='k', lw=0.5)

    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title='Gene Regulation', fontsize=25, title_fontsize=25, 
            bbox_to_anchor=(0.5, 1), loc='lower center', ncol=3, frameon=True)

    plt.xlabel("Log2FC",fontsize=25, color='black')
    plt.ylabel("-Log2FDR", fontsize=25,color='black')
    plt.xticks(fontsize=25, color='black')
    plt.yticks(fontsize=25, color='black')

    sns.despine()
    plt.tight_layout()
    plt.show()

def plot_topological_scores(scores2, gene_names):
    scores2 = np.array(scores2)
    average_scores = np.mean(scores2, axis=1)
    std_errors = np.std(scores2, axis=1) / np.sqrt(scores2.shape[1])

    data = pd.DataFrame({
        'Genes': gene_names,
        'Average Score': average_scores,
        'Standard Error': std_errors
    })

    # Select top 15 by average
    top_15 = data.nlargest(15, 'Average Score')
    top_indices = top_15.index
    scales = [f'Filtration Scale {i+1}' for i in range(scores2.shape[1])]

    long_data = []
    for i in top_indices:
        for j in range(scores2.shape[1]):
            long_data.append((gene_names[i], scales[j], scores2[i,j]))
    long_df = pd.DataFrame(long_data, columns=['Genes', 'Scale', 'Score'])

    num_scales = np.array(scores2).shape[1]

    fig, axes = plt.subplots(nrows=num_scales, ncols=1, figsize=(12, 4*num_scales))
    plt.rcParams['font.family'] = 'DejaVu Serif'

    for idx, ax in enumerate(axes):
        scale_name = scales[idx]
        scale_data = long_df[long_df['Scale'] == scale_name].sort_values('Score', ascending=False)
        sns.barplot(x='Genes', y='Score', data=scale_data, palette='tab20', ax=ax)
        ax.set_title(f'{scale_name}', fontsize=28)
        ax.set_xlabel("", fontsize=28)
        ax.set_ylabel("", fontsize=28)
        ax.tick_params(axis='x', which='major', rotation=45, labelsize=28)
        ax.tick_params(axis='y', which='major', labelsize=28)
        ax.yaxis.set_ticks([])
        ax.tick_params(labelleft=False)
        sns.despine(ax=ax)

    plt.tight_layout()
    plt.show()
