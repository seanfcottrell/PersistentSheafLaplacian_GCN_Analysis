import numpy as np
from PSL import PSL
import networkx as nx
from scipy.stats import wasserstein_distance

def map_graph_nodes(G):
    node_mapping = {node: idx for idx, node in enumerate(G.nodes())}

    G_mapped = nx.Graph()
    G_mapped.add_nodes_from(node_mapping.values())

    # Add edges, mapping nodes to their new integer identifiers
    for u, v, data in G.edges(data=True):
        G_mapped.add_edge(node_mapping[u], node_mapping[v], **data)

    return G_mapped, node_mapping

class PSL_TopPerturbation():
    def __init__(self, gcn, gene, radii):
        self.gcn = gcn
        self.gene = gene
        self.radii = radii

    def psl(self):
        G = self.gcn.copy()
        sheaf = np.ones_like(np.array(G.nodes), dtype=float)
        #sheaf = 2 * sheaf
        sheaf[list(G.nodes).index(self.gene)] = -1
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def perturbed_psl(self):
        G = self.gcn.copy()
        G.remove_node(self.gene)
        sheaf = np.ones_like(np.array(G.nodes), dtype=float)
        #sheaf = 2 * sheaf
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def psl_eigs(self):
        l0s, l1s = self.psl()
        pl0s, pl1s = self.perturbed_psl()
        ### psl eigs
        eigs0,eigs1,peigs0,peigs1 = [], [], [], []
        for laplacian in l0s:
            eigs = np.linalg.eigvals(laplacian)  
            eigs0.append(eigs)
        for laplacian in l1s:
            eigs = np.linalg.eigvals(laplacian)  
            eigs1.append(eigs)
        ### perturbed psl eigs
        for laplacian in pl0s:
            eigs = np.linalg.eigvals(laplacian)  
            peigs0.append(eigs)
        for laplacian in pl1s:
            eigs = np.linalg.eigvals(laplacian)  
            peigs1.append(eigs)
        return eigs0,eigs1,peigs0,peigs1

    def summary_stats(self, eigenvalues):
        # list of summary statistics for each snapshot of the filtration
        summary_stats = []
        for eigs in eigenvalues:
            eigs = np.where(eigs < 1e-6, 0, eigs)
            # summary statistics spectrum
            non_zero_eigs = eigs[eigs != 0]
            stats = np.array([
            np.min(non_zero_eigs) if non_zero_eigs.size > 0 else 0,  # Min
            np.mean(non_zero_eigs) if non_zero_eigs.size > 0 else 0,  # Mean
            np.max(non_zero_eigs) if non_zero_eigs.size > 0 else 0,  # Max
            np.std(non_zero_eigs) if non_zero_eigs.size > 0 else 0,  # Std Dev
            np.sum(non_zero_eigs) if non_zero_eigs.size > 0 else 0,  # Sum
            np.sum(eigs == 0)  # betti number
            ])
            summary_stats.append(stats)
        return summary_stats
    
    def topological_significance(self):
        eigs0,eigs1,peigs0,peigs1 = self.psl_eigs()
        # arrays of l0 and l1 summary stats for each stage of filtration
        l0feats = self.summary_stats(eigs0)
        l1feats = self.summary_stats(eigs1)
        # Combine l0 and l1 spectrum stats into a single list
        feats = []
        for l0_stats, l1_stats in zip(l0feats, l1feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            feats.append(combined)
        feats = np.array(feats)
        # perturbed l0 and l1 summary stats for each stage of filtration
        pl0feats = self.summary_stats(peigs0)
        pl1feats = self.summary_stats(peigs1)
        # Combine perturbed l0 and l1 spectrum stats into a single list
        pfeats = []
        for l0_stats, l1_stats in zip(pl0feats, pl1feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            pfeats.append(combined)
        pfeats = np.array(pfeats)
        # topological significance at each stage of filtration (wasserstein dist between perturbed and non perturbed feature vectors)
        wasserstein_distances = []
        for scale in range(feats.shape[0]):
            # Extract the features for the current scale
            feats_scale = feats[scale, :]
            pfeats_scale = pfeats[scale, :]
            # calculate wasserstein distance between the feats and perturbed feats at that scale
            distance = wasserstein_distance(feats_scale, pfeats_scale)
            wasserstein_distances.append(distance)
        return wasserstein_distances


