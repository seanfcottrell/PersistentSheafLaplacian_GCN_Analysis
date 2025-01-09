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

class SheafTopologySignificance():
    def __init__(self, gcn, gene, radii):
        self.gcn = gcn
        self.gene = gene
        self.radii = radii

    def psl(self):
        G = self.gcn.copy()
        sheaf = np.ones_like(np.array(G.nodes), dtype=float)
        sheaf[list(G.nodes).index(self.gene)] = -1
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def constant_psl(self):
        G = self.gcn.copy()
        sheaf = np.ones_like(np.array(G.nodes), dtype=float)
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def eigs(self):
        psl0s, psl1s = self.psl()
        cpsl0s, cpsl1s = self.constant_psl()
        ### psl eigs
        psl_eigs0,psl_eigs1,cpsl_eigs0,cpsl_eigs1 = [], [], [], []
        for laplacian in psl0s:
            eigs = np.linalg.eigvals(laplacian)  
            psl_eigs0.append(eigs)
        for laplacian in psl1s:
            eigs = np.linalg.eigvals(laplacian)  
            psl_eigs1.append(eigs)
        ### pl eigs
        for laplacian in cpsl0s:
            eigs = np.linalg.eigvals(laplacian)  
            cpsl_eigs0.append(eigs)
        for laplacian in cpsl1s:
            eigs = np.linalg.eigvals(laplacian)  
            cpsl_eigs1.append(eigs)
        return psl_eigs0,psl_eigs1,cpsl_eigs0,cpsl_eigs1

    def summary_stats(self, eigenvalues):
        # list of summary statistics for each snapshot of the filtration
        summary_stats = []
        for eigs in eigenvalues:
            eigs = np.where(eigs < 1e-6, 0, eigs)
            # summary statistics of spectrum
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
        psl_eigs0,psl_eigs1,cpsl_eigs0,cpsl_eigs1 = self.eigs()
        # arrays of PSL l0 and l1 summary stats for each stage of filtration
        psl_0feats = self.summary_stats(psl_eigs0)
        psl_1feats = self.summary_stats(psl_eigs1)
        # Combine PSL l0 and l1 spectrum stats into a single list
        psl_feats = []
        for l0_stats, l1_stats in zip(psl_0feats, psl_1feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            psl_feats.append(combined)
        psl_feats = np.array(psl_feats)

        # constant PSL l0 and l1 summary stats for each stage of filtration
        cpsl0_feats = self.summary_stats(cpsl_eigs0)
        cpsl1_feats = self.summary_stats(cpsl_eigs1)
        # Combine constant PSL l0 and l1 spectrum stats into a single list
        cpsl_feats = []
        for l0_stats, l1_stats in zip(cpsl0_feats, cpsl1_feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            cpsl_feats.append(combined)
        cpsl_feats = np.array(cpsl_feats)

        # sheaf topological significance at each stage of filtration (wasserstein dist between PSL and constant PSL feature vectors)
        wasserstein_distances = []
        for scale in range(psl_feats.shape[0]):
            # Extract the features for the current scale
            psl_feats_scale = psl_feats[scale, :]
            cpsl_feats_scale = cpsl_feats[scale, :]
            # calculate wasserstein distance between the PSL feats and PL feats at that scale
            distance = wasserstein_distance(psl_feats_scale, cpsl_feats_scale)
            wasserstein_distances.append(distance)
        return wasserstein_distances
    
###############################################################

class LogFC_SheafSignificance():
    def __init__(self, gcn, gene, radii):
        self.gcn = gcn
        self.gene = gene
        self.radii = radii

    def psl(self):
        G = self.gcn.copy()
        LogFCs = nx.get_node_attributes(G, 'logFCs')
        sheaf = [LogFCs.get(node, 0) for node in G.nodes()]
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def perturbed_psl(self):
        G = self.gcn.copy()
        LogFCs = nx.get_node_attributes(G, 'logFCs')
        sheaf = [LogFCs.get(node, 0) for node in G.nodes()]
        G.remove_node(self.gene)
        adj, map = map_graph_nodes(G)
        psl = PSL(adj,sheaf,self.radii)
        psl.build_filtration()
        psl.build_simplicial_pair()
        psl.build_matrices()
        l0s, l1s = psl.psl_0(), psl.psl_1()
        return l0s,l1s

    def eigs(self):
        psl0s, psl1s = self.psl()
        ppsl0s, ppsl1s = self.perturbed_psl()
        ### psl eigs
        psl_eigs0,psl_eigs1,ppsl_eigs0,ppsl_eigs1 = [], [], [], []
        for laplacian in psl0s:
            eigs = np.linalg.eigvals(laplacian)  
            psl_eigs0.append(eigs)
        for laplacian in psl1s:
            eigs = np.linalg.eigvals(laplacian)  
            psl_eigs1.append(eigs)
        ### pl eigs
        for laplacian in ppsl0s:
            eigs = np.linalg.eigvals(laplacian)  
            ppsl_eigs0.append(eigs)
        for laplacian in ppsl1s:
            eigs = np.linalg.eigvals(laplacian)  
            ppsl_eigs1.append(eigs)
        return psl_eigs0,psl_eigs1,ppsl_eigs0,ppsl_eigs1

    def summary_stats(self, eigenvalues):
        # list of summary statistics for each snapshot of the filtration
        summary_stats = []
        for eigs in eigenvalues:
            eigs = np.where(eigs < 1e-6, 0, eigs)
            # summary statistics of spectrum
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
        psl_eigs0,psl_eigs1,ppsl_eigs0,ppsl_eigs1 = self.eigs()
        # arrays of PSL l0 and l1 summary stats for each stage of filtration
        psl_0feats = self.summary_stats(psl_eigs0)
        psl_1feats = self.summary_stats(psl_eigs1)
        # Combine PSL l0 and l1 spectrum stats into a single list
        psl_feats = []
        for l0_stats, l1_stats in zip(psl_0feats, psl_1feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            psl_feats.append(combined)
        psl_feats = np.array(psl_feats)

        # constant PSL l0 and l1 summary stats for each stage of filtration
        ppsl0_feats = self.summary_stats(ppsl_eigs0)
        ppsl1_feats = self.summary_stats(ppsl_eigs1)
        # Combine constant PSL l0 and l1 spectrum stats into a single list
        ppsl_feats = []
        for l0_stats, l1_stats in zip(ppsl0_feats, ppsl1_feats):
            combined = np.concatenate((l0_stats, l1_stats))  # Combine the two arrays
            ppsl_feats.append(combined)
        ppsl_feats = np.array(ppsl_feats)

        # sheaf topological significance at each stage of filtration (wasserstein dist between PSL and perturbed PSL feature vectors)
        wasserstein_distances = []
        for scale in range(psl_feats.shape[0]):
            # Extract the features for the current scale
            psl_feats_scale = psl_feats[scale, :]
            ppsl_feats_scale = ppsl_feats[scale, :]
            # calculate wasserstein distance between the PSL feats and perturbed PSL feats at that scale
            distance = wasserstein_distance(psl_feats_scale, ppsl_feats_scale)
            wasserstein_distances.append(distance)
        return wasserstein_distances
