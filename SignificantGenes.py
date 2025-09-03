import concurrent.futures
from SheafLaplacianSignificance import LogFC_SheafSignificance
import scanpy as sc
import numpy as np
import pandas as pd
from tqdm import tqdm

##########
def LogFC_SheafTopologicalSignificances(args):
    # args = (network, gene, radii)
    sts = LogFC_SheafSignificance(args[0], args[1], args[2])
    significances = sts.topological_significance()
    return significances

def LogFC_SheafTopologicalSignificancesParallelComputation(args):
    # args = (network, gene, radii)
    with concurrent.futures.ProcessPoolExecutor() as executor:
        topological_significances = list(
            tqdm(
                executor.map(LogFC_SheafTopologicalSignificances, args),
                total=len(args),
                desc="Log FC Sheaf Laplacian Spectrum Analysis",
                unit="Gene Specific Perturbations"
                )
            )
    return topological_significances
def intersect_top_genes(scores, gene_names, top_n):
    """
    Finds genes that are within the top 'top_n' scores across all filtration scales.

    Parameters:
    - scores: numpy array of shape (num_genes, num_scales), scores of genes across scales.
    - gene_names: list of gene names of length num_genes.
    - top_n: integer, number of top genes to select from each scale.

    Returns:
    - top_genes: list of gene names that are in the top 'top_n' across all scales.
    """
    scores = np.array(scores)
    print('num genes: ', scores.shape[0])
    print('num scales: ', scores.shape[1])

    top_indices_per_scale = []

    for scale_idx in range(scores.shape[1]):  # Loop over each scale 
        # get scores for each scale
        scale_scores = scores[:, scale_idx]
        #print(scale_scores)
        # sort scores
        sorted_indices = np.argsort(-scale_scores)
        # Select the top scores of that scale
        top_indices = sorted_indices[:top_n]
        top_indices_per_scale.append(set(top_indices))

    # Find the intersection of top scores over all scales
    top_genes_in_all_scales_indices = set.intersection(*top_indices_per_scale)
    top_genes_indices = sorted(list(top_genes_in_all_scales_indices))
    top_genes = [gene_names[idx] for idx in top_genes_indices]

    return top_genes

