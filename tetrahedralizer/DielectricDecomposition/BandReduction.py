"""
Tools for calculating information rich subsets of frequency bands.
"""


from enum import Enum
import numpy as np
import sys
from LookupTables import *
from matplotlib import pyplot as plt
from CosineSimilarity import cosine_similarity, cosine_similarity_plot
from tqdm import trange

def hierarchical_band_reduction(materials, frequencies, max_levels=40):
    """"""

    # # --------- #
    # # Load Data #
    # # --------- #
    #
    # er = []
    # cond = []
    #
    # for material in materials:
    #     a, b = permittivity_lookup(material, frequencies)
    #     # Normalise Spectra
    #     a, b = np.array(a), np.array(b)
    #     er.append(a / np.sqrt(np.dot(a, a)))
    #     cond.append(b / np.sqrt(np.dot(b, b)))

    er_all, cond_all = cosine_similarity(materials, frequencies)

    # --------- #
    # Algorithm #
    # --------- #

    H_best = [i for i in range(len(frequencies))]  # H = Hierarchy
    H_level = 0
    loss_best = sys.float_info.max

    while True:

        H_level += 1
        print('Calculating H_level {}'.format(H_level))

        n_pairs = (len(H_best) - 1)
        pairwise_loss = [0] * n_pairs

        for i in trange(n_pairs):
            freq = [frequencies[H_best[j]] for j in [_ for _ in range(i)] + [_ for _ in range(i + 1,len(H_best))]]
            er, cond = cosine_similarity(materials, freq)
            pairwise_loss[i] = (er_all - er).max() + (cond_all - cond).max()   # TODO: NOTE - this algorithm is not fit for what I am actually trying to achieve...

        # Update if loss improved
        argmin = pairwise_loss.index(min(pairwise_loss))
        # if argmin < loss_best:
        if pairwise_loss[argmin] <= 0:
            del H_best[argmin + 1]  # retain leftmost frequency
            loss_best = pairwise_loss[argmin]
        else:
            print('Stopping: loss not improved.')
            break

        # -------------------- #
        # Test for convergence #
        # -------------------- #
        if H_level >= max_levels:
            print('Stopping: Max Levels Reached.')
            break

    # Return list of frequencies.
    return [frequencies[i] for i in H_best]


if __name__ == '__main__':

    # Prepare Data Range
    _materials = VALID_MATERIALS
    _step_GHz = 0.5
    _frequencies = [f * 1e9 for f in np.arange(MIN_FREQUENCY / 1e9, MAX_FREQUENCY / 1e9, _step_GHz)]

    # Calculate Cosine Similarity Matrices
    _er_matrix, _cond_matrix = cosine_similarity(_materials, _frequencies)

    _freq_subset = hierarchical_band_reduction(_materials, _frequencies)

    # Plot Results
    cosine_similarity_plot(_er_matrix, _materials, title='Permittivity')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())
    cosine_similarity_plot(_cond_matrix, _materials, title='Conductivity')
    plt.show()
    input('Press Enter to Continue...')
    plt.close(plt.gcf())




