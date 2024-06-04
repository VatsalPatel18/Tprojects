"""
Copyright Betteromics 2020

Functions to evaluate model performance and feature importance
"""
from typing import List

import sklearn
import sklearn.decomposition
import numpy as np
import matplotlib.pyplot as plt


def get_top_n(A, n=10, bottom=False):
    """
    Returns top n elements in list A as a list of tuples.
    For each tuple, first element is an element of A and the second
    element is its original index in A.  While absolute weight of
    coefficients is used to find top_n, the actual signed weight is
    returned to indicate if feature is a positive or negative one.
    Example:
    test_array = [6,5,4,3,-20,1,9,8,-7,10, 15]
    feature_importance.get_top_n(test_array, 6)
    [(-7, 8), (8, 7), (9, 6), (10, 9), (15, 10), (-20, 4)]
    """
    if len(A) <= n:
        return [(a, i) for i, a in enumerate(A)]
    abs_A = np.abs(A)
    index = range(len(abs_A))
    index_sorted_by_data = sorted(index, key=lambda i: abs_A[i], reverse=bottom)
    top_n_index = index_sorted_by_data[-n:]
    subset = [(A[i], i) for i in top_n_index]
    return subset


def get_top_n_coefficients(model, n=20):
    """
    Given a model, looks at its coefficients (model weights) and grabs
    top 10 (largest) and bottom 10 (smallest) coefficients. Returns a list
    of tuples where in each tuple, firt items is the weight and the second
    item is its index (feature index).
    """
    coeff = []
    try:
        coeff = model.coef_[0]
    except AttributeError:
        coeff = model.feature_importances_
    assert len(coeff) > 0
    top_contributors = get_top_n(coeff, n)
    return top_contributors


def get_top_feature_dict(model, feature_names: List[str], n=20):
    """
    Given a trained model and a list of feature names (str), finds indices for
    features contributing the most to the model (highest abs weights) and returns
    a dictionary mapping feature names to their weights for these top contributors
    """
    top_features = get_top_n_coefficients(model, n=n)
    contributing_features = {}
    for w, i in top_features:
        contributing_features[feature_names[i]] = w
    return contributing_features


def get_pca_features(
    pca: sklearn.decomposition.PCA,
    component_index: int = 0,
    plot_path: str = "pca_weights_distribution.png",
) -> List[int]:
    """
    Extracts features that are contributing to the top PCA component the most.
    """
    if pca.n_components_ < component_index:
        raise Exception(f"Only computed {pca.n_components_} components, requested {component_index}th")
    # assumes PCA was fitted on data
    print("Explained variance", pca.explained_variance_ratio_)
    component_weights = pca.components_[component_index]
    # remove 0-weight features (no contribution), add indices
    zipped = [(w, i) for i, w in enumerate(component_weights) if w != 0]
    component_weights, index = zip(*zipped)
    n = 10
    top_10 = get_top_n(component_weights, n)
    # top_10 index is relative to the filtered zipped weigths. need to remap back to the original
    # feature index
    top_10_orig = [(w, index[i]) for w, i in top_10]
    if len(component_weights) > 2 * n:
        bottom_10 = get_top_n(component_weights, n, bottom=True)
        bottom_10_orig = [(w, index[i]) for w, i in bottom_10]
        tuples = top_10_orig + bottom_10_orig
    else:
        tuples = top_10_orig
    weights, feature_index = zip(*tuples)

    # plot all weights
    plt.clf()
    plt.hist(component_weights, bins=100, density=True, facecolor='g', alpha=0.75)
    plt.grid(True)
    plt.title(f"All non-zero PCA weights on {component_index+1} component")
    plt.savefig(plot_path)
    print(f"Plot saved to {plot_path}")

    return tuples
