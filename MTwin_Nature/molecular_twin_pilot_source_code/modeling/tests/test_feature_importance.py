from dataset_manager.external_projects.cedars.modeling.feature_importance import \
    get_top_n, get_pca_features
import unittest

from sklearn.decomposition import PCA


class TestFeatureImportance(unittest.TestCase):

    def test_get_top_n(self):
        result = get_top_n([])
        self.assertEqual(len(result), 0)

        A = [4, 2, 3, 1]
        result = get_top_n(A, n=2)
        self.assertEqual(len(result), 2)
        # sort tuples by index (2nd item in the tuple)
        self.assertEqual(sorted(result, key=lambda x: x[1]), [(4, 0), (3, 2)])

        result = get_top_n(A, n=2, bottom=True)
        self.assertEqual(len(result), 2)
        # sort tuples by index (2nd item in the tuple)
        self.assertEqual(sorted(result, key=lambda x: x[1]), [(2, 1), (1, 3)])

    def test_get_pca_features(self):
        # data varies in the 1st column, a little less in the 3nd, and is const in the 2nd.
        # PCA will assign the most weight to the 1st component, then 3dd, and 0 to 2nd
        data = [
            [0, 1, 0],
            [1, 1, 0],
            [2, 1, 0],
            [0, 1, 1],
        ]
        pca = PCA(n_components=2)
        pca.fit(data)
        feature_tuples = get_pca_features(pca, component_index=0)
        self.assertEqual(len(feature_tuples), 2)
        feature_tuples = sorted(feature_tuples, key=lambda x: abs(x[0]))
        self.assertEqual(feature_tuples[0][1], 2)
        self.assertEqual(feature_tuples[1][1], 0)
