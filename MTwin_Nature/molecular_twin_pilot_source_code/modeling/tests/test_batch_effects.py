from dataset_manager.external_projects.cedars.modeling.batch_effects import \
    get_na_row_index, get_column_index
import unittest

import pandas as pd
import numpy as np


class TestBatchEffects(unittest.TestCase):

    def test_get_na_row_index(self):
        df = pd.DataFrame([
            [1, np.NaN, 3],
            [np.NaN, np.NaN, np.NaN],
            [0, 0, np.NaN],
            [np.NaN, np.NaN, np.NaN],
            range(3)
        ])
        index = get_na_row_index(df)
        self.assertEqual(len(index), 2)

    def test_get_column_index(self):
        col_names = ["aaa", "aab", "bbb", "ccc"]
        index = get_column_index(col_names)
        # all columns selected
        self.assertTrue(index is None)

        # exclude any that start with "aa"
        index = get_column_index(col_names, exclude_regex="^aa.*")
        # all columns selected
        self.assertEqual(index, [2, 3])

        # only include those that start with "aa"
        index = get_column_index(col_names, include_regex="^aa.*")
        # all columns selected
        self.assertEqual(index, [0, 1])

        # only include those that start with "aa", exclude those that end w/ b
        index = get_column_index(col_names, include_regex="^aa.*", exclude_regex=".*b$")
        # all columns selected
        self.assertEqual(index, [0])

        # edgecase: include will not include any columns (no column has z)
        with self.assertRaises(Exception):
            get_column_index(col_names, include_regex=".*z.*")

        # will exclude all columns
        with self.assertRaises(Exception):
            get_column_index(col_names, exclude=".*")
