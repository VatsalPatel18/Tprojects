# get data for PCA
import argparse
import os
import re
from datetime import datetime
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.decomposition import PCA

import dataset_manager.external_projects.cedars.modeling.feature_importance as FI


def read_data(path: str) -> pd.DataFrame:
    print("Reading")
    t_before = datetime.now()
    df = pd.read_csv(path)
    t_after = datetime.now()
    print("reading took", t_after - t_before)
    return df


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i",
        "--input",
        help="Input csv file",
        required=True)
    # column names to exclude
    parser.add_argument(
        "-x",
        "--exclude",
        help="Regular expression for column names to be excluded",
        required=False)
    parser.add_argument(
        "-n",
        "--include",
        help="Regular expression for column names to include. Can not be combined w/ --exclude",
        required=False)
    parser.add_argument(
        "--log",
        help="Apply a log transform to data before PCA (e.g. for RNA data)",
        action="store_true",
        required=False)
    parser.add_argument(
        "-o",
        "--output",
        help="Output directory",
        required=True)
    parser.add_argument(
        "-v",
        "--outcomes",
        help="Outcome variables",
        required=False)
    args = parser.parse_args()
    return args


def get_na_row_index(df: pd.DataFrame) -> List:
    bool_index = df.isna().all(axis=1)
    row_index = [i for i, b in enumerate(bool_index) if b]
    return row_index


def get_column_index(
    col_names: List[str],
    include_regex: str = None,
    exclude_regex: str = None
) -> List[int]:
    keep_column_index_inc = None
    keep_column_index_exc = None
    if exclude_regex:
        pattern = exclude_regex
        print("Will exclude columns matching the expression:", pattern)
        prog = re.compile(pattern)
        keep_column_index_exc = [i for i, col_name in enumerate(col_names) if not prog.match(col_name)]
        if len(keep_column_index_exc) == 0:
            raise Exception("All columns will be excluded")

    if include_regex:
        pattern = include_regex
        print("Will only keep columns matching the expression:", pattern)
        prog = re.compile(pattern)
        keep_column_index_inc = [i for i, col_name in enumerate(col_names) if prog.match(col_name)]
        if len(keep_column_index_inc) == 0:
            raise Exception("No columns will be included")

    # combine two lists
    keep_column_index = None
    if not keep_column_index_inc:
        keep_column_index = keep_column_index_exc
    elif not keep_column_index_exc:
        keep_column_index = keep_column_index_inc
    else:  # neither is None
        keep_column_index = list(set(keep_column_index_inc).intersection(set(keep_column_index_exc)))
    return keep_column_index


def main(args):
    data = read_data(args.input)
    print(data)

    sample_ids = None
    if "sample" in data.columns:
        sample_ids = data["sample"]

    outcomes = None
    if args.outcomes:
        pattern = args.outcomes
        print("Will analyse PCA projection for these outcome variables:", pattern)
        prog = re.compile(pattern)
        tuples = [(i, col_name) for i, col_name in enumerate(data.columns) if prog.match(col_name)]
        if len(tuples) == 0:
            print(f"No columns matched {pattern}")
        else:
            column_index, outcome_column_names = zip(*tuples)
            column_index = list(column_index)
            remaining_column_index = column_index[: min(10, len(column_index))]
            remaining_column_names = sorted(data.columns[remaining_column_index])
            remaining_column_names = ",".join(remaining_column_names)
            print(f"Outcome columns: {len(column_index)} {remaining_column_names}...")
            outcomes = data.iloc[:, column_index]

    keep_column_index = get_column_index(data.columns, include_regex=args.include, exclude_regex=args.exclude)
    if keep_column_index:
        col_names_idx = keep_column_index[: min(10, len(keep_column_index))]
        col_names_str = sorted(data.columns[col_names_idx])
        col_names_str = ",".join(col_names_str)
        print(f"Keeping {len(keep_column_index)} columns: {col_names_str}")
        X = data.iloc[:, keep_column_index]
    else:
        X = data

    # exclude rows that have only NAs -- no data for those samples
    na_row_index = get_na_row_index(X)
    X = X.drop(na_row_index)
    outcomes = outcomes.drop(na_row_index)
    if sample_ids is not None:
        sample_ids = sample_ids.drop(na_row_index)

    # save intermediate data
    fname = os.path.join(args.output, "filtered_columns.csv")
    X.to_csv(fname, index=False)
    print(f"Saved intermediate data to {fname}")

    # replace NAs w/ 0
    X = X.fillna(value=0)

    if args.log:
        # log transform data before PCA
        X = X + 1
        X = np.log(X)
        print(X)

    # scale variables w/in columns
    min_max_scaler = preprocessing.MinMaxScaler()
    X_scaled = min_max_scaler.fit_transform(X)
    print(X_scaled)

    # extract top 2 components and get back the features contributing the most (by weight)
    pca = PCA(n_components=2)
    pca.fit(X_scaled)
    output_fname = os.path.join(args.output, "pca_weights_clinical_data.png")
    feature_tuples = FI.get_pca_features(
        pca,
        component_index=0,
        plot_path=output_fname)
    weights, index = zip(*feature_tuples)
    contributors = [(X.columns[feature_index], weights[i], feature_index) for i, feature_index in enumerate(index)]
    # sort by weight
    contributors = sorted(contributors, key=lambda x: abs(x[1]), reverse=True)
    contributors_str = [f'{label}: {str("%.6f" % round(weight, 6))}' for label, weight, index in contributors]
    print("Top contributing features: ", "\n".join(contributors_str))

    transformed_X = pca.transform(X_scaled)
    if sample_ids is not None:
        pca_coords_data = pd.DataFrame(data=transformed_X, columns=["pca_x", "pca_y"])
        pca_coords_data["sample"] = sample_ids
        print("PCA coords")
        print(pca_coords_data)
        pca_coord_path = os.path.join(args.output, "projected_data.csv")
        pca_coords_data.to_csv(pca_coord_path, index=False)

    pca_X, pca_Y = zip(*transformed_X)
    for label, weight, index in contributors:
        contributing_feature = X[label]
        plt.clf()
        plt.scatter(pca_X, pca_Y, c=contributing_feature, alpha=0.5, cmap='Blues', edgecolor='black')
        plt.grid(True)
        plt.title(f"Plotting data into PCA space using first 2 PCA components\nColor by {label}")
        esc_label = label.replace("/", "_")
        output_fname = os.path.join(args.output, f"data_in_pca_space_by_{esc_label}.png")
        plt.savefig(output_fname)

    if args.outcomes and outcomes is not None:
        # color PCA plot by the outcome variable, compute distances w/in and between groups
        for label in outcome_column_names:
            contributing_feature = outcomes[label]
            plt.clf()
            plt.scatter(pca_X, pca_Y, c=contributing_feature, alpha=0.5, cmap='Blues', edgecolor='black')
            plt.grid(True)
            plt.title(f"Plotting data into PCA space using first 2 PCA components\nColor by {label}")
            output_fname = os.path.join(args.output, f"outcome_data_in_pca_space_by_{label}.png")
            plt.savefig(output_fname)


if __name__ == "__main__":
    args = parse_arguments()
    main(args)
