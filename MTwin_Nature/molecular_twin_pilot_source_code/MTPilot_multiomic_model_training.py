# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# Copyright (C) 2022 - Betteromics Inc.
import itertools
import multiprocessing
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from multiprocessing import Pool
# import warnings filter
from warnings import simplefilter

# +
import numpy as np
import pandas as pd
from IPython.core.display import HTML, display
from sklearn import preprocessing, svm
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, RFECV, SelectFromModel
from sklearn.linear_model import (Lasso, LinearRegression, LogisticRegression,
                                  SGDClassifier)
from sklearn.model_selection import (LeaveOneOut, cross_val_predict,
                                     cross_validate)
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from tqdm.auto import tqdm

from modeling import feature_importance

# ignore all future warnings
simplefilter(action='ignore')
#simplefilter(action='ignore', category=FutureWarning)
#simplefilter(action='ignore', category=UserWarning)

NUM_CPU_PARALLEL = multiprocessing.cpu_count()

pd.options.display.max_columns = 500

display(HTML("<style>.container { width:95% !important; }</style>"))

# Take Intersection of available analytes or use maximum number of samples
INTERSECTION_OF_ANALYTES = False

# +
# key_column: "Biobank_Number" (GI-##-### format)
clinical_prefixes = ["clinical_",] # "surgery_embed_", "pathology_embed_", "chemotherapy_embed_"]
protein_prefixes = ["plasma_protein_", "tissue_protein_", "plasma_lipid_",]
pathology_prefixes = ["pathology_NF",]
genomics_prefixes = ["CNV_", "freebayes_SNV_", "pindel_INDEL_",]
transcriptomic_prefixes = ["AF4_", "rna_expr_",]
omics_feature_prefixes = protein_prefixes + pathology_prefixes + genomics_prefixes + transcriptomic_prefixes + clinical_prefixes

numerical_labels = ["label_days_to_death", "label_days_to_recurrence", ]
categorical_labels = ["label_deceased", "label_recurred",] # "label_tnm_stage", "label_outcome_categorical",  "label_tnm_sub_stage"]
# -


merged_df = pd.read_csv("./molecular_twin_pilot/outputs/multiomic/early_stage_patients_multiomic_dataset.csv")
merged_df.shape


# +
def subset_df_by_prefixes(df, prefixes):
    column_list = []
    for prefix in prefixes:
        column_list += df.columns[df.columns.str.startswith(prefix)].tolist()
    return df[column_list]


# Regression Models

def LR_Regression():
    return make_pipeline(
        StandardScaler(),
        LinearRegression()
    )


def PCA_LR_Regression():
    return make_pipeline(
        StandardScaler(),
        PCA(20),
        LinearRegression()
    )


def RF_Regression():
    return make_pipeline(
        StandardScaler(),
        RandomForestClassifier()
    )


# Categorical Models

def PCA_LR_Model():
    return make_pipeline(
        StandardScaler(),
        PCA(20),
        LogisticRegression(max_iter=1000)
    )


def SVM_Model():
    return make_pipeline(
        StandardScaler(),
        SGDClassifier()
    )


def L1_Norm_SVM_Model():
    return make_pipeline(
        StandardScaler(),
        SGDClassifier(alpha=0.05, penalty="l1")
    )

# L1-Norm feature elimination with RandomForestClassifier.
def L1_Norm_RF_Model():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        RandomForestClassifier()
    )

# L1-Norm feature elimination with MultiLayerPerceptron.
def L1_Norm_MLP_Model():
    return make_pipeline(
        StandardScaler(),
        SelectFromModel(SGDClassifier(alpha=0.05, penalty="l1")),
        MLPClassifier(hidden_layer_sizes=(512, 256, 128, 64, 32), max_iter=1000)
    )

# Recursive Feature Elimination with cross-validation.
def RFE_LR_Model():
    return make_pipeline(
        StandardScaler(),
        RFECV(LogisticRegression(max_iter=1000), step=0.2)
    )

# Recursive Feature Elimination with cross-validation.
def RFE_RF_Model():
    return make_pipeline(
        StandardScaler(),
        RFECV(RandomForestClassifier(), step=0.2)
    )


def get_precision_and_recall(actual_labels, test_labels_correct):
    import math
    TP = 0
    FP = 0
    FN = 0
    TN = 0
    for actual_label, correct in zip(actual_labels, test_labels_correct):
        if actual_label and correct:
            TP += 1
        if actual_label and not correct:
            FN += 1
        if not actual_label and correct:
            TN += 1
        if not actual_label and not correct:
            FP += 1
    try:
        accuracy = (TP+TN)/(TP+TN+FP+FN)
    except:
        accuracy = math.nan
    try:
        precision = TP/(TP+FP)
    except:
        precision = math.nan
    try:
        recall = TP/(TP+FN)
    except:
        recall = math.nan
    try:
        Spec = TN/(TN+FP)
    except:
        Spec = math.nan
    try:
        F1 = TP/(TP+((FP+FN)/2))
    except:
        F1 = math.nan
    try:
        PPV = TP/(TP+FP)
    except:
        PPV = math.nan
    try:
        NPV = TN/(TN+FN)
    except:
        NPV = math.nan
    return {
        "FP": FP,
        "TN": TN,
        "TP": TP,
        "FN": FN,
        "precision": precision,
        "Sens": recall,
        "Spec": Spec,
        "F1": F1,
        "PPV": PPV,
        "NPV": NPV
    }

def run_cross_validation_get_prob(model, X, y) -> dict:
    loo = LeaveOneOut()

    return cross_val_predict(
        estimator=model,
        X=X,
        y=y,
        cv=loo,
        method='predict_proba'
    )


def run_cross_validation(model, X, y, scoring_metric='balanced_accuracy') -> dict:
    loo = LeaveOneOut()

    scores_dict = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=loo,
        scoring=scoring_metric,
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        return_train_score=True,
        return_estimator=True,
    )
    return scores_dict



def calculate_leave_one_out_metric(cross_validation_dict):
    return cross_validation_dict['test_score'].mean()


def test_feature_target_combo(model, features, labels, scoring_metric='balanced_accuracy'):
    cross_validation_dict = run_cross_validation(model, features, labels, scoring_metric)
    LOO_test_metric = calculate_leave_one_out_metric(cross_validation_dict)
    return cross_validation_dict, LOO_test_metric


# -

# # Remove Samples that are missing any feature class in entirety

if INTERSECTION_OF_ANALYTES:
    singleomics_column_prefixes = omics_feature_prefixes + ['label_']
    features = subset_df_by_prefixes(merged_df, singleomics_column_prefixes)
    for prefix in singleomics_column_prefixes:
        singleomic_columns = features.columns[features.columns.str.contains(prefix)].tolist()
        features.dropna(axis=0, how='all', subset=singleomic_columns, inplace=True)
    merged_df = features

merged_df.shape

# # Binary Categorical

label_df = subset_df_by_prefixes(merged_df, ["label_"])
label_df.head(20)
# +
# Generate combinations of input feature categories for classification as follows:
feature_combos = []

# 1. Clinical & path/chemo/surgery note NLP features as baseline models
feature_combos += (list(itertools.combinations(list(set(clinical_prefixes)), 1)))
print(len(feature_combos))

# 2. Each feature category independently as baseline single-omic models
feature_combos += (list(itertools.combinations(omics_feature_prefixes, 1)))
print(len(feature_combos))

# 3. Each pair-wise feature category as pair-wise information gain models
feature_combos += (list(itertools.combinations(omics_feature_prefixes, 2)))
print(len(feature_combos))

# 4. All N and N-1 combinations of analytes for multi-omic models
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 10)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 9)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 8)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 7)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 6)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 5)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 4)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 3)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 2)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes) - 1)))
feature_combos += (list(itertools.combinations(omics_feature_prefixes, len(omics_feature_prefixes))))

feature_combos = list(set(feature_combos))
feature_combos


# +
model_funcs = [SVM_Model, PCA_LR_Model, L1_Norm_SVM_Model, L1_Norm_RF_Model, # Fast Models 
               L1_Norm_MLP_Model, RFE_LR_Model, RFE_RF_Model] # Slow Models

def process_prefix(prefix):
    rows = []
    features = subset_df_by_prefixes(merged_df, prefix)

    null_idx = ~features.isnull().any(axis=1)
    subsetted_features = features[null_idx]
    num_samples = len(subsetted_features)

    for categorical_label in categorical_labels:
        labels = label_df[categorical_label]
        subsetted_labels = labels[null_idx]
        #subsetted_labels = subsetted_labels[kras_mask]
         
        for model_func in model_funcs:
            try:
                model = model_func()

                score_dict, test_metric = test_feature_target_combo(
                    model=model,
                    features=subsetted_features,
                    labels=subsetted_labels,
                    scoring_metric='balanced_accuracy'
                )


                model_dict = {
                    "feature_prefix": prefix,
                    "num_samples": num_samples,
                    "test_loo_metric": test_metric,
                    "target_label": categorical_label,
                    "num_input_features": len(subsetted_features.columns),
                    "model_type": model_func.__name__
                }

                # Compute top features for SGD_LR_Model
                if model_func == L1_Norm_SVM_Model or model_func == SVM_Model:
                    top_weights = []
                    top_weights_total = {}
                    n_features = 15
                    for estimator in score_dict['estimator']:
                        top_weights_dict = feature_importance.get_top_feature_dict(model=estimator[-1],
                                                                                   feature_names=subsetted_features.columns,
                                                                                   n=n_features)
                        top_weights += list(top_weights_dict.keys())
                        top_weights_total.update(Counter(top_weights_dict))

                    top_weights_frequency = dict(Counter(top_weights))
                    num_samples = len(subsetted_features)
                    top_weights_frequency = {k: v/num_samples for k, v in top_weights_frequency.items()}
                    top_weights_mean = {k: tot/top_weights_frequency[k] for k, tot in top_weights_total.items()}

                    i = 1
                    for feature_name, freq in sorted(top_weights_frequency.items(), key=lambda item: item[1], reverse=True):
                        model_dict[f"top_{n_features}_feature_{i}"] = f'{feature_name} feat_freq:{freq:.4f}, feat_weight_mean:{top_weights_mean[feature_name]:.0f}'
                        i += 1

                # Compute top features for L1_Norm_RF_Model
                elif model_func == L1_Norm_RF_Model:
                    top_weights = []
                    top_weights_total = {}
                    n_features = 15
                    for estimator in score_dict['estimator']:
                        chosen_variables = estimator[1].get_support()
                        top_weights_dict = feature_importance.get_top_feature_dict(model=estimator[-1],
                                                                                   feature_names=subsetted_features.columns[chosen_variables],
                                                                                   n=n_features)
                        top_weights += list(top_weights_dict.keys())
                        top_weights_total.update(Counter(top_weights_dict))

                    top_weights_frequency = dict(Counter(top_weights))
                    num_samples = len(subsetted_features)
                    top_weights_frequency = {k: v/num_samples for k, v in top_weights_frequency.items()}
                    top_weights_mean = {k: tot/top_weights_frequency[k] for k, tot in top_weights_total.items()}

                    i = 1
                    for feature_name, freq in sorted(top_weights_frequency.items(), key=lambda item: item[1], reverse=True):
                        model_dict[f"top_{n_features}_feature_{i}"] = f'{feature_name} feat_freq:{freq:.4f}, feat_weight_mean:{top_weights_mean[feature_name]:.0f}'
                        i += 1
                elif model_func == RFE_LR_Model or model_func == RFE_RF_Model:
                    top_weights = []
                    n_features = 15
                    for estimator in score_dict['estimator']:
                        # https://stackoverflow.com/questions/51181170/selecting-a-specific-number-of-features-via-sklearns-rfecv-recursive-feature-e
                        feature_ranks = estimator[1].ranking_  # selector is a RFECV fitted object
                        feature_ranks_with_idx = enumerate(feature_ranks)
                        sorted_ranks_with_idx = sorted(feature_ranks_with_idx, key=lambda x: x[1])
                        top_n_idx = [idx for idx, rnk in sorted_ranks_with_idx[:n_features]]
                        top_weights += list(subsetted_features.columns[top_n_idx].values)

                    top_weights_frequency = dict(Counter(top_weights))
                    top_weights_frequency = {k: v/num_samples for k, v in top_weights_frequency.items()}
                    i = 1
                    for key, value in sorted(top_weights_frequency.items(), key=lambda item: item[1], reverse=True):
                        model_dict[f"top_{n_features}_feature_{i}"] = f'{key}:{value}'
                        i += 1
            except Exception as e:
                print(e)

            stats_dict = get_precision_and_recall(actual_labels=subsetted_labels.values,
                                                  test_labels_correct=score_dict['test_score'])
            combined_dict = {**model_dict, **stats_dict}
            rows.append(combined_dict)
            print(f"Completed prefix: {prefix}, model: {model_func.__name__}, label:{categorical_label}, num_samples: {num_samples}, num_features: {len(subsetted_features.columns)}, test_metric: {test_metric}")     
    return rows


# +
flattened_rows = []
PARALLELIZE = True

if (PARALLELIZE and NUM_CPU_PARALLEL > 1):
    with Pool(NUM_CPU_PARALLEL-1) as p:
        rows = list(tqdm(p.imap(process_prefix, feature_combos), total=len(feature_combos)))
    flattened_rows = [item for sublist in rows for item in sublist]
else:
    for prefix_item in feature_combos:
        flattened_rows.append(process_prefix(prefix_item))

binary_categorical_df = pd.DataFrame(flattened_rows)
# -

binary_categorical_df.to_csv("./molecular_twin_pilot/outputs/multiomic/multiomic_results_early_stage_patients_with_top_features.csv")


# Read in the above saved file if computed in the cloud.
binary_categorical_df = pd.read_csv("./molecular_twin_pilot/outputs/multiomic/multiomic_results_early_stage_patients_with_top_features.csv")
binary_categorical_df[binary_categorical_df.test_loo_metric > 0.8]
binary_categorical_df.rename(columns={"test_loo_metric": "accuracy",}, inplace=True)

binary_categorical_df_no_clinical = binary_categorical_df[~binary_categorical_df['feature_prefix'].str.contains('clinical_')]
binary_categorical_df_no_clinical = binary_categorical_df_no_clinical[binary_categorical_df_no_clinical['top_15_feature_1'].notna()]
binary_categorical_df_no_clinical

binary_categorical_df_no_clinical_multiomic = binary_categorical_df_no_clinical[binary_categorical_df_no_clinical['feature_prefix'] == "('plasma_protein_', 'tissue_protein_', 'plasma_lipid_', 'pathology_NF', 'CNV_', 'freebayes_SNV_', 'pindel_INDEL_', 'AF4_', 'rna_expr_')"]
binary_categorical_df_no_clinical_multiomic

binary_categorical_df_no_clinical_multiomic.loc[binary_categorical_df_no_clinical_multiomic.groupby(["target_label"])["PPV"].idxmax()]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)
binary_categorical_df_no_clinical.loc[12775].head(600)

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Show top performing multi-omic models
binary_categorical_df.loc[binary_categorical_df.groupby(["target_label"])["PPV"].idxmax(), ["feature_prefix", "num_samples", "accuracy", "target_label", "num_input_features", "model_type","FP", "TN", "TP", "FN", "precision", "Sens", "Spec", "F1", "PPV", "NPV"]]


binary_categorical_df.loc[binary_categorical_df.groupby(["target_label"])["PPV"].idxmax(), ["feature_prefix", "num_samples", "accuracy", "target_label", "num_input_features", "model_type","FP", "TN", "TP", "FN", "precision", "Sens", "Spec", "F1", "PPV", "NPV"]]


# +
# Show top performing single-omic models
binary_categorical_df['feature_prefix'] = binary_categorical_df['feature_prefix'].astype(str)
baseline_models = binary_categorical_df[binary_categorical_df['feature_prefix'].str.split(',').apply(len) == 2]
baseline_models['first_feature'] = baseline_models.feature_prefix.str.split(',').str[0].str[2:-1]
baseline_models['second_feature'] = baseline_models.feature_prefix.str.split(',').str[1].str[1:-2]

single_omics_baseline = baseline_models[baseline_models.second_feature == ""]
single_omics_baseline.drop(columns=['first_feature', 'second_feature'], inplace=True)
single_omics_baseline = single_omics_baseline.loc[:,~single_omics_baseline.columns.str.contains('top_10_feature')].sort_values(["target_label", "accuracy"], ascending=False)
# -

single_omics_baseline.dtypes

# Find top single-omic models with features.
single_omics_baseline_with_features = single_omics_baseline[single_omics_baseline['top_15_feature_1'].notna()]
top_singleomic_baseline = single_omics_baseline_with_features.loc[single_omics_baseline_with_features.groupby(["feature_prefix", "target_label"])["PPV"].idxmax()]
top_singleomic_baseline

# For each single-omic baseline, get all features, then transpose
top_singleomic_baseline["analyte"] = top_singleomic_baseline['feature_prefix'].str.strip("\(,'\)_")
top_singleomic_baseline["file_name"] = "top_features_" + top_singleomic_baseline['analyte'] + "_" + top_singleomic_baseline['target_label'] + ".csv"
top_singleomic_baseline.dropna(axis=1, how='all', inplace=True)
top_singleomic_baseline

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', -1)

# Show top performing tissue protein models as example
tissue_protein_deceased_singleomic_baseline = single_omics_baseline[(single_omics_baseline.feature_prefix == "('tissue_protein_',)") & (single_omics_baseline.target_label == "label_deceased")].sort_values(["target_label", "accuracy"], ascending=False)
tissue_protein_deceased_singleomic_baseline

