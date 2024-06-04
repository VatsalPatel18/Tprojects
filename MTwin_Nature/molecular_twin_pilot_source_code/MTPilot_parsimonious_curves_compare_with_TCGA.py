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

NUM_CPU_PARALLEL = multiprocessing.cpu_count()
MTP_OR_TCGA = 'TCGA' # 'TCGA' or 'MTPilot' to specify which parsimonious curve to compute

pd.options.display.max_columns = 500

display(HTML("<style>.container { width:95% !important; }</style>"))
# +
clinical_prefixes = ["clinical_",]
protein_prefixes = ["plasma_protein_", "tissue_protein_", "plasma_lipid_",]
pathology_prefixes = ["pathology_NF",]
genomics_prefixes = ["CNV_", "freebayes_SNV_", "pindel_INDEL_",]
transcriptomic_prefixes = ["AF4_", "rna_expr_",]
tissue_clinical_prefixes = clinical_prefixes + genomics_prefixes + transcriptomic_prefixes + pathology_prefixes + ['tissue_protein_']
plasma_prefixes = ['plasma_protein_', 'plasma_lipid_']
plasma_clinical_prefixes = clinical_prefixes + ['plasma_protein_', 'plasma_lipid_']
clinical_path_prefixes = clinical_prefixes + pathology_prefixes
clinical_path_dna_prefixes = clinical_prefixes + pathology_prefixes + genomics_prefixes
clinical_path_rna_prefixes = clinical_prefixes + pathology_prefixes + transcriptomic_prefixes
tcga_clinical_dna_rna_prefixes = clinical_prefixes + genomics_prefixes + ['rna_expr_']
clinical_path_plasma_prefixes = clinical_prefixes + pathology_prefixes + ['plasma_protein_', 'plasma_lipid_']
omics_feature_prefixes = protein_prefixes + pathology_prefixes + genomics_prefixes + transcriptomic_prefixes + clinical_prefixes

numerical_labels = ["label_days_to_death", "label_days_to_recurrence", ]
categorical_labels = ["label_deceased", "label_recurred",] 


# +
tcgaDataset = pd.read_csv("./molecular_twin_pilot/outputs/multiomic/tcga_harmonized_validation_dataset.csv")
mtpilotDataset = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/mtpilot_tcga_harmonized_dataset.csv")

# Subset to features available in both TCGA and MTPilot datasets
common_features = set(np.intersect1d(tcgaDataset.columns, mtpilotDataset.columns)) 
tcgaDataset_subsetted = tcgaDataset[common_features]
mtpilotDataset_subsetted = mtpilotDataset[common_features]

merged_df = tcgaDataset_subsetted if (MTP_OR_TCGA == 'TCGA') else mtpilotDataset_subsetted

tcgaDataset_subsetted.shape
mtpilotDataset_subsetted.shape


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


# +
# Generate RFE learning curve and selected features per-analyte, per-scoring_metric
# For multi-omic run, generate per omics source weighted contribution

# Compute learning curve using recursive feature elimination approach, starting from full multi-omic dataset down to top 5 features.
from numpy import nanmean, mean, nanstd, std
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from matplotlib import pyplot

analytes_list = [tcga_clinical_dna_rna_prefixes] # [clinical_path_plasma_prefixes] #+ [omics_feature_prefixes] + [tissue_clinical_prefixes] + [plasma_prefixes] + [plasma_clinical_prefixes] + [clinical_path_prefixes] + [clinical_path_dna_prefixes] + [clinical_path_rna_prefixes] + omics_feature_prefixes
scoring_metrics = ['balanced_accuracy'] #, 'precision', 'average_precision', 'accuracy', ]

# get the dataset
def get_dataset(dataset_df, label, analyte_prefixes):
    X = subset_df_by_prefixes(dataset_df, analyte_prefixes)
    X = X.loc[:, ~X.columns.str.startswith('label_')]
    null_idx = ~X.isnull().any(axis=1)
    X_pruned = X[null_idx]
    y = dataset_df[label]
    y_pruned = y[null_idx]
    return X_pruned, y_pruned
 
# Get a list of models to evaluate
def get_models(max_features):
    models = dict()
    for i in range(0, 25):
        feature_num_limit = round(max_features * np.exp(-i * (0.03 * np.log(max_features))))
        rfe = RFE(LogisticRegression(max_iter=1000), step=0.1, n_features_to_select=feature_num_limit)
        model = RandomForestClassifier()
        models[str(feature_num_limit)] = make_pipeline(
            StandardScaler(),
            rfe,
            model
        )
    return models
 
# Evaluate a give model using cross-validation
def cross_validate_func(model, X, y, scoring_metric):
    cv = LeaveOneOut()

    scores_dict = cross_validate(
        estimator=model,
        X=X,
        y=y,
        cv=cv,
        scoring=scoring_metric,
        # https://scikit-learn.org/stable/modules/model_evaluation.html#scoring-parameter
        return_train_score=True,
        return_estimator=True,
    )
    return scores_dict

def get_feature_weights(feature_columns_index, scores_dict):
    feature_to_weight_dict = {}
    for feature_name in list(feature_columns_index):
        feature_to_weight_dict[feature_name] = []

    for estimator in scores_dict['estimator']:
        for feature_name, score in zip(feature_columns_index[estimator[-2].support_], estimator[-1].feature_importances_):  # For DecisionTreeClassifier and RandomForestClassifier

            feature_to_weight_dict[feature_name].append(abs(score))
    summed_feature_to_weight_dict = {feature: sum(feature_to_weight_dict[feature]) for feature in feature_to_weight_dict.keys()}
    return summed_feature_to_weight_dict
    
def get_analyte_contributions(summed_feature_to_weight_dict):
    clinical_weight = snv_weight = cnv_weight = indel_weight = plasma_protein_weight = tissue_protein_weight = plasma_lipid_weight = pathology_weight = fusion_weight = rna_expr_weight = 0

    for key, value in summed_feature_to_weight_dict.items():
        if "clinical_" in key:
            clinical_weight += value
        elif "freebayes_SNV_" in key:
            snv_weight += value
        elif "CNV_" in key:
            cnv_weight += value
        elif "pindel_INDEL_" in key:
            indel_weight += value
        elif "plasma_protein_" in key:
            plasma_protein_weight += value
        elif "tissue_protein_" in key:
            tissue_protein_weight += value
        elif "plasma_lipid_" in key:
            plasma_lipid_weight += value
        elif "pathology_" in key:
            pathology_weight += value
        elif "AF4_" in key:
            fusion_weight += value
        elif "rna_expr_" in key:
            rna_expr_weight += value
        else:
            print(f"Unknown key: {key}")
        feature_sum = clinical_weight + snv_weight + cnv_weight + indel_weight + plasma_protein_weight + tissue_protein_weight + plasma_lipid_weight + pathology_weight + fusion_weight + rna_expr_weight
    return {'Clinical': round(clinical_weight / feature_sum, 3), 'SNV': round(snv_weight / feature_sum, 3), 'CNV': round(cnv_weight / feature_sum, 3), 'INDEL': round(indel_weight / feature_sum, 3), 'Plasma Protein': round(plasma_protein_weight / feature_sum, 3), 'Tissue Protein': round(tissue_protein_weight / feature_sum, 3), 'Plasma Lipid': round(plasma_lipid_weight / feature_sum, 3), 'Pathology': round(pathology_weight / feature_sum, 3), 'Fusion': round(fusion_weight / feature_sum, 3), 'RNA Expr': round(rna_expr_weight / feature_sum, 3)}


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
    try:
        balanced_accuracy = (recall + Spec)/2
    except:
        balanced_accuracy = math.nan
    return {
        "FP": FP,
        "TN": TN,
        "TP": TP,
        "FN": FN,
        "Acc": accuracy,
        "BalAcc": balanced_accuracy,
        "Precision": precision,
        "Sens": recall,
        "Spec": Spec,
        "F1": F1,
        "PPV": PPV,
        "NPV": NPV
    }

# Runs outer most eval loop with (iteration, scoring_metric, analyte) tuple as argument
def run_outer_eval(argument_tuple):
    iteration_df = pd.DataFrame(columns=['Analyte', 'Scoring Metric', 'Iteration', 'Max Features', 'Score', 'Score StdDev', 'Clinical', 'SNV', 'CNV', 'INDEL', 'Fusion', 'RNA Expr', 'Plasma Protein', 'Tissue Protein', 'Plasma Lipid', 'Pathology', 'Feature Weights'])
    label = 'label_deceased' 
    iteration = argument_tuple[0]
    scoring_metric = argument_tuple[1]
    analyte = argument_tuple[2]
    # define dataset
    if len(analyte) > 1:
        X, y = get_dataset(merged_df, label, analyte)
    else:
        X, y = get_dataset(merged_df, label, [analyte])
    # get the models to evaluate
    models = get_models(len(X.columns))
    # evaluate the models and store results
    for name, model in models.items():
        scores_dict = cross_validate_func(model, X, y, scoring_metric)
        stats_dict = get_precision_and_recall(actual_labels=y, test_labels_correct=scores_dict['test_score'])
        feature_weights_dict = get_feature_weights(X.columns, scores_dict)
        analyte_contribution_dict = get_analyte_contributions(feature_weights_dict)
        print(f"analyte: {analyte}, iteration: {iteration}, scoring_metric: {scoring_metric}, max features: {name} score mean:{nanmean(scores_dict['test_score']):.2f} std:{nanstd(scores_dict['test_score']):.2f} stats: {stats_dict} analyte contributions: {analyte_contribution_dict}")
        iteration_df = iteration_df.append({'Analyte': analyte, 'Scoring Metric': scoring_metric, 'Iteration': iteration, 'Max Features': name, 'Score': nanmean(scores_dict['test_score']), 'Score StdDev': nanstd(scores_dict['test_score']), 'TP': stats_dict['TP'], 'FP': stats_dict['FP'], 'TN': stats_dict['TN'], 'FN': stats_dict['FN'], 'Acc': stats_dict['Acc'], 'BalAcc': stats_dict['BalAcc'], 'Precision': stats_dict['Precision'], 'Sens': stats_dict['Sens'], 'Spec': stats_dict['Spec'], 'F1': stats_dict['F1'], 'PPV': stats_dict['PPV'], 'NPV': stats_dict['NPV'], 'Clinical': analyte_contribution_dict['Clinical'], 'SNV': analyte_contribution_dict['SNV'], 'CNV': analyte_contribution_dict['CNV'], 'INDEL': analyte_contribution_dict['INDEL'], 'Fusion': analyte_contribution_dict['Fusion'], 'RNA Expr': analyte_contribution_dict['RNA Expr'], 'Pathology': analyte_contribution_dict['Pathology'], 'Plasma Protein': analyte_contribution_dict['Plasma Protein'], 'Tissue Protein': analyte_contribution_dict['Tissue Protein'], 'Plasma Lipid': analyte_contribution_dict['Plasma Lipid'], 'Feature Weights': feature_weights_dict}, ignore_index=True)
    return iteration_df


# -

iterations_list = []
for analyte in analytes_list:
    for scoring_metric in scoring_metrics:
        for i in range(1, 4):
            iterations_list.append((i, scoring_metric, analyte))

iterations_list

# +
flattened_rows = []
PARALLELIZE = False

if (PARALLELIZE and NUM_CPU_PARALLEL > 1):
    with Pool(NUM_CPU_PARALLEL-1) as p:
        rows = list(tqdm(p.imap(run_outer_eval, iterations_list), total=len(iterations_list)))
    flattened_rows = [item for sublist in rows for item in sublist]
else:
    for iteration in iterations_list:
        flattened_rows.append(run_outer_eval(iteration))

        
learning_curves_df = pd.DataFrame(columns=['Analyte', 'Scoring Metric', 'Iteration', 'Max Features', 'Score', 'Score StdDev', 'TP', 'FP', 'TN', 'FN', 'Acc', 'BalAcc', 'Precision', 'Sens', 'Spec', 'F1', 'PPV', 'NPV', 'Clinical', 'SNV', 'CNV', 'INDEL', 'Fusion', 'RNA Expr', 'Plasma Protein', 'Tissue Protein', 'Plasma Lipid', 'Pathology', 'Feature Weights'])
for row in flattened_rows:
    per_run_df = pd.DataFrame(row)
    learning_curves_df = learning_curves_df.append(per_run_df, ignore_index=True)
learning_curves_df.to_csv(f"./molecular_twin_pilot/outputs/multiomic/combinations_of_learning_curves_dataframe_validation_{MTP_OR_TCGA}.csv", index=False)
# -

results_df = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/combinations_of_learning_curves_dataframe_validation_{MTP_OR_TCGA}.csv")

# +
import ast
import numpy as np
import matplotlib.patches as mpatches

title_naming_df = {'clinical_': 'Clinical', 'freebayes_SNV_': 'SNV', 'CNV_': 'CNV', 'pindel_INDEL_': 'INDEL', 'pathology_NF': 'Pathology', 'AF4_': 'Fusion', 'rna_expr_': 'RNA Expr', 'plasma_protein_': 'Plasma Protein', 'tissue_protein_': 'Tissue Protein', 'plasma_lipid_': 'Plasma Lipid'}

def parsimonious_graph_title(analyte, separator):
    translated_analytes = [title_naming_df.get(elem, elem) for elem in ast.literal_eval(analyte)]
    joined_analytes = separator.join(translated_analytes)
    return joined_analytes


def plot_multiomic_learning_curve(key, group_df):
    analyte = key[0]
    scoring_metric = key[1]
    group_df['Max Features_int'] = group_df['Max Features'].astype(int)
    group_df.sort_values(["Max Features_int",], ascending=True, inplace=True)
    xaxis = np.log(group_df['Max Features'])
    ax = pyplot.gca()
    ax.margins(x=0.0, y=0.0)
    ax.set(ylim=(0.3, 0.9))  # Do not optimize Y-axis zoom when generating plots for general
    ax.set_facecolor("none")
    
    # Scatter plot of analyte performance at each feature count interval.
    pyplot.rcParams["figure.figsize"]=(32, 16)
    ax.scatter(xaxis, group_df['Acc'], s=65, alpha=1.0)
    pyplot.title(f'{MTP_OR_TCGA} Parsimonious Model Learning Curve for {parsimonious_graph_title(analyte, ", ")}', fontsize=28)
    pyplot.ylabel(f"Accuracy and PPV Score", fontsize=28)
    pyplot.xlabel('Max Features', fontsize=28)
    pyplot.xticks(ticks=xaxis, labels=group_df['Max Features'], fontsize=22)
    pyplot.yticks(fontsize=22)
    pyplot.grid()
    
    # Scatter plot of analyte performance at each feature count interval.
    pyplot.rcParams["figure.figsize"]=(32, 16)
    ax.scatter(xaxis, group_df['PPV'], s=65, alpha=1.0)
    pyplot.ylabel(f"Accuracy and PPV Score", fontsize=28)
    pyplot.xlabel('Max Features', fontsize=28)
    pyplot.xticks(ticks=xaxis, labels=group_df['Max Features'], fontsize=22)
    pyplot.grid()
    
    # Line fit/regression for the analyte performance metric
    ax.plot(np.unique(xaxis), np.poly1d(np.polyfit(xaxis, group_df['Acc'], 8))(np.unique(xaxis)), color = 'blue', linewidth=6.0, alpha=1.0)
    ax.plot(np.unique(xaxis), np.poly1d(np.polyfit(xaxis, group_df['PPV'], 8))(np.unique(xaxis)), color = 'red', linewidth=6.0, alpha=1.0)
    
    blue_patch = mpatches.Patch(color='blue', label='ACC')
    red_patch = mpatches.Patch(color='red', label='PPV')
    pyplot.legend(handles=[blue_patch, red_patch], loc='lower right', bbox_to_anchor=(0.85, 0.0), title='Metrics', fontsize=22, title_fontsize=22)
    
    # Adding Twin Axes to plot for overlayed Accuracy and PPV data points and best-fit lines.
    ax2 = ax.twinx() 
    ax2.margins(x=0.0, y=0.0)
    
    # Plotting Multiomic analyte composition on the first axis as plot background
    ax2.set_ylabel('Analyte % Contribution', fontsize=28) 

    # Restrict legend only to analytes present in the current iteration.
    label_names = ['Plasma Protein', 'Plasma Lipid', 'Tissue Protein', 'RNA Expr', 'Fusion', 'SNV', 'CNV', 'INDEL', 'Pathology', 'Clinical']
    translated_analytes = [title_naming_df.get(elem, elem) for elem in ast.literal_eval(analyte)]
    filtered_label_names = [x if x in translated_analytes else '_'+x+'_' for x in label_names]

    ax2.stackplot(xaxis, group_df['Plasma Protein'].tolist(), group_df['Plasma Lipid'].tolist(), group_df['Tissue Protein'].tolist(), group_df['RNA Expr'].tolist(), group_df['Fusion'].tolist(), group_df['SNV'].tolist(), group_df['CNV'].tolist(), group_df['INDEL'].tolist(), group_df['Pathology'].tolist(), group_df['Clinical'].tolist(), labels=filtered_label_names, baseline='zero', alpha=0.55)
    pyplot.legend(title='Analytes', loc='lower right', fontsize=22)
    ax2.get_legend().get_title().set_fontsize('22')
    
    # Invert the order of axes such that analyte composition in the background and Acc/PPV are in the forefront
    ax2.set_zorder(ax.get_zorder()-1)
    ax2.patch.set_visible(False)
    
    pyplot.yticks(fontsize=22)
    pyplot.figure(figsize=(32, 16), dpi=400)
    pyplot.savefig(f'./molecular_twin_pilot/MTPilot_learning_curve_for_survival_{parsimonious_graph_title(analyte, "_")}_{MTP_OR_TCGA}.png', dpi=600)
    pyplot.show()


def plot_learning_curve(key, group_df):
    analyte = key[0]
    scoring_metric = key[1]
    group_df['Max Features_int'] = group_df['Max Features'].astype(int)
    group_df.sort_values(["Max Features_int",], ascending=True, inplace=True)
    xaxis = np.log(group_df['Max Features'])
    ax = pyplot.gca()
    ax.margins(x=0.0, y=0.0)
    ax.tick_params(axis ='y', color='blue')
    ax.set(ylim=(0.3, 0.9))
    
    # Scatter plot of analyte performance at each feature count interval.
    pyplot.rcParams["figure.figsize"]=(32, 16)
    ax.scatter(xaxis, group_df['Score'], s=65, alpha=1.0)
    pyplot.title(f'Parsimonious Model Learning Curve for {MTP_OR_TCGA} {parsimonious_graph_title(analyte, ", ")}', fontsize=28)
    pyplot.ylabel(f"{'Precision' if scoring_metric == 'precision' else 'Accuracy'}", fontsize=28)
    pyplot.xlabel('Max Features', fontsize=28)
    pyplot.xticks(ticks=xaxis, labels=group_df['Max Features'], fontsize=22)
    pyplot.yticks(fontsize=22)
    pyplot.grid()
    
    # Line fit/regression for the analyte performance
    ax.plot(np.unique(xaxis), np.poly1d(np.polyfit(xaxis, group_df['Score'], 7))(np.unique(xaxis)), color = 'blue', linewidth=6.0, alpha=1.0)
    
    pyplot.yticks(fontsize=22)
    pyplot.savefig(f'./molecular_twin_pilot/MTPilot_learning_curve_for_survival_{parsimonious_graph_title(analyte, "_")}.png', dpi=600)
    pyplot.show()


# -

# Generates learning curve plots per-metrics, per-analyte.
mean_df = results_df.groupby(['Analyte', 'Scoring Metric', 'Max Features']).agg({'Score':'max', 'Score StdDev':'mean', 'FP':'mean', 'TN':'mean', 'TP':'mean', 'FN':'mean', 'Acc':'mean', 'BalAcc':'mean', 'Precision':'mean', 'Sens':'mean', 'Spec':'mean', 'F1':'mean', 'PPV':'mean', 'NPV':'mean', 'Clinical':'mean', 'SNV':'mean', 'CNV':'mean', 'INDEL':'mean', 'Fusion':'mean', 'RNA Expr':'mean', 'Plasma Protein':'mean', 'Tissue Protein':'mean', 'Plasma Lipid':'mean', 'Pathology':'mean'}).reset_index()
mean_df = mean_df.groupby(['Analyte', 'Scoring Metric'])
for key, group in mean_df:
    analyte = key[0]
    if len(analyte) > 1:
        plot_multiomic_learning_curve(key, group)
    else:
        plot_learning_curve(key, group)


results_df[results_df['Max Features'] == 65]

mtpilot_results_df = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/combinations_of_learning_curves_dataframe_validation_MTPilot.csv")

mtpilot_results_df

pd.options.display.max_rows = 40000
pd.options.display.max_colwidth = 40
mtpilot_results_df[(mtpilot_results_df['Max Features'] == 215) & (mtpilot_results_df['Iteration'] == 1)]

# +
pd.options.display.max_rows = 40000
pd.options.display.max_colwidth = 400000000

# Top parsimonious model for MTPilot and their respective top features.
parsimonious_mtpilot_json = mtpilot_results_df[(mtpilot_results_df['Max Features'] == 215) & (mtpilot_results_df['Iteration'] == 1)]['Feature Weights']
parsimonious_mtpilot_json
# -

tcga_results_df = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/combinations_of_learning_curves_dataframe_validation_TCGA.csv")

tcga_results_df[(tcga_results_df['Max Features'] == 65) & (tcga_results_df['Iteration'] == 2)]

# +
pd.options.display.max_rows = 40000
pd.options.display.max_colwidth = 400000000

# Top parsimonious model for TCGA and their respective top features.
parsimonious_tcga_json = tcga_results_df[(tcga_results_df['Max Features'] == 65) & (tcga_results_df['Iteration'] == 2)]['Feature Weights']
parsimonious_tcga_json
# -

