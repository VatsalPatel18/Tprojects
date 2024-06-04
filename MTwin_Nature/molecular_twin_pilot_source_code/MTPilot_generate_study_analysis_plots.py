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

# +
# Copyright (C) 2022 - Betteromics Inc.
import pandas as pd
import tensorflow_data_validation as tfdv
import numpy as np
import csv

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_colwidth', 0)

protein_prefixes = ["plasma_protein_", "tissue_protein_", "plasma_lipid_",]
pathology_prefixes = ["pathology_NF",]
genomics_prefixes = ["CNV_", "freebayes_SNV_", "pindel_INDEL_",]
transcriptomic_prefixes = ["AF4_", "rna_expr_",]
labels = ["label_"]

analytes = ["Clinical", "Plasma_Protein", "Tissue_Protein", "Plasma_Lipid", "Pathology", "CNV", "SNV", "INDEL", "RNA_Fusion", "RNA_Expr", "Multiomic"]
analytes_without_multiomic = ["Clinical", "Tissue_Protein", "Plasma_Protein", "Plasma_Lipid", "Pathology", "CNV", "SNV", "INDEL", "RNA_Fusion", "RNA_Expr"]
study_labels = ["label_deceased", "label_recurred"]
feature_col_prefixes = ["Clinical_", "Plasma_Protein_", "Tissue_Protein_", "Plasma_Lipid_", "Pathology_", "CNV_", "SNV_", "INDEL_", "RNA_Fusion_", "RNA_Expr_"]

analyte_to_prefix = {
    "Clinical": "Clinical_",
    "Plasma_Protein": "Plasma_Protein_",
    "Tissue_Protein": "Tissue_Protein_",
    "Plasma_Lipid": "Plasma_Lipid_",
    "Pathology": "Pathology_",
    "CNV": "CNV_",
    "SNV": "SNV_",
    "INDEL": "INDEL_",
    "RNA_Fusion": "RNA_Fusion_",
    "RNA_Expr": "RNA_Expr_",
    "Multiomic": ""
}

omics_feature_prefixes = protein_prefixes + pathology_prefixes + genomics_prefixes + transcriptomic_prefixes

STAT_COLUMNS = ["Analytes", "# Samples", "# Analytes", "Accuracy", "Study Label", "# Input Feat.", "Classifier Type","FP", "TN", "TP", "FN", "Precision", "Sens", "Spec", "F1", "PPV", "NPV"]
ALL_MODELS_FULL_SAMPLES = "./molecular_twin_pilot/outputs/multiomic/multiomic_results_early_stage_patients.csv"

# +
# Read in the above saved file if computed in the cloud.
binary_categorical_df = pd.read_csv(ALL_MODELS_FULL_SAMPLES)
binary_categorical_df.rename(columns={"test_loo_metric": "Accuracy", "feature_prefix": "Analytes", "target_label": "Study Label", "model_type": "Classifier Type", "num_samples": "# Samples", "num_input_features": "# Input Feat.", "precision": "Precision"}, inplace=True)
binary_categorical_df.drop(columns=["Unnamed: 0"], inplace=True)
binary_categorical_df.replace(["'clinical_'", "'plasma_protein_'", "'tissue_protein_'", "'plasma_lipid_'", "'pathology_NF'", "'CNV_'", "'freebayes_SNV_'", "'pindel_INDEL_'", "'AF4_'", "'rna_expr_'"],
                              ["Clinical", "Plasma_Protein", "Tissue_Protein", "Plasma_Lipid", "Pathology", "CNV", "SNV", "INDEL", "RNA_Fusion", "RNA_Expr"], regex=True, inplace=True)
binary_categorical_df["Analytes"] = binary_categorical_df["Analytes"].str.replace("(", "")
binary_categorical_df["Analytes"] = binary_categorical_df["Analytes"].str.replace(",\)", "")
binary_categorical_df["Analytes"] = binary_categorical_df["Analytes"].str.replace("\)", "")
binary_categorical_df["# Analytes"] = binary_categorical_df['Analytes'].str.split(',').apply(len)

# Cleanup top_15_feature_# columns to be gene,feat_freq 
binary_categorical_topFeatures = binary_categorical_df.loc[:,binary_categorical_df.columns.str.contains('top_15_feature')]
binary_categorical_topFeatures = binary_categorical_topFeatures.replace(",.*", '', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace(" feat_freq:", ',', regex=True)
#binary_categorical_topFeatures = binary_categorical_topFeatures.replace("rna_expr_|plasma_lipid_|pathology_NF|CNV_|freebayes_SNV_|pindel_INDEL_|AF4_|clinical_", '', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("clinical_", 'Clinical_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("rna_expr_", 'RNA_Expr_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("pathology_NF", 'Pathology_NF', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("AF4_", 'RNA_Fusion_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("freebayes_SNV_", 'SNV_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("pindel_INDEL_", 'INDEL_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("tissue_protein_.*_", 'Tissue_Protein_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("plasma_protein_.*_", 'Plasma_Protein_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("plasma_lipid_", 'Plasma_Lipid_', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace("\):", '),', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace(":0\.", ',0.', regex=True)
binary_categorical_topFeatures = binary_categorical_topFeatures.replace(":1\.", ',1.', regex=True)

binary_categorical_stats = binary_categorical_df.loc[:,~binary_categorical_df.columns.str.contains('top_15_feature')]
binary_categorical_df = pd.concat([binary_categorical_stats, binary_categorical_topFeatures], axis='columns')
binary_categorical_df.sort_values(["# Analytes"], ascending=True, inplace=True)
binary_categorical_df.sort_values(["Accuracy", "Precision", "PPV", "Sens", "Spec"], ascending=False, inplace=True)
binary_categorical_df = binary_categorical_df.round(3)


# +
# Generates summary stats table for top performing baseline models per omics source, per label 
baseline_models = binary_categorical_stats.loc[(binary_categorical_stats['# Analytes'] == 1) & (binary_categorical_stats['Analytes'] != "")]
for label in study_labels:
    per_label_baselines = baseline_models.loc[(baseline_models['Study Label'] == label)]
    per_label_baselines.sort_values(["Accuracy", "PPV", "Sens", "Spec"], ascending=False, inplace=True)
    per_label_baselines_stats = per_label_baselines.loc[per_label_baselines.groupby(["Analytes"])["Accuracy"].idxmax(), STAT_COLUMNS]
    per_label_baselines_stats.drop(columns=["Study Label"], inplace=True)
    per_label_baselines_stats.to_csv(f"./molecular_twin_pilot/MTPilot_top_baseline_singleomic_models_{label}.csv", float_format='%.2f', index=False)

# Generate summary stats table for top15 performing multi-omic models per label
for label in study_labels:
    per_label_multiomic = binary_categorical_stats.loc[(binary_categorical_stats['Study Label'] == label)]
    per_label_multiomic.sort_values(["# Analytes"], ascending=True, inplace=True)
    per_label_multiomic.sort_values(["Accuracy", "PPV", "Sens", "Spec"], ascending=False, inplace=True)
    per_label_multiomic.drop(columns=["Study Label", "# Analytes"], inplace=True)
    per_label_multiomic[0:15].to_csv(f"./molecular_twin_pilot/MTPilot_top_multiomic_models_{label}.csv", float_format='%.2f', index=False)

# +
# Generates table of topN single-omics & multi-omic model features
topN_features = pd.DataFrame(columns=['Model', 'Study Label', 'TopN_Features'])

for label in study_labels:
    study_label_df = binary_categorical_df.loc[binary_categorical_df["Study Label"] == label]
    for analyte in analytes:
        if analyte != "Multiomic":
            # Find and save top2 performing single-omic models for each analyte with features available.
            singleomic_models = study_label_df.loc[(study_label_df['# Analytes'] == 1) & (study_label_df['Analytes'] == analyte) & (study_label_df['top_15_feature_1'].notnull())]
            singleomic_models.sort_values(["Accuracy", "PPV"], ascending=False, inplace=True)
            raw_features = singleomic_models[singleomic_models.columns[singleomic_models.columns.str.contains('top_15_feature')]]
            raw_features_list = raw_features[:2].replace(",.*", '', regex=True).values.flatten().tolist()[:20]
            cleaned_features_set = set([x for x in raw_features_list if str(x) != 'nan'])
            cleaned_features_flattened = ",".join(cleaned_features_set)
            topN_features = topN_features.append({'Model': analyte, 'Study Label': label, 'TopN_Features': cleaned_features_flattened}, ignore_index=True)
        elif analyte == "Multiomic":
            # Find and save top10 performing multiomic models with features available.
            multiomic_models = study_label_df.loc[(study_label_df['top_15_feature_1'].notnull())]
            multiomic_models.sort_values(["# Analytes"], ascending=True, inplace=True)
            multiomic_models.sort_values(["Accuracy", "PPV"], ascending=False, inplace=True)
            raw_features = multiomic_models[multiomic_models.columns[multiomic_models.columns.str.contains('top_15_feature')]]
            raw_features_list = raw_features[0:9].replace(",.*", '', regex=True).values.flatten().tolist()
            cleaned_features_set = set([x for x in raw_features_list if str(x) != 'nan'])
            cleaned_features_flattened = ",".join(cleaned_features_set)
            topN_features = topN_features.append({'Model': 'Multiomic', 'Study Label': label, 'TopN_Features': cleaned_features_flattened}, ignore_index=True)

topN_features.to_csv("./molecular_twin_pilot/MTPilot_top_features_with_clinical.csv", float_format='%.2f', index=False)

# +
# Generate topN features and their frequencies for top single-omic (per-analyte) models
topN_features_freq = pd.DataFrame(columns=['Analyte', 'Study Label', 'Feature', 'Frequency'])

for label in study_labels:
    study_label_df = binary_categorical_df.loc[(binary_categorical_df["Study Label"] == label) & (binary_categorical_stats['Analytes'] != "")]
    for analyte in analytes:
        if analyte != "Multiomic":
            # Find and save top single-omic features for each analyte with features frequencies.
            singleomic_models = study_label_df.loc[(study_label_df['# Analytes'] == 1) & (study_label_df['Analytes'] == analyte) & (study_label_df['top_15_feature_1'].notnull())]
            singleomic_models.sort_values(["Accuracy", "PPV"], ascending=False, inplace=True)
            singleomic_models = singleomic_models.replace("RNA_Expr_|Plasma_Lipid_|Pathology_|CNV_|SNV_|INDEL_|RNA_Fusion_|Clinical_", '', regex=True)
            singleomic_models = singleomic_models.replace("Plasma_Protein_|Tissue_Protein_", '', regex=True)
            raw_features = singleomic_models[singleomic_models.columns[singleomic_models.columns.str.contains('top_15_feature')]]
            top_feat_flat = raw_features.iloc[0]
            top_feat_expanded = top_feat_flat.str.rsplit(",", n=1, expand=True)
            raw_features_expanded = pd.DataFrame()
            raw_features_expanded["Feature"] = top_feat_expanded[0]
            raw_features_expanded["Frequency"] = top_feat_expanded[1]
            raw_features_expanded["Analyte"] = analyte
            raw_features_expanded["Study Label"] = label
            raw_features_expanded = raw_features_expanded[raw_features_expanded.Frequency.astype(float) > 0.05]
            topN_features_freq = topN_features_freq.append(raw_features_expanded, ignore_index=True)
            
topN_features_freq.to_csv("./molecular_twin_pilot/MTPilot_top_singleomic_features_and_frequencies.csv", float_format='%.2f', index=False)

# +
import re
import matplotlib.pyplot as plt

def infer_category_from_feature(feature):
    for analyte in analytes:
        if feature.startswith(analyte):
            return analyte

# Generate analyte contribution stats from feature frequencies
def generate_analyte_contribution_stats(feature_freq_df):
    topN_features_freq = pd.DataFrame(columns=['Model_Index', 'Feature', 'Frequency'])
    for index, row in feature_freq_df.iterrows():
        top_feat_flat = feature_freq_df.loc[index]
        top_feat_expanded = top_feat_flat.str.rsplit(",", n=1, expand=True)
        top_feat_expanded.dropna(axis='index', how='all', inplace=True)
        raw_features_expanded = pd.DataFrame()
        raw_features_expanded["Feature"] = top_feat_expanded[0]
        raw_features_expanded["Frequency"] = top_feat_expanded[1]
        raw_features_expanded["Model_Index"] = index
        topN_features_freq = topN_features_freq.append(raw_features_expanded, ignore_index=True)  
    
    topN_features_freq['Frequency'] = topN_features_freq['Frequency'].apply(pd.to_numeric, errors='coerce')
    topN_features_freq = topN_features_freq[topN_features_freq['Frequency'] > 0.05]
    topN_features_freq['Feature_Category'] = topN_features_freq['Feature'].map(lambda x: infer_category_from_feature(x))
    per_model_analyte_stats = topN_features_freq.groupby(['Model_Index', 'Feature_Category'])['Frequency'].sum().reset_index()
    per_model_feature_sums = per_model_analyte_stats.groupby(['Model_Index']).sum().reset_index()
    per_model_feature_sums.rename(columns={"Frequency":"Frequency_Sum"}, inplace=True)
    analyte_contribution_stats = per_model_analyte_stats.merge(right=per_model_feature_sums, on=['Model_Index'], how='left')
    analyte_contribution_stats['Percent_Contribution'] = (analyte_contribution_stats['Frequency'] / analyte_contribution_stats['Frequency_Sum']) * 100
    analyte_contribution_stats['Contributing_Analytes'] = analyte_contribution_stats.groupby(['Model_Index'])['Feature_Category'].transform(','.join)
    return analyte_contribution_stats


# Save top multi-omic features with features frequencies.
def output_multiomic_top_features_and_frequencies(multiomic_models, label, dedupe):
    topN_features_freq = pd.DataFrame()
    multiomic_models = multiomic_models[multiomic_models['Study Label'] == label]
    multiomic_models.sort_values(["Study Label", "Accuracy", "PPV"], ascending=False, inplace=True)
    if dedupe:
        multiomic_models.drop_duplicates(['Study Label', 'Contributing_Analytes'], keep='first', inplace=True)

    multiomic_models = multiomic_models.head(15)
    raw_features = multiomic_models[multiomic_models.columns[multiomic_models.columns.str.contains('top_15_feature')]]
    count = 1
    for index, row in raw_features.iterrows():
        top_feat_flat = raw_features.loc[index]
        top_feat_expanded = top_feat_flat.str.rsplit(",", n=1, expand=True)
        raw_features_expanded = pd.DataFrame()
        raw_features_expanded["Feature"] = top_feat_expanded[0]
        raw_features_expanded["Frequency"] = top_feat_expanded[1]
        raw_features_expanded["Model"] = count
        raw_features_expanded["Study Label"] = label
        raw_features_expanded = raw_features_expanded[raw_features_expanded.Frequency.astype(float) > 0.05]
        topN_features_freq = topN_features_freq.append(raw_features_expanded, ignore_index=True)
        count += 1
    topN_features_freq.to_csv(f"./molecular_twin_pilot/MTPilot_top_multiomic_features_and_frequencies_{label}{'_deduped' if dedupe else ''}.csv", float_format='%.2f', index=False)


# +
# Generates summary statistics for top multi-omic models.
# 1) Create summary stacked box-plots of per-analyte composition for top models
# 2) Per top multiomic model, outputs topN features and frequencies

# Find topN models for objective label with feature frequencies available;
multiomic_models = binary_categorical_df.loc[(binary_categorical_df['Analytes'] != "") & (binary_categorical_df['top_15_feature_1'].notnull())]
multiomic_models.sort_values(["# Analytes"], ascending=True, inplace=True)
multiomic_models.sort_values(["Study Label", "Accuracy", "PPV"], ascending=False, inplace=True)
multiomic_models.drop_duplicates(['Study Label', 'Analytes'], keep='first', inplace=True)

# Generate per model, per analyte frequency and total model feature_freq sum
raw_features = multiomic_models[multiomic_models.columns[multiomic_models.columns.str.contains('top_15_feature')]]
analyte_contribution_stats = generate_analyte_contribution_stats(raw_features)

# Generate within each model, flattened per-analyte percent contribution stats.
per_model_per_analyte_stats = analyte_contribution_stats.pivot_table(index='Model_Index', columns='Feature_Category', values='Percent_Contribution', fill_value=0)
analyte_contribution_stats = analyte_contribution_stats.merge(right=per_model_per_analyte_stats, on=['Model_Index'])

# Merge back per-model, per-analyte contribution stats with multiomic_models master dataframe
multiomic_models.reset_index(inplace=True)
multiomic_models.rename(columns={"index":"Model_Index"}, inplace=True)
multiomic_stats = multiomic_models.merge(right=analyte_contribution_stats, on=['Model_Index'])

for dedup_identical_models in [True, False]:
    for label in study_labels:
        multiomic_results = multiomic_stats[multiomic_stats['Study Label'] == label]
        multiomic_results.sort_values(["Study Label", "Accuracy", "PPV"], ascending=False, inplace=True)
        if dedup_identical_models:
            # Dedup models with exact same analytes
            multiomic_results.drop_duplicates(['Study Label', 'Contributing_Analytes'], keep='first', inplace=True)
        
        # Subset multiomic_results to analyte contributions and topN results.
        multiomic_results.sort_values(["Accuracy", "PPV"], ascending=False, inplace=True)
        multiomic_results = multiomic_results.head(15)
        multiomic_results.sort_values(["Accuracy", "PPV"], ascending=False, inplace=True)
        multiomic_results['Model_Id'] = np.arange(1, multiomic_results.shape[0] + 1)
        
        # Output flattened feature frequency file for topN models
        output_multiomic_top_features_and_frequencies(multiomic_results, label, dedup_identical_models)
        
        multiomic_plot = multiomic_results[['Model_Id'] + analytes_without_multiomic]
        multiomic_plot.sort_values(['Model_Id'], ascending=False, inplace=True)
        ax = multiomic_plot.plot(kind='barh', x="Model_Id", figsize=(15, 10), fontsize=12, stacked=True)
        ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=14)
        plt.title(f"Top 15 Model Per-Analyte Contributions for {'Survival' if label == 'label_deceased' else 'Recurrence'}", color='black', fontsize=20)
        plt.ylabel("Top 15 Multiomic Models with Accuracy & PPV", fontsize=14)
        plt.xlabel("Percent Analyte Contribution", fontsize=14)

        # Add "ACC:0.XX, PPV:0.YY" labes on each bar.
        for rowNum, row in multiomic_results.iterrows():
            ax.text(45, 15 - (row['Model_Id']+0.14), f"ACC:{row['Accuracy']:,.2f}, PPV:{row['PPV']:,.2f}", color='black', fontsize=12)

        plt.savefig(f"./molecular_twin_pilot/MTPilot_multiomic_analyte_contribution_{label}{'_deduped' if dedup_identical_models else ''}.png", bbox_inches='tight', dpi=400)  

# +
raw_multiomic_features = pd.read_csv("./molecular_twin_pilot/outputs/multiomic/early_stage_patients_multiomic_dataset.csv")
raw_multiomic_features.columns = [c.replace('clinical_', 'Clinical_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('rna_expr_', 'RNA_Expr_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('pathology_NF', 'Pathology_NF') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('AF4_', 'RNA_Fusion_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('freebayes_SNV_', 'SNV_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('pindel_INDEL_', 'INDEL_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('tissue_protein_', 'Tissue_Protein_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('plasma_protein_', 'Plasma_Protein_') for c in raw_multiomic_features.columns]
raw_multiomic_features.columns = [c.replace('plasma_lipid_', 'Plasma_Lipid_') for c in raw_multiomic_features.columns]

numerical_labels = ["label_days_to_death", "label_days_to_recurrence", ]
categorical_labels = ["label_deceased", "label_recurred",]
metadata_columns = numerical_labels + ["Biobank_Number"]
pruned_multiomic_data = raw_multiomic_features.drop(columns=metadata_columns)

def subset_df_by_prefixes(df, prefixes):
    column_list = []
    for prefix in prefixes:
        column_list += df.columns[df.columns.str.startswith(prefix)].tolist()
    return df[column_list]

def intersect_omics_features(df, singleomics_column_prefixes, labels):
    features = subset_df_by_prefixes(df, singleomics_column_prefixes + labels)
    for prefix in singleomics_column_prefixes:
        singleomic_columns = features.columns[features.columns.str.contains(prefix)].tolist()
        features.dropna(axis=0, how='all', subset=singleomic_columns, inplace=True)
    features.dropna(axis=0, how='all', inplace=True)
    features.dropna(axis=1, how='all', inplace=True)
    return features


# +
# Generate a clustermap of input features.
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

feature_box_plots = PdfPages(f'./molecular_twin_pilot/MTPilot_clustermaps.pdf')

def generate_clustermap(analyte, label, features_df, label_sort=False):
    if label_sort:
        features_df.sort_values(label, inplace=True)
    color_dict=dict(zip(np.unique(features_df[label]),np.array(['b','r'])))
    deceased_df = pd.DataFrame({label:features_df[label]})
    deceased_colors = deceased_df[label].map(color_dict)

    metric_op = 'euclidean' # 'sqeuclidean' or 'correlation' (only if not sparse)

    colormap = sns.color_palette("vlag", as_cmap=True)
    graph = sns.clustermap(features_df, metric=metric_op, standard_scale=1, yticklabels=False, row_cluster=True, col_cluster=False, row_colors=deceased_colors, cmap=colormap) #[deceased_colors, recurred_colors])
    graph.ax_heatmap.set_title(f'MTPilot_{analyte}_Clustermap_{label}', fontsize=16, verticalalignment='top')
    graph.cax.set_position((.13,.255,.03,.15))
    plt.savefig(f'./molecular_twin_pilot/MTPilot_{analyte}_clustermap_{label}_label_sort_{label_sort}.png', dpi=410)


# -

# Generates clustermap of each omics category features for each study label
for label in study_labels:
    study_label_df = binary_categorical_df.loc[binary_categorical_df["Study Label"] == label]
    for analyte in analytes:
        if analyte == "Multiomic":
            col_prefix = feature_col_prefixes
        else:
            col_prefix = [analyte_to_prefix[analyte]]
        analyte_restricted_df = intersect_omics_features(pruned_multiomic_data, col_prefix, [label])

        analyte_restricted_df.dropna(axis=0, how='all', inplace=True)
        analyte_restricted_df.dropna(axis=1, how='all', inplace=True)
        analyte_restricted_df = analyte_restricted_df.apply(pd.to_numeric, errors='coerce', axis=1)
        analyte_restricted_df = analyte_restricted_df[analyte_restricted_df.columns[analyte_restricted_df.nunique() > 1]]

        generate_clustermap(analyte, label, analyte_restricted_df, label_sort=False)
        generate_clustermap(analyte, label, analyte_restricted_df, label_sort=True)
        print(f'analyte: {analyte} label: {label} shape: {analyte_restricted_df.shape}')

# +
# Plot heatmap of patient study labels.
from matplotlib.pyplot import figure
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.style.use('seaborn')


label_data = raw_multiomic_features[categorical_labels + numerical_labels]
label_data["label_deceased"] = label_data["label_deceased"] * label_data["label_days_to_death"].max()
label_data["label_recurred"] = label_data["label_recurred"] * label_data["label_days_to_recurrence"].max()

colormap = sns.color_palette("vlag", as_cmap=True)
plt.imshow(label_data, cmap=colormap)
plt.grid(False)
plt.savefig(f'./molecular_twin_pilot/MTPilot_label_heatmap.png', dpi=300)

# +
# Generate 3D grid-search plot for each label (omics-set vs classifier-type)
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

for label in study_labels:
    # Make data.
    study_label_df = binary_categorical_df.loc[binary_categorical_df["Study Label"] == label]
    x = study_label_df["Analytes"].astype('category').cat.codes
    y = study_label_df["Classifier Type"].astype('category').cat.codes
    z = study_label_df["Accuracy"]
    
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    colormap = sns.color_palette("vlag", as_cmap=True)

    surf = ax.plot_trisurf(x, y, z, cmap=colormap,
                           linewidth=0, antialiased=True, vmin=0.4, vmax=1.0, shade=False)
    
    # Customize the z axis.
    ax.set_zlim(0.4, 1.0)
    ax.zaxis.set_major_locator(LinearLocator(8))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')
    z_label = "Accuracy for Deceased label"
    if label == "label_recurred":
        z_label = "Accuracy for Recurred label"
    ax.set_zlabel(z_label)
    ax.set_xlabel("Analytes")
    ax.set_ylabel("Classifier Types")

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=10)
    fig.set_size_inches(24, 16)

    #plt.show()
    plt.savefig(f'./molecular_twin_pilot/MTPilot_grid_search_{label}.png', dpi=600)

# +
# Generate 3D search space plot for each label
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator
import numpy as np

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

# Make data.
X = np.arange(-5, 5, 0.25)
Y = np.arange(-5, 5, 0.25)
X, Y = np.meshgrid(X, Y)
R = np.sqrt(X**2 + Y**2)
Z = np.sin(R)

# Plot the surface.
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)

# Customize the z axis.
ax.set_zlim(-1.01, 1.01)
ax.zaxis.set_major_locator(LinearLocator(10))
# A StrMethodFormatter is used automatically
ax.zaxis.set_major_formatter('{x:.02f}')

# Add a color bar which maps values to colors.
fig.colorbar(surf, shrink=0.5, aspect=5)

plt.show()

# +
import umap.umap_ as umap

def generate_umap(analyte, label, feature_df, feature_type):
    trans = umap.UMAP(n_neighbors=5, random_state=12).fit(feature_df)
    plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], c=feature_df[label], s=20, cmap='Spectral')
    plt.title(f'MTPilot UMAP for {analyte} colored by {label}', fontsize=12);
    plt.savefig(f'./molecular_twin_pilot/MTPilot_{analyte}_umap_{label}_{feature_type}.png', dpi=500)
    plt.show() 
    
# Only retain patients that have multiomic data.
def subset_df_by_prefixes(df, prefixes):
    column_list = []
    for prefix in prefixes:
        column_list += df.columns[df.columns.str.startswith(prefix)].tolist()
    return df[column_list]

def intersect_omics_features(df, singleomics_column_prefixes, labels):
    features = subset_df_by_prefixes(df, singleomics_column_prefixes + labels)
    for prefix in singleomics_column_prefixes:
        singleomic_columns = features.columns[features.columns.str.contains(prefix)].tolist()
        features.dropna(axis=0, how='all', subset=singleomic_columns, inplace=True)
    features.dropna(axis=0, how='all', inplace=True)
    features.dropna(axis=1, how='all', inplace=True)
    return features


# -

# Generates UMAP cluster plots for each omics category features for outcome label (for topN_features)
for label in study_labels:
    for analyte in analytes:
        top_features_row = topN_features[(topN_features["Model"] == analyte) & (topN_features["Study Label"] == label)]
        top_features_list = top_features_row["TopN_Features"].str.split(",").tolist()[0]
        if analyte == "Multiomic":
            col_prefix = feature_col_prefixes
        else:
            col_prefix = [analyte_to_prefix[analyte]]
        analyte_restricted_df = intersect_omics_features(pruned_multiomic_data, col_prefix, [label])
        top_columns = [label]
        for top_feature in top_features_list:
            if top_feature != "":
                top_columns += analyte_restricted_df.columns[analyte_restricted_df.columns.to_series().str.contains(top_feature, regex=False)].tolist()
        top_columns_feature_df = analyte_restricted_df[top_columns]
        top_columns_feature_df.dropna(axis=0, how='all', inplace=True)
        top_columns_feature_df.dropna(axis=1, how='all', inplace=True)
        top_columns_feature_df_pruned = top_columns_feature_df.apply(pd.to_numeric, errors='coerce', axis=1)
        top_columns_feature_df_pruned = top_columns_feature_df[top_columns_feature_df.columns[top_columns_feature_df.nunique() > 1]]

        generate_umap(analyte, label, top_columns_feature_df_pruned, "topN_features")
        print(f'analyte: {analyte} label: {label} shape: {top_columns_feature_df_pruned.shape}')

# Generates UMAP cluster plots for each omics category features for outcome label (for raw features)
for label in study_labels:
    for analyte in analytes:
        if analyte == "Multiomic":
            col_prefix = feature_col_prefixes
        else:
            col_prefix = [analyte_to_prefix[analyte]]
        analyte_restricted_df = intersect_omics_features(pruned_multiomic_data, col_prefix, [label])
        generate_umap(analyte, label, analyte_restricted_df, "raw_features")
        print(f'analyte: {analyte} label: {label} shape: {analyte_restricted_df.shape}')

# +
# Function to plot asymetric violin performance plots for single-omic models
import matplotlib.pyplot as plt
import seaborn as sns
PLOT_COLUMNS = ["Analytes", "Study Label", "Accuracy", "Precision", "Sens", "Spec", "PPV", "NPV", "F1"]

def generate_singleomic_violin_plots(input_df, metric1, metric2, label, add_multiomic=False):
    # Singleomic analytes.
    singleomic_df = input_df.loc[(input_df["Study Label"] == label) & (input_df["# Analytes"] == 1) & (input_df["Analytes"] != ""), PLOT_COLUMNS]
    singleomic_df.sort_values(["Study Label", "Analytes"], ascending=False, inplace=True)
    metric1_data = singleomic_df[['Analytes', metric1]]
    metric1_data.rename(columns={metric1:'Value'}, inplace=True)
    metric1_data['Metric'] = metric1
    metric2_data = singleomic_df[['Analytes', metric2]]
    metric2_data.rename(columns={metric2:'Value'}, inplace=True)
    metric2_data['Metric'] = metric2
    plot_data = pd.concat([metric1_data, metric2_data])
    
    if add_multiomic:
        # Multiomic analytes
        multiomic_df = input_df.loc[(input_df["Study Label"] == label) & (input_df["# Analytes"] > 1) & (input_df["Analytes"] != ""), PLOT_COLUMNS]
        multiomic_df["Analytes"] = "Multiomic"
        multiomic_df.sort_values(["Study Label", "Analytes"], ascending=False, inplace=True)
        multi_metric1_data = multiomic_df[['Analytes', metric1]]
        multi_metric1_data.rename(columns={metric1:'Value'}, inplace=True)
        multi_metric1_data['Metric'] = metric1
        multi_metric2_data = multiomic_df[['Analytes', metric2]]
        multi_metric2_data.rename(columns={metric2:'Value'}, inplace=True)
        multi_metric2_data['Metric'] = metric2
        plot_data = pd.concat([plot_data, multi_metric1_data, multi_metric2_data])

    plt.figure(figsize=(32, 12))
    sns.set(font_scale = 2)
    ax = sns.violinplot(x="Analytes", y="Value", data=plot_data, scale="area", hue="Metric", split=True, palette="Set2", scale_hue=True, inner='quartile')
    ax.legend(loc=3)
    plt.savefig(f"./molecular_twin_pilot/MTPilot_singleomic_violins_{label}_{metric1}_{metric2}{'_multi' if add_multiomic else ''}.png", dpi=600)


# Function to plot asymetric violin performance plots for multi-omic models
def generate_multiomic_violin_plots(input_df, metric1, metric2, label):
    study_label_df = input_df.loc[(input_df["Study Label"] == label) & (input_df["Analytes"] != ""), PLOT_COLUMNS]
    study_label_df.sort_values(["Study Label", "Analytes"], ascending=False, inplace=True)
    sens_data = study_label_df[['Analytes', metric1]]
    sens_data.rename(columns={metric1:'Value'}, inplace=True)
    sens_data['Metric'] = metric1
    spec_data = study_label_df[['Analytes', metric2]]
    spec_data.rename(columns={metric2:'Value'}, inplace=True)
    spec_data['Metric'] = metric2

    plot_data = pd.concat([sens_data, spec_data])
    plt.figure(figsize=(32, 12))
    sns.set(font_scale = 2)
    ax = sns.violinplot(x="Analytes", y="Value", data=plot_data, scale="count", hue="Metric", split=True, palette="Set2", scale_hue=True, inner='quartile', cut=0, bw=0.5)
    ax.legend(loc=3)
    plt.savefig(f'./molecular_twin_pilot/MTPilot_multiomic_violins_{label}_{metric1}_{metric2}.png', dpi=600)


# +
# Generate asymmetric violin plots per single-omics analytes x=analyte, y=accuracy, hue= (with sensitivity and specificity as left and right violin thicknesses).
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_COLUMNS = ["Analytes", "Study Label", "Accuracy", "Sens", "Spec", "PPV", "NPV", "F1"]
metric_pairs = [("Sens", "Spec"), ("Accuracy", "PPV")]
        
for add_multiomic in [True, False]:
    for metric_pair in metric_pairs:
        for label in study_labels:
            generate_singleomic_violin_plots(binary_categorical_df, metric_pair[0], metric_pair[1], label, add_multiomic)

# +
# Generate asymmetric violin plots per multi-omics analytes  (with sensitivity and specificity as left and right violin thicknesses).
import matplotlib.pyplot as plt
import seaborn as sns

PLOT_COLUMNS = ["Analytes", "Study Label", "Accuracy", "Sens", "Spec", "PPV", "NPV", "F1"]
metric_pairs = [("Sens", "Spec"), ("Accuracy", "PPV")]

#deep_learning = binary_categorical_df[binary_categorical_df["Classifier Type"].isin(['RFE_RF_Model', 'RFE_LR_Model', 'L1_Norm_MLP_Model'])]
flattened_multiomic_analytes = binary_categorical_df.Analytes.str.split(',', expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'Analytes'}).join(binary_categorical_df.drop('Analytes', axis=1))
flattened_multiomic_analytes['Analytes'] = flattened_multiomic_analytes['Analytes'].str.strip()
#flattened_multiomic_analytes = flattened_multiomic_analytes[flattened_multiomic_analytes.Accuracy > 0.7]
for metric_pair in metric_pairs:
    for label in study_labels:
        generate_multiomic_violin_plots(flattened_multiomic_analytes, metric_pair[0], metric_pair[1], label)

# +
# Function to plot asymetric violin performance plots for multi-omic models, one vs rest of analytes
import matplotlib.pyplot as plt
import seaborn as sns
def generate_multiomic_violin_plots_one_vs_rest(input_df, metric, label):
    study_label_df = input_df.loc[(input_df["Study Label"] == label) & (input_df["Analytes"] != ""), PLOT_COLUMNS]
    study_label_df.sort_values(["Study Label", "Analytes"], ascending=False, inplace=True)
    plot_data = pd.DataFrame()
    for analyte in analytes:
        if analyte != "Multiomic":
            sens_data = study_label_df[study_label_df['Analytes'] == analyte][['Analytes', metric]]
            sens_data.rename(columns={metric:'Value'}, inplace=True)
            sens_data['Metric'] = 'Analyte'
            spec_data = study_label_df[study_label_df['Analytes'] != analyte][['Analytes', metric]]
            spec_data.rename(columns={metric:'Value'}, inplace=True)
            spec_data['Analytes'] = analyte
            spec_data['Metric'] = 'Rest'
            plot_data = pd.concat([plot_data, sens_data, spec_data])
    plt.figure(figsize=(32, 12))
    sns.set(font_scale = 2)
    ax = sns.violinplot(x="Analytes", y="Value", data=plot_data, scale="count", width=-.2, hue="Metric", split=True, palette="Set2", scale_hue=True, inner='quartile', cut=0)
    ax.legend(loc=3)
    plt.savefig(f'./molecular_twin_pilot/MTPilot_multiomic_violins_{label}_{metric}_one_vs_rest.png', dpi=600)

flattened_multiomic_analytes = binary_categorical_df.Analytes.str.split(',', expand=True).stack().reset_index(level=0).set_index('level_0').rename(columns={0:'Analytes'}).join(binary_categorical_df.drop('Analytes', axis=1))
flattened_multiomic_analytes['Analytes'] = flattened_multiomic_analytes['Analytes'].str.strip()
#flattened_multiomic_analytes = flattened_multiomic_analytes[flattened_multiomic_analytes.Accuracy > 0.7]
generate_multiomic_violin_plots_one_vs_rest(flattened_multiomic_analytes, 'Accuracy', 'label_deceased')

# +
import plotly.express as plt_exp
# %matplotlib inline

# Scatter plots of analyte performances.
scatter_plot_df = binary_categorical_df.loc[binary_categorical_df["Analytes"] != ""][STAT_COLUMNS]
scatter_plot_df['Analyte'] = np.where(scatter_plot_df['# Analytes'] == 1, scatter_plot_df['Analytes'], 'Multiomic')
scatter_plot_df['Analyte Encoded'] = scatter_plot_df['Analyte'].map({'Multiomic': 0, 'Clinical': 10, 'CNV': 1, 'SNV': 2, 'INDEL': 3, 'RNA_Expr': 4, 'RNA_Fusion': 5, 'Plasma_Lipid': 6, 'Plasma_Protein': 7, 'Pathology': 8, 'Tissue_Protein': 9, })
metric_pairs = [("Sens", "Spec"), ("Accuracy", "PPV")]
color_by_metrics = ["Classifier Type", "# Analytes"]

for color_by_metric in color_by_metrics:
    for metric_pair in metric_pairs:
        for label in study_labels:
            subset_df = scatter_plot_df[scatter_plot_df["Study Label"] == label]
            fig = plt_exp.scatter(subset_df, x = metric_pair[0], y = metric_pair[1], color = color_by_metric,
                                  marginal_x='violin', marginal_y='violin', width=1500, height=1500,
                                  title=f"MTPilot {metric_pair[0]} vs {metric_pair[1]} for {'Deceased' if label == 'label_deceased' else 'Recurred'} colored by {color_by_metric}")
            fig.update_layout(font_size=24)
            fig.update_layout(legend=dict(yanchor="top", y=1.0, xanchor="right", x=1.0))
            fig.show()
            fig.write_image(f"./molecular_twin_pilot/MTPilot_scatter_{label}_{metric_pair[0]}_{metric_pair[1]}_colorby_{color_by_metric}.png")


# +
# Scatter plots of analyte performances.
scatter_plot_df = binary_categorical_df.loc[binary_categorical_df["Analytes"] != ""][STAT_COLUMNS]
scatter_plot_df['Analyte'] = np.where(scatter_plot_df['# Analytes'] == 1, scatter_plot_df['Analytes'], 'Multiomic')
scatter_plot_df['Analyte Encoded'] = scatter_plot_df['Analyte'].map({'Multiomic': 0, 'Clinical': 10, 'CNV': 1, 'SNV': 2, 'INDEL': 3, 'RNA_Expr': 4, 'RNA_Fusion': 5, 'Plasma_Lipid': 6, 'Plasma_Protein': 7, 'Pathology': 8, 'Tissue_Protein': 9, })
metric_pairs = [("Sens", "Spec"), ("Accuracy", "PPV")]     

for clip_threshold in [True, False]:
    for metric_pair in metric_pairs:
        for label in study_labels:
            subset_df = scatter_plot_df[scatter_plot_df["Study Label"] == label]
            if clip_threshold:
                subset_df = subset_df[subset_df["Accuracy"] > 0.75]
            fig = plt.figure(figsize=(12, 10))
            axis = plt.scatter(subset_df[metric_pair[0]], subset_df[metric_pair[1]], c=subset_df['# Analytes'], alpha=0.8, cmap='RdBu')
            plt.xlabel(metric_pair[0])
            plt.ylabel(metric_pair[1])
            cbar = fig.colorbar(axis)
            cbar.set_label('# Analytes')
            plt.title(f"MTPilot {metric_pair[0]} vs {metric_pair[1]} for {'Deceased' if label == 'label_deceased' else 'Recurred'} {'(top quartile)' if clip_threshold else ''}")
            plt.savefig(f"./molecular_twin_pilot/MTPilot_scatter_{label}_{metric_pair[0]}_{metric_pair[1]}{'_clipped' if clip_threshold else ''}.png", dpi=600)


# Visualize model results in PandasGUI (visual dataframe charting library)
# #!pip install pandasgui
from pandasgui import show
gui = show(binary_categorical_df)

# LUX is another package that automatically suggests visualizations on Pandas Dataframe based on available columns
# #!pip install lux-api
# #!jupyter nbextension install --py luxwidget
# #!jupyter nbextension enable --py luxwidget
import lux
classifier_results_df = pd.read_csv(ALL_MODELS_FULL_SAMPLES)
classifier_results_df

binary_categorical_df.shape

binary_categorical_df.loc[binary_categorical_df.groupby(["Study Label"])["Accuracy"].idxmax(), STAT_COLUMNS]

binary_categorical_df[(binary_categorical_df["Study Label"] == 'label_deceased') & (binary_categorical_df.Accuracy > 0.82)].sort_values(["Study Label", "Accuracy"], ascending=False)

binary_categorical_df.loc[binary_categorical_df.groupby(["Study Label"])["Accuracy"].idxmin(), STAT_COLUMNS]

binary_categorical_df.loc[726].head(35)

dataset_json = binary_categorical_df[STAT_COLUMNS]
dataset_json = dataset_json.sample(n=1016).to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          iframe = document.getElementById('iframe');
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=dataset_json)
display(HTML(html))

# Table of top single-omics models performance summaries
#singleomics_baseline_columns = ["plasma_protein_", "tissue_protein_", "tissue_lipid_", "pathology_NF", "CNV_", "freebayes_SNV_", "pindel_INDEL_", "AF4_", "rna_gene_expr_",]
#singleomics_models = binary_categorical_df[(binary_categorical_df['Analytes'].str.split(',').apply(len)) == 1]
singleomics_models = binary_categorical_df.loc[(binary_categorical_df['# Analytes'] == 1)]
singleomics_models
#singleomics_models.loc[singleomics_models.groupby(["Study Label", "Analytes"])["Accuracy"].idxmax()]


singleomics_models.groupby(["Study Label", "Analytes"])["Accuracy"].idxmax()

singleomics_models.loc[singleomics_models.groupby(["Study Label", "Analytes"])["Accuracy"].idxmax(), STAT_COLUMNS]

#singleomics_models.loc[singleomics_models.groupby(["Study Label", "Analytes"])["Accuracy"].idxmax()]
singleomics_per_label_per_analyte_results = singleomics_models.groupby(["Study Label", "Analytes"])
singleomics_per_label_per_analyte_results["Accuracy"].max()

binary_categorical_df.loc[(binary_categorical_df['# Analytes'] == 1) & (binary_categorical_df.groupby(["Study Label"])["Accuracy"].idxmax()), STAT_COLUMNS]

# Table of top performing multi-omic models
binary_categorical_df.loc[binary_categorical_df.groupby(["Study Label"])["Accuracy"].idxmax(), STAT_COLUMNS]

analyte = binary_categorical_df.loc[610, 'Analytes'].split(',')
analyte[1]

# Baseline models
#clinical_baseline_columns = ["clinical_", "surgery_embed_", "pathology_embed_", "chemotherapy_embed_"]
singleomics_baseline_columns = ["plasma_protein_", "tissue_protein_", "tissue_lipid_", "pathology_NF", "CNV_", "freebayes_SNV_", "pindel_INDEL_", "AF4_", "rna_gene_expr_",]
baseline_models = multiomic_grid_search[multiomic_grid_search['feature_prefix'].str.split(',').apply(len) == 2]

baseline_models['first_feature'] = baseline_models.feature_prefix.str.split(',').str[0].str[2:-1]

baseline_models['second_feature'] = baseline_models.feature_prefix.str.split(',').str[1].str[1:-2]
baseline_models.shape

baseline_models

single_omics_baseline = baseline_models[baseline_models.second_feature == ""]
single_omics_baseline_performance = single_omics_baseline.loc[:,~single_omics_baseline.columns.str.contains('top_15_feature')]
single_omics_baseline_performance

single_omics_baseline_performance.loc[single_omics_baseline_performance.groupby(["target_label", "first_feature"])["F1"].idxmax()]

dataset_json = single_omics_baseline_performance
dataset_json = dataset_json.sample(n=84).to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="800"></facets-dive>
        <script>
          var data = {jsonstr};
          iframe = document.getElementById('iframe');
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=dataset_json)
display(HTML(html))

clinical_baseline = baseline_models[((baseline_models.first_feature == "clinical_") | (baseline_models.second_feature == "'clinical_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
clinical_baseline

single_omics_baseline.loc[742].head(15)

af4_baseline = baseline_models[((baseline_models.first_feature == "AF4_") | (baseline_models.second_feature == "'AF4_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
af4_baseline

cnv_baseline = baseline_models[((baseline_models.first_feature == "CNV_") | (baseline_models.second_feature == "'CNV_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
cnv_baseline

snv_baseline = baseline_models[((baseline_models.first_feature == "freebayes_SNV_") | (baseline_models.second_feature == "'freebayes_SNV_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
snv_baseline

indel_baseline = baseline_models[((baseline_models.first_feature == "pindel_INDEL_") | (baseline_models.second_feature == "'pindel_INDEL_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
indel_baseline

protein_baseline = baseline_models[((baseline_models.second_feature == "") & (baseline_models.model_type != "PCA_LR_Model") & (baseline_models.model_type != "L1_Norm_MLP_Model") & (baseline_models.target_label == "label_recurrence_binary"))].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
protein_baseline

pathology_baseline = baseline_models[((baseline_models.first_feature == "pathology_NF") | (baseline_models.second_feature == "'pathology_NF")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
pathology_baseline

pathology_baseline = baseline_models[(baseline_models.feature_prefix == "('pathology_NF',)")]
pathology_baseline

pathology_baseline.loc[131].head(20)

rna_gene_expr_baseline = baseline_models[((baseline_models.first_feature == "rna_gene_expr_") | (baseline_models.second_feature == "'rna_gene_expr_")) &
                               (baseline_models.model_type == "L1_Norm_RF_Model")].sort_values(["target_label", "test_loo_accuracy"], ascending=False)
rna_gene_expr_baseline

performance_multiomic_columns = multiomic_grid_search.loc[:,~multiomic_grid_search.columns.str.startswith('top_15_feature')]
top_performing_multiomic_models = performance_multiomic_columns.groupby(["target_label"]).apply(lambda grp: grp.nlargest(10, 'test_loo_accuracy'))
top_performing_multiomic_models

performance_multiomic_columns = multiomic_grid_search.loc[:,~multiomic_grid_search.columns.str.startswith('top_15_feature')]
top_performing_multiomic_models = performance_multiomic_columns.groupby(["target_label"]).apply(lambda grp: grp.nlargest(10, 'F1'))
top_performing_multiomic_models

multiomic_grid_search.loc[147].head(20)

multiomic_grid_search.head(20)

multiomic_grid_search = pd.read_csv(ALL_MODELS_FULL_SAMPLES)
multiomic_grid_search.drop('Unnamed: 0', axis=1, inplace=True)
multiomic_grid_search['F1'] = multiomic_grid_search['TP']/(multiomic_grid_search['TP'] + ((multiomic_grid_search['FP'] + multiomic_grid_search['FN'])/2))
dataset_json_df = multiomic_grid_search.loc[:,~multiomic_grid_search.columns.str.contains('top_15_feature')]

dataset_json_df

dataset_json_df.replace(to_replace=r'\'|\[|\]|\(|\)', value='', regex=True, inplace=True)

dataset_json = dataset_json_df.sample(n=870).to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="500"></facets-dive>
        <script>
          var data = {jsonstr};
          iframe = document.getElementById('iframe');
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=dataset_json)
display(HTML(html))

# +
full_samples_run = pd.read_csv(ALL_MODELS_FULL_SAMPLES)
full_samples_run.drop('Unnamed: 0', axis=1, inplace=True)
full_samples_run['F1'] = full_samples_run['TP']/(full_samples_run['TP'] + ((full_samples_run['FP'] + full_samples_run['FN'])/2))
full_samples_run = full_samples_run.loc[:,~full_samples_run.columns.str.contains('top_15_feature')]

min_samples_run = pd.read_csv(ALL_MODELS_FULL_SAMPLES)
min_samples_run.drop('Unnamed: 0', axis=1, inplace=True)
min_samples_run['F1'] = min_samples_run['TP']/(min_samples_run['TP'] + ((min_samples_run['FP'] + min_samples_run['FN'])/2))
min_samples_run = min_samples_run.loc[:,~min_samples_run.columns.str.contains('top_15_feature')]

combined_df = pd.concat([full_samples_run, min_samples_run], ignore_index=True)
combined_df
# -

combined_df.replace(to_replace=r'\'|\[|\]|\(|\)', value='', regex=True, inplace=True)
dataset_json = combined_df.sample(n=1128).to_json(orient='records')
HTML_TEMPLATE = """
        <script src="https://cdnjs.cloudflare.com/ajax/libs/webcomponentsjs/1.3.3/webcomponents-lite.js"></script>
        <link rel="import" href="https://raw.githubusercontent.com/PAIR-code/facets/1.0.0/facets-dist/facets-jupyter.html">
        <facets-dive id="elem" height="600"></facets-dive>
        <script>
          var data = {jsonstr};
          iframe = document.getElementById('iframe');
          document.querySelector("#elem").data = data;
        </script>"""
html = HTML_TEMPLATE.format(jsonstr=dataset_json)
display(HTML(html))

# Ingest and process clinical baseline models
baseline_clinical_columns = ["['clinical_']", "['surgery_embed_']", "['pathology_embed_']", "['chemotherapy_embed_']", "('AF4_',)", "('freebayes_SNV_',)", "('pindel_INDEL_',)", "('CNV_',)", "('protein_',)", "('rna_gene_expr_',)", "('pathology_NF',)",]
baseline_clinical = multiomic_grid_search[multiomic_grid_search['feature_prefix'].isin(baseline_clinical_columns)]
baseline_clinical

# Baseline models
baseline_clinical_models = baseline_clinical[baseline_clinical['feature_prefix'].str.split(',').apply(len) == 2]
baseline_clinical_models

baseline_clinical_columns = baseline_clinical.loc[:,~baseline_clinical.columns.str.startswith('top_15_feature')]
top_performing_baseline_clinical_models = baseline_clinical_columns.groupby(["target_label", "feature_prefix"]).apply(lambda grp: grp.nlargest(1, 'test_loo_accuracy'))
top_performing_baseline_clinical_models

baseline_clinical[baseline_clinical.feature_prefix.isin(["['clinical_']"])]

baseline_clinical.loc[3].head(20)

protein_tensors = "./molecular_twin_pilot/CS_MT_normalized_embedding_projector/CS_MT_embedding_projector_protein_dataset.tsv"
protein_tensors_df = pd.read_csv(protein_tensors, delimiter='\t')
protein_tensors_df

protein_tensors_df.dropna(axis='index', how='all', inplace=True)
dataset_tsv = protein_tensors_df.to_csv('./molecular_twin_pilot/CS_MT_normalized_embedding_projector/CS_MT_embedding_projector_protein_dataset_pruned.tsv', sep='\t', na_rep='0.0', index=False, header=False)

