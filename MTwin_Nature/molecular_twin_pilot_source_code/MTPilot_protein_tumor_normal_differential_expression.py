# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# Copyright (C) 2022 - Betteromics Inc.
# %load_ext autoreload
# %autoreload 2

# +
import dataset_manager.proto.workflow_pb2 as wf
from dataset_manager.workflow_engine.utils import (createOperation, sanitize_colname, createDirectorySource,
                                                   createSource, createWorkflow, cleanUpRCommand)
from google.protobuf.json_format import MessageToJson
from IPython.core.display import display, HTML
from IPython.display import IFrame
from witwidget.notebook.visualization import WitConfigBuilder
from witwidget.notebook.visualization import WitWidget

import google.protobuf.json_format as json_format
import functools
import ipyparams
import numpy as np
import os, datetime
import oyaml as yaml
import re
import pandas as pd
#import dataset_manager.workflow_engine.argo.engine as engine
#import dataset_manager.workflow_engine.argo.translation as translation
import sister
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_data_validation as tfdv

# Importing visualization libraries
from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# -

import pandas as pd
BASE_DIR = "/Users/onikolic/Downloads/molecular_twin_pilot/proteomics/"
plasmaProteinRaw = pd.read_csv(BASE_DIR + "MTPilot_Proteomics_Plasma_combined.tsv", delimiter='\t')
tissueProteinRaw = pd.read_csv(BASE_DIR + "TissueProteinLevel_AllPatientSamples.csv")
clinicalLabels = pd.read_csv("/Users/onikolic/Downloads/molecular_twin_pilot/outputs/clinical_labels.csv")

# ### Unpaired plasma protein differential expression
# - Remove proteins with more than 25% missing values
# - Fill missing values via median/2 (alternative options: zero, mean, median, imputeNN)
# - Remove proteins with low variance
# - Remove highly correlated proteins
# - Drop proteomic features where Mann-Whitney U-test between each protein's unpaired tumor - normal distributions are below certain threshold (p-value < 0.05)
# https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/

# +
import pandas as pd
import numpy as np
from enum import Enum
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance", copy=False, add_indicator=False)

class FillNA_Strategy(Enum):
    ZERO = 1
    AVG_COLUMN = 2
    MEDIAN_COLUMN = 3
    MEDIAN_DIV_2_COLUMN = 4
    IMPUTE_NN = 5
    AVG_CONTROL = 6
    
def fillNA(dataframe, fill_strategy=FillNA_Strategy.ZERO):
    if (fill_strategy == FillNA_Strategy.ZERO):
        dataframe.fillna(value=0, inplace=True)
    elif (fill_strategy == FillNA_Strategy.AVG_COLUMN):
        dataframe.fillna(dataframe.mean(), inplace=True, )
    elif (fill_strategy == FillNA_Strategy.MEDIAN_COLUMN):
        dataframe.fillna(dataframe.median(), inplace=True)
    elif (fill_strategy == FillNA_Strategy.MEDIAN_DIV_2_COLUMN):
        median_div_2 = dataframe.median() / 2
        dataframe.fillna(median_div_2, inplace=True)
    elif (fill_strategy == FillNA_Strategy.IMPUTE_NN):
        cleanedNumericFeatures = dataframe.select_dtypes(include=['number'])
        imputer.fit_transform(cleanedNumericFeatures)
    else:
        print("unknown fill_na strategy: " + fill_na)
    return dataframe
    
def plasmaProteinFeaturesCleaned(plasmaProteinFeaturesRenamed):
    import numpy as np

    cleaned = plasmaProteinFeaturesRenamed
    # Drop unnecessary columns.
    cleaned.drop(columns=['Subject', 'Group', '1/iRT_protein'], inplace=True, errors='ignore')

    cleaned.columns = cleaned.columns.str.replace(',', '')
    cleaned.columns = cleaned.columns.str.replace('(', '')
    cleaned.columns = cleaned.columns.str.replace(')', '')
    cleaned.columns = cleaned.columns.str.replace('|', '_')
    cleaned.columns = cleaned.columns.str.replace(' ', '_')
    cleaned.columns = cleaned.columns.str.replace('/', '_')

    # Remove errant commas and semi-colons within cells for csv parsing
    cleaned.replace(',', '', regex=True, inplace=True)
    cleaned.replace(';', '', regex=True, inplace=True)
    cleaned.replace('\([0-9]*\)', '', regex=True, inplace=True)
    
    # Drop empty rows and columns, fill empty cells with appropriate defaults.
    cleaned.dropna(axis='index', how='all', inplace=True)
    cleaned.dropna(axis='columns', how='all', inplace=True)

    return fillNA(cleaned, FillNA_Strategy.MEDIAN_DIV_2_COLUMN)

# Filter out proteomic features where tumor distribution approximates normals (ie via Mann-Whitney U-test between unpaired tumor - normal distributions) having two-tailed p-value < 0.05
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp

class DistributionSimilarityStrategy(Enum):
    MANN_WHITNEY_U_TEST = 1
    WILCOXON = 2
    KOLMOGOROV_SMIRNOV = 3

def getDistributionSimilarityTest(normal_dataframe, tumor_dataframe,
                                  distribution_similarity_strategy=DistributionSimilarityStrategy.MANN_WHITNEY_U_TEST):
    distribution_similarity_test = pd.DataFrame()

    # Perform distribution similarity test between tumor and normal distributions per protein.
    for protein_column in tumor_dataframe.select_dtypes([np.number]).columns:
        normal_protein_data = []
        if protein_column in normal_dataframe.columns:
            normal_protein_data = normal_dataframe[protein_column]
            if (distribution_similarity_strategy == DistributionSimilarityStrategy.MANN_WHITNEY_U_TEST):
                distribution_similarity_test[protein_column] = mannwhitneyu(normal_protein_data, tumor_dataframe[protein_column])
            elif (distribution_similarity_strategy == DistributionSimilarityStrategy.WILCOXON):
                distribution_similarity_test[protein_column] = wilcoxon(normal_protein_data, tumor_dataframe[protein_column])
            elif (distribution_similarity_strategy == DistributionSimilarityStrategy.KOLMOGOROV_SMIRNOV):
                distribution_similarity_test[protein_column] = ks_2samp(normal_protein_data, tumor_dataframe[protein_column])
        else:
            print(f"Protein {protein_column} is not present in the normals.  Cannot establish prior distribution")
            # Ensure tumor proteins with no normal pairs are retained in the statistical similarity tests.
            #distribution_similarity_test[protein_column] = [100, 0.01]
    return distribution_similarity_test

# Drop out low variance proteins.
from sklearn.feature_selection import VarianceThreshold

def pruneLowVarianceFeatures(features, metadataColumns, threshold=0.10):
    feature_selector = VarianceThreshold(threshold)
    numericFeatures = features.drop(metadataColumns, axis=1)
    feature_selector.fit(numericFeatures)
    high_variance_columns = numericFeatures.columns[feature_selector.get_support(indices=True)]
    return features[ metadataColumns + high_variance_columns.tolist() ]

# Get highly correlated columns that can be dropped out
def get_correlated_columns(df_in, threshold=0.95):
    max_cols = min(20000, len(df_in.columns))
    clipped_df = df_in.iloc[:, : max_cols]

    # Create correlation matrix
    corr_matrix = clipped_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find columns with correlation greater than threshold
    correlated_columns = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Return all highly correlated columns
    return correlated_columns

def log_normalize(df):
    for column in df.select_dtypes(include = [np.number]).columns:
        df[column] = np.log10(df[column])
    return df
    
plasmaProteinRaw['Subject'] = plasmaProteinRaw.Subject.astype(str)
plasmaProteinRaw['Biobank_Number']  = plasmaProteinRaw['Subject']

# Split combined, raw plasma proteins into tumor and normal datasets.
plasmaProtein_Tumor = plasmaProteinRaw.loc[plasmaProteinRaw.Subject.str.contains('GI-')]
plasmaProtein_Normal = plasmaProteinRaw.loc[plasmaProteinRaw.Subject.str.contains('HMN')]

plasmaProtein_Tumor.columns = plasmaProtein_Tumor.columns.str.replace("_HUMAN", "")
plasmaProtein_Normal.columns = plasmaProtein_Normal.columns.str.replace("_HUMAN", "")

# Drop out proteins with missing values above 25%
plasmaProtein_Tumor_dense = plasmaProtein_Tumor[plasmaProtein_Tumor.columns[plasmaProtein_Tumor.isnull().mean() < 0.75]]
plasmaProtein_Normal_dense = plasmaProtein_Normal[plasmaProtein_Normal.columns[plasmaProtein_Normal.isnull().mean() < 0.75]]

# Drop out low variance proteins.
plasmaProtein_Tumor_low_var = pruneLowVarianceFeatures(plasmaProtein_Tumor_dense, ['Biobank_Number', 'Subject', 'Group'], threshold=0.2)
plasmaProtein_Normal_low_var = pruneLowVarianceFeatures(plasmaProtein_Normal_dense, ['Biobank_Number', 'Subject', 'Group'], threshold=0.2)

# Clean & FillNAs: 0, mean_column, median_column, imputeNN
plasmaProteinTumor_fillNAs = plasmaProteinFeaturesCleaned(plasmaProtein_Tumor_low_var)
plasmaProteinNormal_fillNAs = plasmaProteinFeaturesCleaned(plasmaProtein_Normal_low_var)

# Log transform all numeric columns in dataframe as last step
plasmaProteinTumor_logNorm = log_normalize(plasmaProteinTumor_fillNAs)
plasmaProteinNormal_logNorm = log_normalize(plasmaProteinNormal_fillNAs)

# Drop out highly correlated tumor proteins as an optimization
plasmaProteinTumor_fillNAs.drop(get_correlated_columns(plasmaProteinTumor_fillNAs, 0.95), axis=1, inplace=True)

plasma_protein_distribution_similarity_test = getDistributionSimilarityTest(plasmaProteinNormal_logNorm, plasmaProteinTumor_logNorm, DistributionSimilarityStrategy.MANN_WHITNEY_U_TEST)
retain_columns = []
for protein_column in plasma_protein_distribution_similarity_test:
    if plasma_protein_distribution_similarity_test.iloc[1][protein_column] < 0.05:
        retain_columns.append(protein_column)
prunedPlasmaTumorProteins = plasmaProteinTumor_logNorm[retain_columns + ["Biobank_Number"]]
# -

# Save featurized, plasma proteomic dataset
OUTPUT_DIR = "./molecular_twin_pilot/outputs/clinical"
prunedPlasmaTumorProteins.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_plasma_proteomic_features.csv"), index=False)
plasmaProtein_Tumor.to_csv(os.path.join(OUTPUT_DIR, "raw_plasma_proteomic_features.csv"), index=False)

prunedPlasmaTumorProteins.head()

plasmaProteinRaw.shape

plasmaProtein_Tumor.shape

plasmaProtein_Normal.shape

prunedPlasmaTumorProteins.shape

# Look at combined Tumor & Normal statistics from raw input file
plasmaProteinRaw_statistics = tfdv.generate_statistics_from_dataframe(plasmaProteinRaw)
tfdv.visualize_statistics(plasmaProteinRaw_statistics)

# Look at raw distribution tfdv.stats on columns for Tumor vs Normal
plasmaProteinTumor_statistics = tfdv.generate_statistics_from_dataframe(plasmaProtein_Tumor)
plasmaProteinNormal_statistics = tfdv.generate_statistics_from_dataframe(plasmaProtein_Normal)
tfdv.visualize_statistics(lhs_statistics=plasmaProteinNormal_statistics, lhs_name='Plasma Normals', rhs_statistics=plasmaProteinTumor_statistics, rhs_name='Plasma Tumor')

# +
# Look at distribution tfdv.stats on cleaned columns
plasmaProteinTumorFillNAs_statistics = tfdv.generate_statistics_from_dataframe(plasmaProteinTumor_logNorm)
plasmaProteinNormalFillNAs_statistics = tfdv.generate_statistics_from_dataframe(plasmaProteinNormal_logNorm)

tfdv.visualize_statistics(lhs_statistics=plasmaProteinNormalFillNAs_statistics, lhs_name='Plasma Normals (fillNAs)', rhs_statistics=plasmaProteinTumorFillNAs_statistics, rhs_name='Plasma Tumor (fillNAs)')
# -

plasma_protein_distribution_similarity_test.head()

# Plot distribution similarity measure and p-values (for either independent Mann-Whitney U-test or dependent Wilcoxon signed-rank test)
plasma_protein_distribution_similarity_test_measure = plasma_protein_distribution_similarity_test.iloc[0].values
plasma_protein_distribution_similarity_test_measure.sort
plt.hist(plasma_protein_distribution_similarity_test_measure, 20, facecolor='blue', alpha=0.5)

# Plot Mann-Whitney U-test measure and p-value
plasma_protein_distribution_similarity_test_pvalue = plasma_protein_distribution_similarity_test.iloc[1].values
plasma_protein_distribution_similarity_test_pvalue.sort
plt.hist(plasma_protein_distribution_similarity_test_pvalue, 5, facecolor='blue', alpha=0.5, density=True)

# +
retain_columns = []
for protein_column in plasma_protein_distribution_similarity_test:
    #if plasma_protein_distribution_similarity_test.iloc[1][protein_column] < plasma_protein_distribution_similarity_test.loc[:, protein_column].quantile(0.25):
    if plasma_protein_distribution_similarity_test.iloc[1][protein_column] < 0.05:
        retain_columns.append(protein_column)
print(len(retain_columns))


# Retain only columns where tumor distribution differed sufficient from normal distribution according to Mann-Whitney test.
prunedPlasmaTumorProteins = plasmaProtein_Tumor[plasmaProtein_Tumor.columns[plasmaProtein_Tumor.columns.isin(retain_columns)]]
prunedPlasmaTumorProteins.shape
# -
# ### Paired tissue protein differential expression (pool_tumor/pool_normals, paired t/n,)
# - Remove proteins with more than 25% missing values
# - Fill missing values via median/2 (alternative options: zero, mean, median, imputeNN)
# - Remove proteins with low variance
# - Remove highly correlated proteins
# - Drop proteomic features where Wilcoxon T-test between each protein's paired tumor - normal distributions are below certain threshold (p-value < 0.05)
# https://machinelearningmastery.com/nonparametric-statistical-significance-tests-in-python/

# +
import numpy as np
import pandas as pd
import re
from enum import Enum
from sklearn.impute import KNNImputer
imputer = KNNImputer(n_neighbors=5, weights="distance", copy=False, add_indicator=False)

def remove_constant_value_features(df):
    return [e for e in df.columns if df[e].nunique() == 1]

class FillNA_Strategy(Enum):
    ZERO = 1
    AVG_COLUMN = 2
    MEDIAN_COLUMN = 3
    MEDIAN_DIV_2_COLUMN = 4
    IMPUTE_NN = 5
    AVG_CONTROL = 6
    
def fillNA(dataframe, fill_strategy=FillNA_Strategy.ZERO):
    if (fill_strategy == FillNA_Strategy.ZERO):
        dataframe.fillna(value=0, inplace=True)
    elif (fill_strategy == FillNA_Strategy.AVG_COLUMN):
        dataframe.fillna(dataframe.mean(), inplace=True, )
    elif (fill_strategy == FillNA_Strategy.MEDIAN_COLUMN):
        dataframe.fillna(dataframe.median(), inplace=True)
    elif (fill_strategy == FillNA_Strategy.MEDIAN_DIV_2_COLUMN):
        dataframe.fillna(dataframe.median() / 2, inplace=True)
    elif (fill_strategy == FillNA_Strategy.IMPUTE_NN):
        tissueProteinFeaturesNumericFeatures = dataframe.select_dtypes(include=['number'])
        imputer.fit_transform(tissueProteinFeaturesNumericFeatures)
    else:
        print("unknown fill_na strategy: " + fill_na)
    return dataframe

# Filter out proteomic features where tumor distribution approximates normals (ie via Mann-Whitney U-test between unpaired tumor - normal distributions) having two-tailed p-value < 0.05
from scipy.stats import mannwhitneyu
from scipy.stats import wilcoxon
from scipy.stats import ks_2samp

class DistributionSimilarityStrategy(Enum):
    MANN_WHITNEY_U_TEST = 1
    WILCOXON = 2
    KOLMOGOROV_SMIRNOV = 3

def getDistributionSimilarityTest(normal_dataframe, tumor_dataframe,
                                  distribution_similarity_strategy=DistributionSimilarityStrategy.MANN_WHITNEY_U_TEST):
    distribution_similarity_test = pd.DataFrame()

    # Compute distribution similarity between tumor and normals per protein.
    for protein_column in tumor_dataframe.select_dtypes([np.number]).columns:
        normal_protein_data = []
        if protein_column in normal_dataframe.columns:
            normal_protein_data = normal_dataframe[protein_column]
            if (distribution_similarity_strategy == DistributionSimilarityStrategy.MANN_WHITNEY_U_TEST):
                distribution_similarity_test[protein_column] = mannwhitneyu(normal_protein_data, tumor_dataframe[protein_column])
            elif (distribution_similarity_strategy == DistributionSimilarityStrategy.WILCOXON):
                distribution_similarity_test[protein_column] = wilcoxon(normal_protein_data, tumor_dataframe[protein_column])
            elif (distribution_similarity_strategy == DistributionSimilarityStrategy.KOLMOGOROV_SMIRNOV):
                distribution_similarity_test[protein_column] = ks_2samp(normal_protein_data, tumor_dataframe[protein_column])
        else:
            print(f"Protein {protein_column} is not present in the normals.  Cannot establish prior distribution")
            distribution_similarity_test[protein_column] = [100, 0.01] # Ensure tumor proteins with no normal pairs are retained in the statistical similarity tests.
    return distribution_similarity_test

def tissueProteinFeaturesCleaned(tissueProteinFeatures):
    # Drop unnecessary column
    tissueProteinFeatures.drop(columns=['SampleCount', 'Pool_Normal', 'Pool_Normal_1', 'Pool_Normal_2', 'Pool_Tumor', 'Pool_Tumor_1', 'Pool_Tumor_2'], inplace=True, errors='ignore')

    # Cleanup invalid characters in column names
    tissueProteinFeatures.columns = tissueProteinFeatures.columns.str.replace('_', '-')
    tissueProteinFeatures = tissueProteinFeatures.add_prefix("GI-")
    tissueProteinFeatures.rename(columns={"GI-Protein": "Biobank_Number"}, inplace=True)

    # Transpose protein features to be columnar.
    tissueProteinFeatures = tissueProteinFeatures.set_index('Biobank_Number').transpose()

    # Cleanup invalid characters in protein column names
    tissueProteinFeatures.columns = tissueProteinFeatures.columns.str.replace('|', '_')

    # Drop empty rows and columns, fill empty cells with appropriate defaults.
    tissueProteinFeatures.dropna(axis='index', how='all', inplace=True)
    tissueProteinFeatures.dropna(axis='columns', how='all', inplace=True)
    
    return tissueProteinFeatures

# Drop out low variance proteins.
from sklearn.feature_selection import VarianceThreshold

def pruneLowVarianceFeatures(features, metadataColumns, threshold=0.10):
    feature_selector = VarianceThreshold(threshold)
    numericFeatures = features.drop(metadataColumns, axis=1)
    feature_selector.fit(numericFeatures)
    high_variance_columns = numericFeatures.columns[feature_selector.get_support(indices=True)]
    return features[ metadataColumns + high_variance_columns.tolist() ]

# Get highly correlated columns that can be dropped out
def get_correlated_columns(df_in, threshold=0.95):
    max_cols = min(20000, len(df_in.columns))
    clipped_df = df_in.iloc[:, : max_cols]

    # Create correlation matrix
    corr_matrix = clipped_df.corr().abs()

    # Select upper triangle of correlation matrix
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))

    # Find columns with correlation greater than threshold
    correlated_columns = [column for column in upper.columns if any(upper[column] > threshold)]
    
    # Return all highly correlated columns
    return correlated_columns

def log_normalize(df):
    for column in df.select_dtypes(include = [np.number]).columns:
        df[column] = np.log10(df[column])
    return df

tissueProteinFeatures = tissueProteinFeaturesCleaned(tissueProteinRaw)

# Split tissue proteins into tumor and normal datasets, strip column suffixes. 
tissueProteinTumor = tissueProteinFeatures.loc[tissueProteinFeatures.index.str.contains('-T')]
tissueProteinTumor.index = tissueProteinTumor.index.str.replace("-T", "", regex=True)
tissueProteinNormal = tissueProteinFeatures.loc[tissueProteinFeatures.index.str.contains('-N')]
tissueProteinNormal.index = tissueProteinNormal.index.str.replace("-N", "", regex=True)

tissueProteinTumor.columns = tissueProteinTumor.columns.str.replace("_HUMAN", "")
tissueProteinNormal.columns = tissueProteinNormal.columns.str.replace("_HUMAN", "")

# Drop columns that contain all constants
tissueProteinNormal = tissueProteinNormal.drop(columns=remove_constant_value_features(tissueProteinNormal))
tissueProteinTumor = tissueProteinTumor.drop(columns=remove_constant_value_features(tissueProteinTumor))

# Subset tissue proteins only to paired samples with both tumor and normal protein readouts
normal_tissue_samples = tissueProteinNormal.index
tumor_tissue_samples = tissueProteinTumor.index

intersect_samples = normal_tissue_samples.intersection(tumor_tissue_samples)
tissueProteinNormal = tissueProteinNormal.loc[intersect_samples, :]
tissueProteinTumor = tissueProteinTumor.loc[intersect_samples, :]

# Subset to only proteins with both tumor & normal patients
tissue_tumor_proteins = tissueProteinTumor.columns
tissue_normal_proteins = tissueProteinNormal.columns

intersect_proteins = tissue_normal_proteins.intersection(tissue_tumor_proteins)
tissueProteinNormal = tissueProteinNormal.loc[:, intersect_proteins]
tissueProteinTumor = tissueProteinTumor.loc[:, intersect_proteins]

# Drop out proteins with missing values above 25%
tissueProteinTumor_dense = tissueProteinTumor[tissueProteinTumor.columns[tissueProteinTumor.isnull().mean() < 0.75]]
tissueProteinNormal_dense = tissueProteinNormal[tissueProteinNormal.columns[tissueProteinNormal.isnull().mean() < 0.75]]

# Fill null values for each protein
tissueProteinTumor_fillna = fillNA(tissueProteinTumor_dense, FillNA_Strategy.MEDIAN_DIV_2_COLUMN)
tissueProteinNormal_fillna = fillNA(tissueProteinNormal_dense, FillNA_Strategy.MEDIAN_DIV_2_COLUMN)

# Drop out low variance proteins.
#tissueProteinTumor_low_var = pruneLowVarianceFeatures(tissueProteinTumor_fillna, [], threshold=0.2)
#tissueProteinNormal_low_var = pruneLowVarianceFeatures(tissueProteinNormal_fillna, [], threshold=0.2)
tissueProteinTumor_low_var = tissueProteinTumor_fillna
tissueProteinNormal_low_var = tissueProteinNormal_fillna

# Drop out highly correlated tumor proteins as an optimization
tissueProteinTumor_low_var.drop(get_correlated_columns(tissueProteinTumor_low_var, 0.95), axis=1, inplace=True)

# Log transform all numeric columns in dataframe
def log_normalize(df):
    for column in df.select_dtypes(include = [np.number]).columns:
        df[column] = np.log10(df[column])
    return df

# Log transform all numeric columns in dataframe as last step
tissueProteinTumor_logNorm = log_normalize(tissueProteinTumor_low_var)
tissueProteinNormal_logNorm = log_normalize(tissueProteinNormal_low_var)

# Compute differential expression of proteins between tumor and normal tissue.
differentialProteinExpression = tissueProteinNormal.subtract(tissueProteinTumor, fill_value=0.0)

# Compute Wilcoxon test between paired, dependent tumor - normal samples
tissue_protein_distribution_similarity_test = getDistributionSimilarityTest(tissueProteinNormal_logNorm, tissueProteinTumor_logNorm, DistributionSimilarityStrategy.WILCOXON)

retain_columns = []
for protein_column in tissue_protein_distribution_similarity_test:
    if tissue_protein_distribution_similarity_test.iloc[1][protein_column] < 0.05:
        retain_columns.append(protein_column)

# Retain only proteins for which tumor distribution differed sufficiently from normal distribution according to Wilcoxon test.
prunedTissueTumorProteins = tissueProteinTumor_logNorm[retain_columns]
# -

# Save featurized, plasma proteomic dataset
OUTPUT_DIR = "./molecular_twin_pilot/outputs/clinical"
#OUTPUT_DIR = "/Users/onikolic/Downloads/"
prunedTissueTumorProteins['Biobank_Number']=prunedTissueTumorProteins.index
prunedTissueTumorProteins = prunedTissueTumorProteins.reset_index(drop=True)
prunedTissueTumorProteins.to_csv(os.path.join(OUTPUT_DIR, "preprocessed_tissue_proteomic_features.csv"), index=False)
tissueProteinTumor.to_csv(os.path.join(OUTPUT_DIR, "raw_tissue_proteomic_features.csv"), index=False)

tissueProteinTumor.head()

tissueProteinNormal.head()

tissueProteinTumor_dense.head()

tissueProteinNormal_dense.head()

tissueProteinTumor_logNorm.head()

tissueProteinNormal_logNorm.head()

prunedTissueTumorProteins.shape

tissue_protein_distribution_similarity_test.head()

# Plot distribution similarity measure (for either independent Mann-Whitney U-test or dependent Wilcoxon signed-rank test)
tissue_protein_distribution_similarity_test_measure = tissue_protein_distribution_similarity_test.iloc[0].values
tissue_protein_distribution_similarity_test_measure.sort
plt.hist(tissue_protein_distribution_similarity_test_measure, 20, facecolor='blue', alpha=0.5, density=True)

# Plot distribution similarity p-values (for either independent Mann-Whitney U-test or dependent Wilcoxon signed-rank test)
tissue_protein_distribution_similarity_test_pvalue = tissue_protein_distribution_similarity_test.iloc[1].values
tissue_protein_distribution_similarity_test_pvalue.sort
plt.hist(tissue_protein_distribution_similarity_test_pvalue, 20, facecolor='blue', alpha=0.5, density=True)

prunedTissueTumorProteins.shape

prunedDifferentialProteinExpression_statistics = tfdv.generate_statistics_from_dataframe(prunedTissueTumorProteins)
tfdv.visualize_statistics(prunedDifferentialProteinExpression_statistics)

# +
# Look at distribution tfdv.stats on cleaned columns
tissueProteinTumor_statistics = tfdv.generate_statistics_from_dataframe(tissueProteinTumor_logNorm)
tissueProteinNormal_statistics = tfdv.generate_statistics_from_dataframe(tissueProteinNormal_logNorm)

tfdv.visualize_statistics(lhs_statistics=tissueProteinNormal_statistics, lhs_name='Tissue Normals', rhs_statistics=tissueProteinTumor_statistics, rhs_name='Tissue Tumor')
# -

eda_numcat(prunedTissueTumorProteins, method='pps', x='label_outcome_binary')

# +
# Plasma Lipids cleaning.
LIPIDS_DIR = "./molecular_twin_pilot/lipidomics/"

lipidSpeciesConcentrationRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_lipid_species_concentration.csv'))
lipidSpeciesCompositionRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_lipid_species_composition.csv'))
lipidClassConcentrationRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_lipid_class_concentration.csv'))
lipidClassCompositionRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_lipid_class_composition.csv'))
fattyAcidSpeciesConcentrationRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_fatty_acid_concentration.csv'))
fattyAcidSpeciesCompositionRaw = pd.read_csv(os.path.join(LIPIDS_DIR, 'MTPilot_fatty_acid_composition.csv'))


# +
def prefixData(df, prefix):
    prefixedData = df.add_prefix(prefix)
    prefixedData.rename(columns={prefix + "Name": "Biobank_Number"}, inplace=True)
    return prefixedData

# Drop out low variance proteins.
from sklearn.feature_selection import VarianceThreshold

def pruneLowVarianceFeatures(features, metadataColumns, threshold=0.10):
    feature_selector = VarianceThreshold(threshold)
    numericFeatures = features.drop(metadataColumns, axis=1)
    feature_selector.fit(numericFeatures)
    high_variance_columns = numericFeatures.columns[feature_selector.get_support(indices=True)]
    return features[ metadataColumns + high_variance_columns.tolist() ]

lipidSpeciesConcentrationPrefix = prefixData(lipidSpeciesConcentrationRaw, 'species_conc_')
lipidSpeciesCompositionPrefix = prefixData(lipidSpeciesCompositionRaw, 'species_comp_')
lipidClassConcentrationPrefix = prefixData(lipidClassConcentrationRaw, 'class_conc_')
lipidClassCompositionPrefix = prefixData(lipidClassCompositionRaw, 'class_comp_')
fattyAcidSpeciesConcentrationPrefix = prefixData(fattyAcidSpeciesConcentrationRaw, 'fatty_acid_conc_')
fattyAcidSpeciesCompositionPrefix = prefixData(fattyAcidSpeciesCompositionRaw, 'fatty_acid_comp_')

mergedRawLipids = lipidSpeciesConcentrationPrefix.merge(lipidSpeciesCompositionPrefix, how='outer')
mergedRawLipids = mergedRawLipids.merge(lipidClassConcentrationPrefix, how='outer')
mergedRawLipids = mergedRawLipids.merge(lipidClassCompositionPrefix, how='outer')
mergedRawLipids = mergedRawLipids.merge(fattyAcidSpeciesConcentrationPrefix, how='outer')
mergedRawLipids = mergedRawLipids.merge(fattyAcidSpeciesCompositionPrefix, how='outer')

# Drop empty rows and columns, fill empty cells with appropriate defaults.
mergedLipidsPruned = mergedRawLipids.dropna(axis='index', how='all', inplace=False)
mergedLipidsPruned.dropna(axis='columns', how='all', inplace=True)

# Drop out lipids with missing values above 25%
mergedLipidsPruned_dense = mergedLipidsPruned[mergedLipidsPruned.columns[mergedLipidsPruned.isnull().mean() < 0.75]]

# Fill null values for each lipid
mergedLipidsPruned_fillNA = fillNA(mergedLipidsPruned_dense, FillNA_Strategy.MEDIAN_COLUMN)

# Drop out low variance lipids.
mergedLipidsPruned_low_var = pruneLowVarianceFeatures(mergedLipidsPruned_fillNA, ['Biobank_Number'], threshold=0.2)

# Save featurized, lipidomic dataset
OUTPUT_DIR = "./molecular_twin_pilot/outputs/clinical"
mergedLipidsPruned_low_var.to_csv(os.path.join(OUTPUT_DIR, "pruned_lipidomic_features.csv"), index=False)
# -

mergedRawLipids.shape

mergedLipidsPruned_dense.shape

mergedLipidsPruned_fillNA.shape

mergedLipidsPruned_low_var.shape

# +
# Merge all omic files into one
ROOT_DIR = "./molecular_twin_pilot/"
# Clinical+Labels
# DNA: SNV, CNV, INDELS
# RNA: Fusions, GeneExpr
# Proteomics: Plasma Proteins, Tissue Proteins, Lipids
# Pathology
clinicalFeatures = pd.read_csv(os.path.join(ROOT_DIR, "outputs/clinical", "early_stage_patients_clinical_features.csv"))
snvRaw = pd.read_csv(os.path.join(ROOT_DIR, "DNA/test-output", "process-freebayes-variants.csv"))
cnvRaw = pd.read_csv(os.path.join(ROOT_DIR, "DNA/test-output", "process-cnv-files.csv"))
indelRaw = pd.read_csv(os.path.join(ROOT_DIR, "DNA/test-output", "process-pindel-variants.csv"))
fusionRaw = pd.read_csv(os.path.join(ROOT_DIR, "RNA/test-output", "process-rna-fusion.csv"))

rnaGeneExprRaw = pd.read_csv(os.path.join(ROOT_DIR, "RNA/deseq2", "cancer_gene_level_abundances.csv"))
rnaGeneExprDiffExpr = pd.read_csv(os.path.join(ROOT_DIR, "RNA/deseq2", "differentially_expressed_genes_2000.csv"))

plasmaProteinRaw = pd.read_csv(os.path.join(ROOT_DIR, "outputs/clinical", "preprocessed_plasma_proteomic_features.csv"))
tissueProteinRaw = pd.read_csv(os.path.join(ROOT_DIR, "outputs/clinical", "preprocessed_tissue_proteomic_features.csv"))
lipidRaw = pd.read_csv(os.path.join(ROOT_DIR, "outputs/clinical", "pruned_lipidomic_features.csv"))
pathologyRaw = pd.read_csv(os.path.join(ROOT_DIR, "outputs/pathology", "pathologyFeatures.csv"))
# -

clinicalFeatures.shape

snvRaw.shape

cnvRaw.shape

indelRaw.shape

fusionRaw.shape

rnaGeneExprRaw.shape

plasmaProteinRaw.shape

tissueProteinRaw.shape

lipidRaw.shape

pathologyRaw.shape


# +
# Study Labels cleaning: 1) convert NAs to 0; 2) Columns already prefixed with 'label_'
def cleanLabel(labels):
    import re
    labels.fillna(0, downcast='infer', inplace=True)
    labels["Biobank_Number"] = labels["Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    labels["Biobank_Number"] = labels["Biobank_Number"].str.upper()
    return labels

# Clinical features cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'clinical_'
def cleanClinical(clinicalFeatures):
    import re
    clinicalFeatures.fillna(0, downcast='infer', inplace=True)
    #clinicalFeatures = clinicalFeatures.add_prefix('clinical_')
    #clinicalFeatures["Biobank_Number"] = clinicalFeatures["clinical_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    clinicalFeatures["Biobank_Number"] = clinicalFeatures["Biobank_Number"].str.upper()
    #clinicalFeatures.drop(["clinical_Biobank_Number"], axis=1, inplace=True)
    return clinicalFeatures

# RNASeq features cleaning: -1) remove uninformative genes; 0) Drop 'index' column; 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'rna_expr_'
def cleanRNASeq(rnaGeneExprFeatures, retain_geneset):
    import re
    rnaGeneExprFeatures.fillna(0, downcast='infer', inplace=True)
    rnaGeneExprFeatures["Biobank_Number"] = rnaGeneExprFeatures["sample"].str.upper()
    rnaGeneExprFeatures = rnaGeneExprFeatures[retain_geneset['gene_name'].tolist() + ['Biobank_Number']]
    rnaGeneExprFeatures = rnaGeneExprFeatures.add_prefix('rna_expr_')
    rnaGeneExprFeatures.rename(columns={"rna_expr_Biobank_Number": "Biobank_Number",}, inplace=True)
    return rnaGeneExprFeatures

# Plasma Protein cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'plasma_protein_'
def cleanPlasmaProtein(plasmaProteinFeatures):
    import re
    plasmaProteinFeatures.fillna(0, downcast='infer', inplace=True)
    plasmaProteinFeatures = plasmaProteinFeatures.add_prefix('plasma_protein_')
    plasmaProteinFeatures["Biobank_Number"] = plasmaProteinFeatures["plasma_protein_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    plasmaProteinFeatures["Biobank_Number"] = plasmaProteinFeatures["Biobank_Number"].str.upper()
    plasmaProteinFeatures.drop(["plasma_protein_Biobank_Number"], axis=1, inplace=True)
    return plasmaProteinFeatures

# Tissue Protein cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'tissue_protein_'
def cleanTissueProtein(tissueProteinFeatures):
    import re
    tissueProteinFeatures.fillna(0, downcast='infer', inplace=True)
    tissueProteinFeatures = tissueProteinFeatures.add_prefix('tissue_protein_')
    tissueProteinFeatures.rename(columns={"tissue_protein_Biobank_Number": "Biobank_Number"}, inplace=True)
    return tissueProteinFeatures

# Plasma Lipid cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'plasma_lipid_'
def cleanPlasmaLipid(plasmaLipidFeatures):
    import re
    plasmaLipidFeatures.fillna(0, downcast='infer', inplace=True)
    plasmaLipidFeatures = plasmaLipidFeatures.add_prefix('plasma_lipid_')
    plasmaLipidFeatures.rename(columns={"plasma_lipid_Biobank_Number": "Biobank_Number"}, inplace=True)
    return plasmaLipidFeatures

# Pathology cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'pathology_'
def cleanPathology(pathologyFeatures):
    import re
    pathologyFeatures.fillna(0, downcast='infer', inplace=True)
    pathologyFeatures = pathologyFeatures.add_prefix('pathology_')
    pathologyFeatures["Biobank_Number"] = pathologyFeatures["pathology_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    pathologyFeatures["Biobank_Number"] = pathologyFeatures["Biobank_Number"].str.upper()
    pathologyFeatures.drop(["pathology_Biobank_Number"], axis=1, inplace=True)
    return pathologyFeatures

# DNA CNVs cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'CNV_'
def cleanCnv(cnvFeatures):
    import re
    cnvFeatures.fillna(0, downcast='infer', inplace=True)
    cnvFeatures["Biobank_Number"] = cnvFeatures["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    cnvFeatures["Biobank_Number"] = cnvFeatures["Biobank_Number"].str.upper()
    cnvFeatures.drop(["sample"], axis=1, inplace=True)
    return cnvFeatures

# DNA SNVs cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'freebayes_SNV_'
def cleanSnv(snvCleaned):
    import re
    snvCleaned.fillna(0, downcast='infer', inplace=True)
    snvCleaned["Biobank_Number"] = snvCleaned["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    snvCleaned["Biobank_Number"] = snvCleaned["Biobank_Number"].str.upper()
    snvCleaned.drop(["sample"], axis=1, inplace=True)
    return snvCleaned

# DNA Indels cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'pindel_INDEL_'
def cleanIndel(indelsCleaned):
    import re
    indelsCleaned.fillna(0, downcast='infer', inplace=True)
    indelsCleaned["Biobank_Number"] = indelsCleaned["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    indelsCleaned["Biobank_Number"] = indelsCleaned["Biobank_Number"].str.upper()
    indelsCleaned.drop(["sample"], axis=1, inplace=True)
    return indelsCleaned

# RNA Fusions cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'AF4_'
def cleanFusion(fusionFeatures):
    import re
    fusionFeatures.fillna(0, downcast='infer', inplace=True)
    fusionFeatures["Biobank_Number"] = fusionFeatures["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    fusionFeatures["Biobank_Number"] = fusionFeatures["Biobank_Number"].str.upper()
    fusionFeatures.drop(["sample"], axis=1, inplace=True)
    return fusionFeatures


# +
clinicalFeatures = cleanClinical(clinicalFeatures)
#labels = cleanLabel(labels)
snvCleaned = cleanSnv(snvRaw)
cnvCleaned = cleanCnv(cnvRaw)
indelCleaned = cleanIndel(indelRaw)
fusionCleaned = cleanFusion(fusionRaw)
rnaExprCleaned = cleanRNASeq(rnaGeneExprRaw, rnaGeneExprDiffExpr)
plasmaProteinCleaned = cleanPlasmaProtein(plasmaProteinRaw)
tissueProteinCleaned = cleanTissueProtein(tissueProteinRaw)
plasmaLipidCleaned = cleanPlasmaLipid(lipidRaw)
pathologyCleaned = cleanPathology(pathologyRaw)

mergedDataset = clinicalFeatures.merge(snvCleaned, how='left')
mergedDataset = mergedDataset.merge(cnvCleaned, how='left')
mergedDataset = mergedDataset.merge(indelCleaned, how='left')
mergedDataset = mergedDataset.merge(fusionCleaned, how='left')
mergedDataset = mergedDataset.merge(rnaExprCleaned, how='left')
mergedDataset = mergedDataset.merge(plasmaProteinCleaned, how='left')
mergedDataset = mergedDataset.merge(tissueProteinCleaned, how='left')
mergedDataset = mergedDataset.merge(plasmaLipidCleaned, how='left')
mergedDataset = mergedDataset.merge(pathologyCleaned, how='left')
mergedDataset.shape

# +
# Extract KRAS wild-type vs mutated sub-populations for SNV, CNV, INDELS & rna_expr
kras_columns = [c for c in mergedDataset.columns.values if "KRAS" in c]
for column in kras_columns:
    mergedDataset[column + "_mutated"] = mergedDataset[column].map(lambda x: 1 if pd.notnull(x) and x != 0 else x)
    
[c for c in mergedDataset.columns.values if "_mutated" in c]
# -

import plotly_express as px
px.histogram(mergedDataset, "CNV_KRAS")

# Save featurized, multi-omic dataset
OUTPUT_DIR = "./molecular_twin_pilot/outputs/multiomic"
mergedDataset.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_multiomic_dataset_with_kras_mutation.csv"), index=False)

# +
# Output all input feature correlations with each study objective.
import pandas as pd
import numpy as np
from scipy import stats

label_columns = ["label_patient_survival", "label_recurrence"]

multiomic_feature_corr = pd.DataFrame()
for feature in mergedDataset.columns:
    if feature not in (label_columns + ["Biobank_Number"]):
        for label in label_columns:
            df_clean = mergedDataset[[feature, label]].dropna()
            print(f"feature: {feature}")
            print(f"label: {label}")
            pearson_corr = stats.pearsonr(df_clean[feature], df_clean[label])
            multiomic_feature_corr.loc[feature, label + "_pearson_rho"] = pearson_corr[0]
            multiomic_feature_corr.loc[feature, label + "_pearson_pval"] = pearson_corr[1]
        
            spearman_corr = stats.spearmanr(df_clean[feature], df_clean[label])
            multiomic_feature_corr.loc[feature, label + "_spearman_rho"] = spearman_corr[0]
            multiomic_feature_corr.loc[feature, label + "_spearman_pval"] = spearman_corr[1]
multiomic_feature_corr.to_csv(os.path.join(OUTPUT_DIR, "feature_correlation_with_objective_labels.csv"), index=True)
# -

# Generate box plots for each of the input columns, for each label.
from matplotlib.backends.backend_pdf import PdfPages
for label in label_columns:
    feature_box_plots = PdfPages(f'/Users/onikolic/Downloads/box_plots_for_input_features_for_{label}.pdf')
    for gene in mergedDataset.columns:
        ax = mergedDataset.boxplot(column=gene, by=label, figsize=(8,4))
        plt.title('')
        plt.suptitle('')
        ax.set_title(f'{gene} for {label}');
        ax = plt.savefig(feature_box_plots, format='pdf')
    feature_box_plots.close()



