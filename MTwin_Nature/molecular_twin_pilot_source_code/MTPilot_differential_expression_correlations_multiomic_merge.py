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
# %load_ext autoreload
# %autoreload 2

# +
from IPython.core.display import display, HTML
from IPython.display import IFrame

import functools
import numpy as np
import os, datetime
import re
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow_data_validation as tfdv

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# -

import pandas as pd
BASE_DIR = "./molecular_twin_pilot/proteomics/"
plasmaProteinRaw = pd.read_csv(BASE_DIR + "MTPilot_Proteomics_Plasma_combined.tsv", delimiter='\t')
tissueProteinRaw = pd.read_csv(BASE_DIR + "TissueProteinLevel_AllPatientSamples.csv")
clinicalLabels = pd.read_csv("./molecular_twin_pilot/outputs/clinical_labels.csv")

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
#plasmaProtein_Tumor_low_var = pruneLowVarianceFeatures(plasmaProtein_Tumor_dense, ['Biobank_Number', 'Subject', 'Group'], threshold=0.2)
#plasmaProtein_Normal_low_var = pruneLowVarianceFeatures(plasmaProtein_Normal_dense, ['Biobank_Number', 'Subject', 'Group'], threshold=0.2)
plasmaProtein_Tumor_low_var = plasmaProtein_Tumor_dense
plasmaProtein_Normal_low_var = plasmaProtein_Normal_dense

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
tfdv.visualize_statistics(lhs_statistics=plasmaProteinNormal_statistics, lhs_name='Plasma Normal Protein', rhs_statistics=plasmaProteinTumor_statistics, rhs_name='Plasma Tumor Protein')

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
print("final:")
print(prunedTissueTumorProteins['P12107_COBA1'])
# -

tissueProteinTumor.shape

tissueProteinNormal.shape

tissueProteinFeatures.shape

# Look at raw distribution tfdv.stats on columns for Tumor vs Normal
tissueProteinTumor_statistics = tfdv.generate_statistics_from_dataframe(tissueProteinTumor)
tissueProteinNormal_statistics = tfdv.generate_statistics_from_dataframe(tissueProteinNormal)
tfdv.visualize_statistics(lhs_statistics=tissueProteinNormal_statistics, lhs_name='Tissue Normal Protein', rhs_statistics=tissueProteinTumor_statistics, rhs_name='Tissue Tumor Protein')

# Plot distribution similarity measure and p-values (for either independent Mann-Whitney U-test or dependent Wilcoxon signed-rank test)
tissue_protein_distribution_similarity_test_measure = tissue_protein_distribution_similarity_test.iloc[0].values
tissue_protein_distribution_similarity_test_measure.sort
plt.hist(tissue_protein_distribution_similarity_test_measure, 20, facecolor='blue', alpha=0.5)

# Plot Wilcoxon Signed Rank Test measure and p-value
tissue_protein_distribution_similarity_test_pvalue = tissue_protein_distribution_similarity_test.iloc[1].values
tissue_protein_distribution_similarity_test_pvalue.sort
plt.hist(tissue_protein_distribution_similarity_test_pvalue, 5, facecolor='blue', alpha=0.5, density=True)

merged_clinical_protein_types.head(20)

gene_raw_tumor.head(20)

# +
# Investigate tissue EMT proteins
from matplotlib.backends.backend_pdf import PdfPages
ROOT_DIR = "./molecular_twin_pilot/"
EMT_gene_list = ["O00391_QSOX1", "P02452_CO1A1", "P02461_CO3A1", "P05997_CO5A2", "P08123_CO1A2", "P12107_COBA1", "P20908_CO5A1",
                 "P21333_FLNA", "P21980_TGM2", "P25067_CO8A2", "P35052_GPC1", "P35555_FBN1", "P49747_COMP", "Q13361_MFAP5",
                 "Q14195_DPYL3", "Q16787_LAMA3", "Q6FHJ7_SFRP4", "Q9NR99_MXRA5"]
EMT_gene_plots = PdfPages(f'{OUTPUT_DIR}/TissueProtein_EMT_plots.pdf')

# Read in tissue protein types
tissue_protein_types = pd.read_csv("{OUTPUT}/tissue_protein_types.csv")

# Read and clean clinical outcomes
clinicalOutcomes = pd.read_csv(os.path.join(ROOT_DIR, "clinical", "MTPilot_Final_Outcomes.csv"))
label_columns = ['label_deceased', 'label_recurred']
label_columns_with_biobank = label_columns + ['Biobank_Number']

clinicalOutcomes.rename(columns={"biobank_number": "Biobank_Number"}, inplace=True)
clinicalOutcomes['label_deceased'] = clinicalOutcomes['vital_status'].map({'ALIVE': 0, 'DEAD': 1})
clinicalOutcomes['label_recurred'] = clinicalOutcomes['reason_of_recurrence'].map({'NaN': 0, 'Death': 1, 'Cancer': 1})
clinicalOutcomes[label_columns] = clinicalOutcomes[label_columns].apply(pd.to_numeric, errors='coerce', axis=1)
clinicalOutcomes['label_recurred'].fillna(0, inplace=True)
clinicalOutcomes = clinicalOutcomes[label_columns_with_biobank]

merged_clinical_protein_types = clinicalOutcomes.merge(tissue_protein_types, how='left')
merged_clinical_protein_types['Survival'] = merged_clinical_protein_types['label_deceased'].map({0: 'Alive', 1: 'Dead'})
merged_clinical_protein_types['Recurrence'] = merged_clinical_protein_types['label_recurred'].map({0: 'NonRecurred', 1: 'Recurred'})
#merged_clinical_protein_types.set_index('Biobank_Number', inplace=True)

for gene in EMT_gene_list:
    gene_raw_normal = pd.DataFrame()
    gene_raw_normal[gene] = tissueProteinNormal[gene]
    gene_raw_normal['Sample Type'] = 'NonTumor'
    gene_raw_tumor = pd.DataFrame()
    gene_raw_tumor[gene] = tissueProteinTumor[gene]
    gene_raw_tumor.reset_index(inplace=True)
    gene_raw_tumor.rename(columns={"index": "Biobank_Number"}, inplace=True)
    gene_raw_tumor = gene_raw_tumor.merge(merged_clinical_protein_types, how='left')
    gene_raw_tumor['Sample Type'] = gene_raw_tumor['tissue_type']
    gene_raw = pd.concat([gene_raw_normal, gene_raw_tumor])

    gene_log_normal = pd.DataFrame()
    gene_log_normal[gene] = tissueProteinNormal_logNorm[gene]
    gene_log_normal['Sample Type'] = 'NonTumor'
    gene_log_tumor = pd.DataFrame()
    gene_log_tumor[gene] = tissueProteinTumor_logNorm[gene]
    gene_log_tumor.reset_index(inplace=True)
    gene_log_tumor.rename(columns={"index": "Biobank_Number"}, inplace=True)
    gene_log_tumor = gene_log_tumor.merge(merged_clinical_protein_types, how='left')
    gene_log_tumor['Sample Type'] = gene_log_tumor['tissue_type']
    gene_log = pd.concat([gene_log_normal, gene_log_tumor])
    
    # Generate raw per-gene box plots between Tumor, T_NoT, NonTumor
    ax = gene_raw.boxplot(column=gene, by='Sample Type', figsize=(12,6))
    plt.title('')
    plt.suptitle('')
    ax.set_title(f'Raw {gene}');
    ax = plt.savefig(EMT_gene_plots, format='pdf')
    
    # Generate log-normalized per-gene box plots between Tumor, T_NoT, NonTumor
    ax = gene_log.boxplot(column=gene, by='Sample Type', figsize=(12,6))
    plt.title('')
    plt.suptitle('')
    ax.set_title(f'Log Norm {gene}');
    ax = plt.savefig(EMT_gene_plots, format='pdf')
    
    # Generate survival per-gene box plots for Tumor + T_NoT
    ax = gene_raw_tumor.boxplot(column=gene, by='Survival', figsize=(12,6))
    plt.title('')
    plt.suptitle('')
    ax.set_title(f'Survival for Tumor & T_NoT {gene}');
    ax = plt.savefig(EMT_gene_plots, format='pdf')
    
    # Generate survival per-gene box plots for Tumor
    ax = gene_raw_tumor[gene_raw_tumor["Sample Type"] == "Tumor"].boxplot(column=gene, by='Survival', figsize=(12,6))
    plt.title('')
    plt.suptitle('')
    ax.set_title(f'Survival for Tumor only {gene}');
    ax = plt.savefig(EMT_gene_plots, format='pdf')
EMT_gene_plots.close()

# +
# Merge raw Tumor/Normal COBA1 gene
coba1_tumor = pd.DataFrame()
coba1_tumor['COBA1_gene'] = tissueProteinTumor["P12107_COBA1"]
coba1_tumor['type'] = 'T'
coba1_normal = pd.DataFrame()
coba1_normal['COBA1_gene'] = tissueProteinNormal["P12107_COBA1"]
coba1_normal['type'] = 'N'

merged_coba1 = pd.concat([coba1_tumor, coba1_normal])
# -

merged_coba1

# Generate box plots for each of the input columns, for each label.
from matplotlib.backends.backend_pdf import PdfPages
for label in ['type']:
    feature_box_plots = PdfPages(f'{OUTPUT_DIR}/box_plots_for_raw_col11a1.pdf')
    for gene in ['COBA1_gene']:
        ax = merged_coba1.boxplot(column=gene, by=label, figsize=(8,4))
        plt.title('')
        plt.suptitle('')
        ax.set_title(f'input {gene}');
        ax = plt.savefig(feature_box_plots, format='pdf')
    feature_box_plots.close()

# +
# Merge log normalized tumor/normal coBa1 gene
coba1_tumor = pd.DataFrame()
coba1_tumor['COBA1_gene'] = tissueProteinTumor_logNorm["P12107_COBA1"]
coba1_tumor['type'] = 'T'
coba1_normal = pd.DataFrame()
coba1_normal['COBA1_gene'] = tissueProteinNormal_logNorm["P12107_COBA1"]
coba1_normal['type'] = 'N'

merged_coba1 = pd.concat([coba1_tumor, coba1_normal])
# -

# Generate box plots for each of the input columns, for each label.
from matplotlib.backends.backend_pdf import PdfPages
for label in ['type']:
    feature_box_plots = PdfPages(f'{OUTPUT_DIR}/box_plots_for_lognorm_col11a1.pdf')
    for gene in ['COBA1_gene']:
        ax = merged_coba1.boxplot(column=gene, by=label, figsize=(8,4))
        plt.title('')
        plt.suptitle('')
        ax.set_title(f'log norm {gene}');
        ax = plt.savefig(feature_box_plots, format='pdf')
    feature_box_plots.close()



raw_coba1_df = tissueProteinTumor[["P12107_COBA1"]]
raw_coba1_df.head()

processed_coba1_df.head()

# +
clinicalFeatures = pd.read_csv(os.path.join(ROOT_DIR, "outputs/clinical", "early_stage_patients_clinical_features.csv"))
label_columns = ["label_patient_survival", "label_recurrence"]

labels_df = clinicalFeatures[label_columns + ["Biobank_Number"]]
labels_df['Biobank_Number'] = labels_df['Biobank_Number'].str.upper()
labels_df.set_index('Biobank_Number', inplace=True)

labels_df.to_csv(os.path.join("MTPilot_study_labels.csv"), index=True)

raw_coba1_df = tissueProteinTumor[["P12107_COBA1"]]
raw_coba1_df.rename(columns={"P12107_COBA1": "input_P12107_COBA1", }, inplace=True)

processed_coba1_df = tissueProteinNormal_logNorm[["P12107_COBA1"]]
processed_coba1_df.rename(columns={"P12107_COBA1": "lognorm_P12107_COBA1", }, inplace=True)

mergedCoba1_df = labels_df.join(raw_coba1_df, how='left')
mergedCoba1_df = mergedCoba1_df.join(processed_coba1_df, how='left')
# -

mergedCoba1_df.head(60)

# +
#prunedTissueTumorProteins[]
#label_columns = ["label_patient_survival", "label_recurrence"]

# Generate box plots for each of the input columns, for each label.
from matplotlib.backends.backend_pdf import PdfPages
for label in label_columns:
    feature_box_plots = PdfPages(f'{OUTPUT_DIR}/box_plots_for_col11a1_{label}.pdf')
    for gene in ['input_P12107_COBA1', 'lognorm_P12107_COBA1']:
        ax = mergedCoba1_df.boxplot(column=gene, by=label, figsize=(8,4))
        plt.title('')
        plt.suptitle('')
        ax.set_title(f'{gene}');
        ax = plt.savefig(feature_box_plots, format='pdf')
    feature_box_plots.close()
# -

# Save featurized, plasma proteomic dataset
OUTPUT_DIR = "./molecular_twin_pilot/outputs/clinical"
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
clinicalOutcomes = pd.read_csv(os.path.join(ROOT_DIR, "clinical", "MTPilot_Final_Outcomes.csv"))
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

clinicalOutcomes.head()

# +
label_columns = ['label_deceased', 'label_recurred', 'label_days_to_death', 'label_days_to_recurrence']
label_columns_with_biobank = label_columns + ['Biobank_Number']

clinicalOutcomes.rename(columns={"biobank_number": "Biobank_Number", "is_neoadjuvant_chemo": "received_neoadjuvant_chemo", "day_of_death": "label_days_to_death", "day_of_recurrence": "label_days_to_recurrence"}, inplace=True)
clinicalOutcomes['label_deceased'] = clinicalOutcomes['vital_status'].map({'ALIVE': 0, 'DEAD': 1})
clinicalOutcomes['label_recurred'] = clinicalOutcomes['reason_of_recurrence'].map({'NaN': 0, 'Death': 1, 'Cancer': 1})
clinicalOutcomes[label_columns] = clinicalOutcomes[label_columns].apply(pd.to_numeric, errors='coerce', axis=1)
clinicalOutcomes['label_days_to_death'].fillna(6*365, inplace=True)
clinicalOutcomes['label_days_to_death'] = clinicalOutcomes['label_days_to_death'].replace(np.nan, 6*365)
clinicalOutcomes['label_days_to_recurrence'].fillna(6*365, inplace=True)
clinicalOutcomes['label_days_to_recurrence'] = clinicalOutcomes['label_days_to_recurrence'].replace(np.nan, 6*365)
clinicalOutcomes['label_recurred'].fillna(0, inplace=True)

# +
# Compute overall MTPilot study Kaplan Meier curve
# #!pip install lifelines

from lifelines import KaplanMeierFitter

## Generate KM-Plot 
durations = clinicalOutcomes['label_days_to_death']
event_observed = clinicalOutcomes['label_deceased']

## create a kmf object
kmf = KaplanMeierFitter() 

## Fit the data into the model
kmf.fit(durations, event_observed,label='MTPilot Kaplan Meier Estimate')

## Create an estimate
kmf.plot(xlabel='days to death') ## ci_show is meant for Confidence interval, since our data set is too tiny, thus i am not showing it.
# -

durations_neoadjuvant = clinicalOutcomes.loc[clinicalOutcomes['received_neoadjuvant_chemo'] == False, 'label_days_to_death']
durations_neoadjuvant

# +
# Plot Kaplan Meier for neoadjuvant vs non-neoadjuvant chemo cohorts.
kmf1 = KaplanMeierFitter() ## instantiate the class to create an object

## Two Cohorts are compared. Cohort 1. Did not receive neo-adjuvant chemo, and Cohort 2. Received neo-adjuvant chemo.
groups = clinicalOutcomes['received_neoadjuvant_chemo']   
i1 = (groups == 'True')      ## group i1 , having the pandas series  for the 1st cohort
i2 = (groups == 'False')     ## group i2 , having the pandas series  for the 2nd cohort

durations_neoadjuvant = clinicalOutcomes.loc[clinicalOutcomes['received_neoadjuvant_chemo'] == True, 'label_days_to_death']
durations_no_neoadjuvant = clinicalOutcomes.loc[clinicalOutcomes['received_neoadjuvant_chemo'] == False, 'label_days_to_death']
event_observed_neoadjuvant = clinicalOutcomes.loc[clinicalOutcomes['received_neoadjuvant_chemo'] == True, 'label_deceased']
event_observed_no_neoadjuvant = clinicalOutcomes.loc[clinicalOutcomes['received_neoadjuvant_chemo'] == False, 'label_deceased']

## fit the model for 1st cohort
kmf1.fit(durations_neoadjuvant, event_observed_neoadjuvant, label='yes neoadjuvant chemo')
a1 = kmf1.plot()

## fit the model for 2nd cohort
kmf1.fit(durations_no_neoadjuvant, event_observed_no_neoadjuvant, label='no neoadjuvant chemo')
kmf1.plot(ax=a1, xlabel='days to death')

# +
# Generate box plot distributions for all Clinical features in the study.
from quickda.explore_data import *
from quickda.clean_data import *
from quickda.explore_numeric import *
from quickda.explore_categoric import *
from quickda.explore_numeric_categoric import *
from quickda.explore_time_series import *

#eda_num(clinicalFeatures)
# -

# Generate categorical distribution plots for categorical variables.
eda_cat(clinicalFeatures, x='label_recurrence')

# Generate categorical distribution plots for categorical variables.
eda_cat(clinicalFeatures, x='clinical_Chemotherapy_Summary_ord')

snvRaw.shape

cnvRaw.shape

indelRaw.shape

fusionRaw.shape

rnaGeneExprDiffExpr.shape

plasmaProteinRaw.shape

tissueProteinRaw.shape

lipidRaw.shape

pathologyRaw.shape


# +
# Study Labels cleaning: 1) convert NAs to 0; 2) Columns already prefixed with 'label_'
def cleanLabel(labels):
    import re
    labels.fillna(0, downcast='infer', inplace=True)
    #labels["Biobank_Number"] = labels["Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    #labels["Biobank_Number"] = labels["Biobank_Number"].str.upper()
    #labels["Biobank_Number"] = labels["Biobank_Number"].astype(str)
    return labels[label_columns_with_biobank]

# Clinical features cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'clinical_'
def cleanClinical(clinicalFeatures):
    import re
    clinicalFeatures.fillna(0, downcast='infer', inplace=True)
    #clinicalFeatures = clinicalFeatures.add_prefix('clinical_')
    #clinicalFeatures["Biobank_Number"] = clinicalFeatures["clinical_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
    clinicalFeatures["Biobank_Number"] = clinicalFeatures["Biobank_Number"].str.upper()
    #clinicalFeatures.drop(["clinical_Biobank_Number"], axis=1, inplace=True)
    clinicalFeatures.drop(["label_patient_survival", "label_recurrence"], axis=1, inplace=True)
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


# -

# Clean and pre-process all single-omic sources.
clinicalFeatures = cleanClinical(clinicalFeatures)
labels = cleanLabel(clinicalOutcomes)
snvCleaned = cleanSnv(snvRaw)
cnvCleaned = cleanCnv(cnvRaw)
indelCleaned = cleanIndel(indelRaw)
fusionCleaned = cleanFusion(fusionRaw)
rnaExprCleaned = cleanRNASeq(rnaGeneExprRaw, rnaGeneExprDiffExpr)
plasmaProteinCleaned = cleanPlasmaProtein(plasmaProteinRaw)
tissueProteinCleaned = cleanTissueProtein(tissueProteinRaw)
plasmaLipidCleaned = cleanPlasmaLipid(lipidRaw)
pathologyCleaned = cleanPathology(pathologyRaw)

labels['label_deceased'].value_counts()

# Merge all single-omics sources.
mergedDataset1 = clinicalFeatures.merge(labels, how='left')
mergedDataset2 = mergedDataset1.merge(snvCleaned, how='left')
mergedDataset3 = mergedDataset2.merge(cnvCleaned, how='left')
mergedDataset4 = mergedDataset3.merge(indelCleaned, how='left')
mergedDataset5 = mergedDataset4.merge(fusionCleaned, how='left')
mergedDataset6 = mergedDataset5.merge(rnaExprCleaned, how='left')
mergedDataset7 = mergedDataset6.merge(plasmaProteinCleaned, how='left')
mergedDataset8 = mergedDataset7.merge(tissueProteinCleaned, how='left')
mergedDataset9 = mergedDataset8.merge(plasmaLipidCleaned, how='left')
mergedDataset10 = mergedDataset9.merge(pathologyCleaned, how='left')
mergedDataset10.shape

clinicalFeatures.columns

clinicalFeatures["clinical_Sex_ord"].value_counts()

mergedDataset10[["clinical_Age_at_Diagnosis", "clinical_Sex_ord", "clinical_BMI", "label_deceased", "label_recurred"]]

# Generate categorical distribution plots for key study variables {age, sex, BMI, Stage, TumorSize, Outcome, Recurrence}
reportingDataset["Age at Diagnosis", "Sex", "BMI", "Deceased", "Recurred"] = mergedDataset10[["clinical_Age_at_Diagnosis", "clinical_Sex_ord", "clinical_BMI", "label_deceased", "label_recurred"]]
reportingDataset[["Deceased", "Recurred"]] = reportingDataset[["Deceased", "Recurred"]].astype(bool)

reportingDataset["Deceased"].value_counts().plot(title="Deceased", kind='bar')

reportingDataset["Deceased"].value_counts()

reportingDataset["Recurred"].value_counts().plot(title="Recurred", kind='bar')

#reportingDataset.rename(columns={"clinical_Age_at_Diagnosis": "Age at Diagnosis", "clinical_Sex_ord":"Sex", "clinical_BMI":"BMI", "label_deceased":"Deceased", "label_recurred":"Recurred"}, inplace=True)
reportingDataset.hist(figsize=(18,12))

# +
# Spot check each omics source distributions.
import tensorflow_data_validation as tfdv

# Look at the CEL distribution tfdv.stats on Black vs SWOG columns for each gene.
singleomicData_statistics = tfdv.generate_statistics_from_dataframe(pathologyCleaned)
tfdv.visualize_statistics(singleomicData_statistics)

# +
# Transform multi-omic data (via Box-Cox or log/exp transforms)
#from skew_autotransform import skew_autotransform

#logTransformedDF = skew_autotransform(mergedDataset.copy(deep=True), plot = False, exp = True, threshold = 0.5, exclude = label_columns_with_biobank)
#boxCoxTransformedDF = skew_autotransform(mergedDataset.copy(deep=True), plot = False, exp = False, threshold = 0.5, exclude = label_columns_with_biobank)
#logTransformedDF.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_multiomic_dataset_log_transformed.csv"), index=False)
#boxCoxTransformedDF.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_multiomic_dataset_box_cox_transformed.csv"), index=False)
# -

# Save featurized, multi-omic dataset
OUTPUT_DIR = "/molecular_twin_pilot/outputs/multiomic"
mergedDataset10.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_multiomic_dataset.csv"), index=False)
clinicalOutcomes.to_csv(os.path.join(OUTPUT_DIR, "mtpilot_study_labels.csv"), index=False)

# +
import os

def generate_tf_embedding_projector_dataset(dataset):
    # Create TF Embedding Projector metadata dataset, with human readable clinical columns.
    metadata_columns = dataset.columns[dataset.columns.str.startswith("clinical_")].tolist() + \
                       dataset.columns[dataset.columns.str.startswith("label_")].tolist() + ["Biobank_Number"]
    metadata_dataset = dataset.loc[:, metadata_columns]
    metadata_tsv = metadata_dataset.to_csv(os.path.join(OUTPUT_DIR, 'CS_MT_embedding_projector_multiomic_metadata.tsv'), sep='\t', index=False, header=True)

    # Create TF Embedding Projector tensor dataset for multi-omic dataset.
    multiomic_train_dataset = dataset.select_dtypes(['number'])
    multiomic_train_dataset = multiomic_train_dataset[multiomic_train_dataset.columns.drop(list(multiomic_train_dataset.filter(regex='label_')) + \
                                                                                           list(multiomic_train_dataset.filter(regex='clinical_')) + \
                                                                                           list(multiomic_train_dataset.filter(regex='surgery_embed_')) + \
                                                                                           list(multiomic_train_dataset.filter(regex='pathology_embed_')) + \
                                                                                           list(multiomic_train_dataset.filter(regex='chemotherapy_embed_')))]
    dataset_tsv = multiomic_train_dataset.to_csv(os.path.join(OUTPUT_DIR, 'CS_MT_embedding_projector_multiomic_dataset.tsv'), sep='\t', na_rep='0.0', index=False, header=False)

    # Create TF Embedding Projector tensor dataset for each single-omic dataset. # left out: "surgery_embed_", "pathology_embed_", "chemotherapy_embed_", 
    column_prefixes = ["clinical_", "plasma_protein_", "tissue_protein_", "plasma_lipid_", "pathology_NF", "CNV_", "freebayes_SNV_", "pindel_INDEL_", "AF4_", "rna_expr_"]
    for prefix in column_prefixes:
        singleomic_columns = dataset.columns[dataset.columns.str.contains(prefix)].tolist()
        singleomic_dataset = dataset.dropna(axis=0, how='all', subset=singleomic_columns)
        singleomic_metadata = singleomic_dataset.loc[:, metadata_columns]
        singleomic_train_dataset = singleomic_dataset.loc[:, singleomic_columns]
        singleomic_metadata_tsv = singleomic_metadata.to_csv(os.path.join(OUTPUT_DIR, f'CS_MT_embedding_projector_{prefix}metadata.tsv'), sep='\t', index=False, header=True)
        singleomic_dataset_tsv = singleomic_train_dataset.to_csv(os.path.join(OUTPUT_DIR, f'CS_MT_embedding_projector_{prefix}dataset.tsv'), sep='\t', na_rep='0.0', index=False, header=False)



# -

generate_tf_embedding_projector_dataset(mergedDataset10)

clinicalOutcomes.to_csv(OUTPUT_DIR, index=False)

# +
# Output all input feature correlations with each study objective.
import pandas as pd
import numpy as np
from scipy import stats

multiomic_feature_corr = pd.DataFrame()
for feature in mergedDataset.columns:
    if feature not in label_columns_with_biobank:
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

OUTPUT_DIR

# +
# Generate box plots for each of the input columns, for each label.
from matplotlib.backends.backend_pdf import PdfPages
label_columns_readable = ["deceased", "recurred"]
mergedDataset['deceased'] = mergedDataset['label_deceased']
mergedDataset['recurred'] = mergedDataset['label_recurred']

for label in label_columns_readable:
    for gene in list(set(mergedDataset.columns.tolist())-set(label_columns_with_biobank)):
    #for gene in mergedDataset.columns:
        print(f"mtpilot_box_plots_for_{gene}_for_{label}.png")
        feature_box_plots = PdfPages(f'{OUTPUT_DIR}/mtpilot_box_plots_for_{gene}_for_{label}.pdf')
        ax = mergedDataset.boxplot(column=gene, by=label, figsize=(16,8), fontsize=32, boxprops=dict(linewidth=3.5), whiskerprops=dict(linewidth=3.5))
        plt.title('')
        plt.suptitle('')
        ax.set_title(f'{gene} for {label}', fontsize=32);
        gene_file_name = re.sub('[^0-9a-zA-Z]+', '_', gene)
        ax = plt.savefig(f'{OUTPUT_DIR}/mtpilot_box_plots_for_{gene_file_name}_for_{label}.png', format='png')
    feature_box_plots.close()
# -

# Output top performing tissue_protein, rna_expr, snv, cnv, indel features for each label.
multiomicResults = pd.read_csv(os.path.join(ROOT_DIR, "outputs/multiomic", "multiomic_results_early_stage_patients_all_analytes.csv"))


# +
multiomicResults['feature_prefix'] = multiomicResults['feature_prefix'].astype(str)
multiomicResults.rename(columns={"test_loo_metric": "accuracy",}, inplace=True)
multiomicResults.drop(columns=['Unnamed: 0', ], inplace=True)
baseline_models = multiomicResults[multiomicResults['feature_prefix'].str.split(',').apply(len) == 2]
baseline_models['first_feature'] = baseline_models.feature_prefix.str.split(',').str[0].str[2:-1]
baseline_models['second_feature'] = baseline_models.feature_prefix.str.split(',').str[1].str[1:-2]

single_omics_baseline = baseline_models[baseline_models.second_feature == ""]
single_omics_baseline.drop(columns=['first_feature', 'second_feature'], inplace=True)
single_omics_baseline = single_omics_baseline.loc[:,~single_omics_baseline.columns.str.contains('top_10_feature')].sort_values(["target_label", "accuracy"], ascending=False)
# -

singleomic_baseline = single_omics_baseline[(single_omics_baseline.feature_prefix == "('AF4_',)") & (single_omics_baseline.target_label == "label_recurrence")].sort_values(["target_label", "accuracy"], ascending=False)
singleomic_baseline

baseline_models.loc[1423].head(60)

multiomicResults.loc[multiomicResults.groupby(["target_label"])["accuracy"].idxmax()]

multiomicResults.loc[multiomicResults.groupby(["target_label"])["F1"].idxmax()]

multiomicResultsStats = multiomicResults.loc[:,~multiomicResults.columns.str.contains('top_10_feature')]

pd.set_option('display.max_colwidth', -1)
#multiomicResultsStats.loc[multiomicResultsStats.groupby(["target_label"])["accuracy"].idxmax()]
multiomicResultsStats.loc[multiomicResultsStats.groupby(["target_label"])["accuracy"].nlargest(10).index.get_level_values(1)]

pd.set_option('display.max_colwidth', -1)
#multiomicResultsStats.loc[multiomicResultsStats.groupby(["target_label"])["F1"].idxmax()]
topMultiomicResultsStats = multiomicResultsStats.loc[multiomicResultsStats.groupby(["target_label"])["accuracy"].nlargest(10).index.get_level_values(1)]

topSurvivalMultiomicResultsStats = topMultiomicResultsStats[(topMultiomicResultsStats.target_label == "label_patient_survival")]
topSurvivalMultiomicResultsStats.to_csv(os.path.join(OUTPUT_DIR, "topMultiomicModelStats_patient_survival.csv"), index=False)

topRecurrenceMultiomicResultsStats = topMultiomicResultsStats[(topMultiomicResultsStats.target_label == "label_recurrence")]
topRecurrenceMultiomicResultsStats.to_csv(os.path.join(OUTPUT_DIR, "topMultiomicModelStats_recurrence.csv"), index=False)

multiomicResults.loc[355, multiomicResults.columns.str.contains('top_10_feature')].head(60)

