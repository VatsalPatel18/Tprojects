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
# TCGA Validation steps for MTPilot
# 1. Clean and normalize phenotypic data for TCGA and MTPilot
# 2. Ingest, join and normalize TCGA RNA, SNV, INDEL, CNV analyte features
# 3. Train on MTPilot, validate on TCGA for each single-omic and multi-omic analytes

# %load_ext autoreload
# %autoreload 2

# +
import boto3
import datetime
import io
import itertools
import numpy as np
import os
import pandas as pd
import s3fs
import tensorflow_data_validation as tfdv

from pyensembl import EnsemblRelease
from sklearn import preprocessing, svm
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

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:95% !important; }</style>"))
display(HTML("<style>div.output_scroll { height: 44em; }</style>"))
pd.options.display.float_format = "{:,.2f}".format
pd.set_option('display.max_colwidth', 0)

BASE_DIR = "./molecular_twin_pilot/"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
TCGA_INPUT_DIR = os.path.join(BASE_DIR, "tcga_validation")
TCGA_OUTPUT_DIR = os.path.join(OUTPUT_DIR, "tcga_validation")


# -

def cleanDataframe(cleaned_df):
    # Cleanup invalid characters in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(',', '')
    cleaned_df.columns = cleaned_df.columns.str.replace('(', '')
    cleaned_df.columns = cleaned_df.columns.str.replace(')', '')
    cleaned_df.columns = cleaned_df.columns.str.replace('|', '_')
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    cleaned_df.columns = cleaned_df.columns.str.replace('/', '_')
    # Remove errant commas and semi-colons within cells for csv parsing
    cleaned_df.replace(',', '', regex=True, inplace=True)
    cleaned_df.replace(';', '', regex=True, inplace=True)
    cleaned_df.replace("'--", '', regex=True, inplace=True)
    cleaned_df.replace('\([0-9]*\)', '', regex=True, inplace=True)
    cleaned_df.replace(r'^\s*$', np.nan, regex=True, inplace=True)
    # Drop empty rows and columns, fill empty cells with appropriate defaults.
    cleaned_df.dropna(axis='index', how='all', inplace=True)
    cleaned_df.dropna(axis='columns', how='all', inplace=True)
    cleaned_df.fillna(value="", inplace=True, downcast='infer')
    return cleaned_df


# +
# Harmonize clinical variables between TCGA and MTPilot datasets

# TCGA harmonization
tcgaClinical = pd.read_csv(os.path.join(TCGA_INPUT_DIR, "TCGA-PAAD.GDC_phenotype.tsv"), sep='\t')
tcgaClinicalCleaned = cleanDataframe(tcgaClinical)
tcgaClinicalCleaned['sample_id'] = tcgaClinicalCleaned['submitter_id.samples']
tcgaClinicalCleaned['label_deceased'] = tcgaClinicalCleaned['vital_status.demographic'].map({'Dead': 1, 'Alive': 0})
tcgaClinicalCleaned['clinical_Age_at_Diagnosis'] = tcgaClinicalCleaned['age_at_initial_pathologic_diagnosis'].astype(int)
tcgaClinicalCleaned['clinical_Sex'] = tcgaClinicalCleaned['gender.demographic'].map({'male': 0, 'female': 1})
tcgaClinicalCleaned['clinical_Race'] = tcgaClinicalCleaned['race.demographic'].map({'white': 0, 'asian': 1, 'black or african american': 2, 'not reported': 3})
tcgaClinicalCleaned['clinical_Ethnicity'] = tcgaClinicalCleaned['ethnicity.demographic'].map({'not hispanic or latino': 0, 'not reported': 1, 'hispanic or latino': 2})
tcgaClinicalCleaned['clinical_site_icd_10'] = tcgaClinicalCleaned['icd_10_code.diagnoses'].map({'C25.0': 0, 'C25.1': 1, 'C25.2': 2, 'C25.7': 3, 'C25.8': 4, 'C25.9': 4})
tcgaClinicalCleaned['clinical_morphology'] = tcgaClinicalCleaned['morphology.diagnoses'].map({'8020/3': 0, '8140/3': 1, '8246/3': 2, '8255/3': 3, '8480/3': 4, '8500/3': 5})
tcgaClinicalCleaned['clinical_path_stage_n'] = tcgaClinicalCleaned['pathologic_N'].map({'N0': 0, 'N1': 1, 'N1b': 1, 'NX': 2})
tcgaClinicalCleaned['clinical_path_stage_t'] = tcgaClinicalCleaned['pathologic_T'].map({'T1': 0, 'T2': 1, 'T3': 2, 'T4': 3, 'TX': 4})
tcgaClinicalCleaned['clinical_path_stage_tnm'] = tcgaClinicalCleaned['tumor_stage.diagnoses'].map({'stage i': 1, 'stage ia': 1, 'stage ib': 1, 'stage iia': 2, 'stage iib': 2, 'stage iii': 3, 'stage iv': 4, 'not reported': 5})
tcgaClinicalCleaned['clinical_alcohol_history'] = tcgaClinicalCleaned['alcoholic_exposure_category'].map({'': 0, 'None': 1, 'Occasional Drinker': 2, 'Social Drinker': 2, 'Weekly Drinker': 3, 'Daily Drinker': 3})
tcgaClinicalCleaned['clinical_tobacco_history'] = tcgaClinicalCleaned['tobacco_smoking_history'].map({'': 0, 1.0: 1, 2.0: 1, 3.0: 1, 4.0: 1, 5.0: 1})
tcgaClinicalCleaned['clinical_family_history_of_cancer'] = tcgaClinicalCleaned['family_history_of_cancer'].map({'NO': 0, '': 1, 'YES': 2})
tcgaClinicalCleaned['clinical_max_tumor_size_mm'] = pd.to_numeric(tcgaClinicalCleaned['maximum_tumor_dimension'], errors='coerce') * 10
tcgaClinicalCleaned['clinical_max_tumor_size_mm'].fillna(value=tcgaClinicalCleaned['clinical_max_tumor_size_mm'].mean(), inplace=True)
tcgaClinicalCleaned['clinical_grade'] = tcgaClinicalCleaned["neoplasm_histologic_grade"].map({'G1':0, 'G2': 1, 'G3': 2, 'G4': 3, 'GX': 4})
tcgaClinicalCleaned['clinical_patient_history_of_cancer'] = tcgaClinicalCleaned["prior_malignancy.diagnoses"].map({'no': 0, 'yes': 1})

# MTPilot harmonization
mtpilotClinical = pd.read_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_clinical_features_with_raw.csv"))
mtpilotClinical['sample_id'] = mtpilotClinical['Biobank_Number']
mtpilotClinical['label_deceased'] = mtpilotClinical['Vital_Status'].map({'dead': 1, 'alive': 0})
mtpilotClinical['clinical_Age_at_Diagnosis'] = mtpilotClinical['Age_at_Diagnosis'].astype(int)
mtpilotClinical['clinical_Sex'] = mtpilotClinical['Sex'].map({'1 (male)': 0, '2 (female)': 1})
mtpilotClinical['clinical_Race'] = mtpilotClinical['Race'].map({'white': 0, 'asian other/nos': 1, 'hawaiian':1, 'japanese': 1, 'chinese': 1, 'filipino': 1, 'vietnamese': 1, 'black': 2, 'other': 3})
mtpilotClinical['clinical_Ethnicity'] = mtpilotClinical['Spanish_Hispanic_Origin'].map({'non-spanish': 0, 'not reported': 1, 'south/central american': 2, 'cuban':2, 'mexican':2, 'spanish surname only':2, 'spanish nos': 2})
mtpilotClinical['clinical_site_icd_10'] = mtpilotClinical['Site_-_Primary_ICD-O-3'].map({'c250 (pancreas head)': 0, 'c251 (pancreas body)': 1, 'c252 (pancreas tail)': 2, 'c257 (pancreas other spec.)': 3, 'c258 (pancreas overlapping)': 4})
mtpilotClinical['clinical_morphology'] = mtpilotClinical['Histology_Behavior_ICD-O-3'].map({81403.0: 1, 85003.0: 5})
mtpilotClinical['clinical_path_stage_n'] = mtpilotClinical['TNM_Mixed_Stage_N_Code'].map({'p0': 0, 'pn0': 0, 'c0': 0, 'cn0': 0, 'p1': 1, 'pn1': 1, 'c1': 1, 'cn1': 1, 'pn2': 2})
mtpilotClinical['clinical_path_stage_t'] = mtpilotClinical['TNM_Mixed_Stage_T_Code'].map({'pt1a': 0, 'pt1b':0, 'pt1c': 0, 'ct1c':0, 'pt2': 1, 'p2': 1, 'c2': 1, 'ct2': 1, 'p3': 2, 'c3':2, 'pt3':2})
mtpilotClinical['clinical_path_stage_tnm'] = mtpilotClinical['TNM_Mixed_Stage'].map({'1a': 1, '1b': 1, '2a': 2, '2b': 2})
mtpilotClinical['clinical_alcohol_history'] = mtpilotClinical['Patient_History_Alcohol'].map({'no hx alcohol use': 0, 'none': 1, 'unknown use': 1, 'past hx not current': 2, 'current use': 3})
mtpilotClinical['clinical_tobacco_history'] = mtpilotClinical['Patient_History_Tobacco'].map({'none': 0, 'never used': 0, 'previous use': 1, 'cigarette smoker current': 1})
mtpilotClinical['clinical_family_history_of_cancer'] = mtpilotClinical['Family_history_1st_any_cancer'].map({'none': 0, 'unknown': 1, '1 relative': 2, '2 relatives': 2})
mtpilotClinical['clinical_max_tumor_size_mm'] = mtpilotClinical['Tumor_Size_Summary']
mtpilotClinical['clinical_grade'] = mtpilotClinical["Grade_Mixed"].map({'1 (well differentiated g1)':0, '2 (moderately differentiated g2)': 1, '3 (poorly differentiated g3)': 2})
mtpilotClinical['clinical_patient_history_of_cancer'] = np.where((mtpilotClinical['Patient_History_of_Cancer_Seq_1'] == 1.0) | (mtpilotClinical['Patient_History_of_Cancer_Seq_2'] == 2.0), 1, 0)

# Apply Inclusion/Exclusion criteria used for MTPilot to both datasets (include stage 1 & 2, include age >= 40)
tcgaClinicalCleaned = tcgaClinicalCleaned.loc[(tcgaClinicalCleaned["clinical_path_stage_tnm"] <= 2) & (tcgaClinicalCleaned['clinical_Age_at_Diagnosis'] >= 40)]
mtpilotClinical = mtpilotClinical.loc[(mtpilotClinical["clinical_path_stage_tnm"] <= 2) & (mtpilotClinical['clinical_Age_at_Diagnosis'] >= 40)]

# Retain sample_id, harmonized clinical variables and study label
tcgaClinicalCleaned = tcgaClinicalCleaned[['sample_id', 'label_deceased'] + tcgaClinicalCleaned.columns[tcgaClinicalCleaned.columns.str.startswith('clinical_')].tolist()]
mtpilotClinicalCleaned = mtpilotClinical[['sample_id', 'label_deceased'] + mtpilotClinical.columns[mtpilotClinical.columns.str.startswith('clinical_')].tolist()]

limit_to_parsimonious_features = False # If True, limit analysis only on the features selected by the MTPilot parsimonious model for DNA, RNA & Clinical, otherwise use all available features.
mtpilot_parsimonious_features = ['rna_expr_NIPAL2','clinical_max_tumor_size_mm','rna_expr_ABHD2','rna_expr_SORL1','CNV_MYB','rna_expr_MFN2','rna_expr_LINC01145','rna_expr_IGFBP3','rna_expr_PRKX','rna_expr_SLC40A1','CNV_HRAS','rna_expr_RNA5SP389','rna_expr_FBXW7','rna_expr_TIPARP','clinical_path_stage_n','CNV_CDKN1C','rna_expr_DTX3L','rna_expr_PARP14','rna_expr_STAT1','rna_expr_ZNF704','rna_expr_DDX60','rna_expr_KCNK1','CNV_GATA4','rna_expr_SH3PXD2B','freebayes_SNV_ERG','rna_expr_RGS5','freebayes_SNV_PCDH17','CNV_NT5C2','CNV_ERCC2','CNV_HIST1H1E','CNV_HIST1H3B','rna_expr_RNF103','rna_expr_CTSS','CNV_NUP98','rna_expr_CPD','CNV_SMC3','rna_expr_FBXO2','rna_expr_NBPF26','rna_expr_NPIPB3','rna_expr_TULP3','freebayes_SNV_PTPRD','CNV_HIST1H4E','freebayes_SNV_CUL4B','rna_expr_ATAD3A','rna_expr_DCN','CNV_ERCC1','freebayes_SNV_SMARCA4','rna_expr_GTF2IRD2B','rna_expr_ICAM1','rna_expr_DDX60L','CNV_TBC1D12','rna_expr_BCL9L','CNV_ERBB2','CNV_CDK12','CNV_SUFU','CNV_POLE','rna_expr_SYTL1','rna_expr_DFFA','rna_expr_SLFN11','CNV_GATA6','rna_expr_TLE3','rna_expr_GNAQ','rna_expr_ZNF141','rna_expr_SRSF4','clinical_family_history_of_cancer','CNV_HLA-DRB6','CNV_CBLC','CNV_FGF8','rna_expr_MGLL','rna_expr_VASP','clinical_tobacco_history','CNV_AKT1','rna_expr_ZDHHC7','CNV_VSIR','rna_expr_LRIG3','rna_expr_SERINC5','rna_expr_CASP10','rna_expr_URI1','rna_expr_SHROOM3','CNV_TCF7L2','freebayes_SNV_NGF','freebayes_SNV_PBRM1','freebayes_SNV_EPHB2','CNV_CD79A','CNV_FGF3','CNV_ARHGAP35','CNV_NRG1','rna_expr_REG3G','rna_expr_UPF2','freebayes_SNV_FCGR2A','freebayes_SNV_HIST1H1E','freebayes_SNV_GATA4','CNV_FGF4','rna_expr_ZCCHC24','CNV_MIB1','rna_expr_CACNA1D','rna_expr_MLKL','CNV_SMARCA1','rna_expr_PLS1','rna_expr_MS4A7','rna_expr_BTG2','rna_expr_PARP9','rna_expr_SCTR','rna_expr_BTN2A2','clinical_Sex','CNV_GNA13','rna_expr_ST3GAL6','rna_expr_EHBP1L1','rna_expr_SMC4','rna_expr_XAF1','freebayes_SNV_SMAD4','rna_expr_C1GALT1','rna_expr_PRICKLE2','rna_expr_TOMM7','rna_expr_KIF13B','rna_expr_CPM','rna_expr_OPHN1','CNV_ALK','freebayes_SNV_BCR','rna_expr_GPR137B','rna_expr_CBS','rna_expr_PIK3AP1','rna_expr_MDM2','rna_expr_PPP1R15A']
# -

# +
# Ingest, join and normalize TCGA SNV and INDEL data
tcgaDNA = pd.read_csv(os.path.join(TCGA_INPUT_DIR, "TCGA-PAAD.varscan2_snv.tsv"), sep='\t')

# Filter out low quality and lowest 1-%tile allele frequencies as noise.
tcgaDNA_filtered = tcgaDNA[tcgaDNA['filter'] == 'PASS']
tcgaDNA_filtered = tcgaDNA_filtered[tcgaDNA_filtered['dna_vaf'] > tcgaDNA_filtered['dna_vaf'].quantile(.01)]
tcgaDNA_filtered['value'] = 1

# Extract and reshape SNVs
tcgaSNV = tcgaDNA_filtered[tcgaDNA_filtered['effect'].isin(['missense_variant', 'synonymous_variant'])]
tcgaSNV_processed_df = tcgaSNV[["Sample_ID", "gene", "value"]].pivot_table(index="gene", columns="Sample_ID", values="value").T.reset_index()
tcgaSNV_processed_df.columns = ["freebayes_SNV_" + c for c in tcgaSNV_processed_df.columns]
tcgaSNV_processed_df.rename(columns={'freebayes_SNV_Sample_ID': 'sample_id'}, inplace=True)

# Extract and reshape INDELs
tcgaINDEL = tcgaDNA_filtered[~tcgaDNA_filtered['effect'].isin(['missense_variant', 'synonymous_variant'])]
tcgaINDEL_processed_df = tcgaINDEL[["Sample_ID", "gene", "value"]].pivot_table(index="gene", columns="Sample_ID", values="value").T.reset_index()
tcgaINDEL_processed_df.columns = ["pindel_INDEL_" + c for c in tcgaINDEL_processed_df.columns]
tcgaINDEL_processed_df.rename(columns={'pindel_INDEL_Sample_ID': 'sample_id'}, inplace=True)

# +
# Ingest, normalize and join TCGA CNV data
BED_FILE = os.path.join(BASE_DIR, "clinical/xTv4_panel_probe_gene_targets.bed")

def normalize_cnv(raw_cnv_df, bed_file=BED_FILE):
    bed_file_df = pd.read_csv(bed_file, sep="\t", names=["Chrom", "Start", "End", "gene"])
    # Only retain gain or loss CNVs, filter out neutral CNVs within -0.3 to 0.3
    filtered_cnv_df = raw_cnv_df[(raw_cnv_df["value"] > 0.3) | (raw_cnv_df["value"] < -0.3)]

    # Merge bed file regions
    def merge_bed_regions(gene_df, axis=None):
        min_start = gene_df.Start.min()
        max_end = gene_df.End.max()
        last_entry = gene_df.iloc[-1]
        last_entry.Start = min_start
        last_entry.End = max_end
        return last_entry
    merged_bed_file = bed_file_df.groupby("gene").agg(axis="columns", func=merge_bed_regions).reset_index().sort_values(['Chrom', 'Start'])

    merged_cnv_bed_df = filtered_cnv_df.merge(merged_bed_file, how='left', on ='Chrom', suffixes=("_cnv", "_bed"))

    # Filter for partial overlap
    partial_overlap_df = merged_cnv_bed_df[((merged_cnv_bed_df.Start_cnv <= merged_cnv_bed_df.Start_bed) & (merged_cnv_bed_df.Start_bed <= merged_cnv_bed_df.End_cnv)) | ((
        merged_cnv_bed_df.Start_cnv <= merged_cnv_bed_df.End_bed)  & (merged_cnv_bed_df.End_bed <= merged_cnv_bed_df.End_cnv))]

    processed_cnv_df = partial_overlap_df[["sample", "gene", "value"]].pivot_table(index="gene", columns="sample", values="value").T.reset_index()
    processed_cnv_df.columns = ["CNV_" + c for c in processed_cnv_df.columns]
    processed_cnv_df.rename(columns={'CNV_sample': 'sample_id'}, inplace=True)
    return processed_cnv_df

tcgaCNV = pd.read_csv(os.path.join(TCGA_INPUT_DIR, "TCGA-PAAD.cnv.tsv"), sep='\t')
tcgaCNV_processed_df = normalize_cnv(tcgaCNV)

# +
# Ingest, normalize and join TCGA RNA data

# Load TCGA RNA sample sheet for joining individual RNA files with samples
tcgaRNA_sample_sheet = pd.read_csv(os.path.join(TCGA_INPUT_DIR, "TCGA-PAAD.rna.gdc_sample_sheet.tsv"), sep='\t')
tcgaRNA_sample_sheet_filtered = tcgaRNA_sample_sheet[tcgaRNA_sample_sheet['Data Category'] == 'Transcriptome Profiling'][['File ID', 'Sample ID']]

S3_BUCKET = "betteromics-data-platform" # Set S3 bucket for input data.
S3_TCGA_RNA_PREFIX = "molecular_twin_pilot/tcga_validation/RNA/" # Input data file path
boto3_session = boto3.Session(profile_name="legacy")
s3 = boto3.resource('s3')
bucket = s3.Bucket(S3_BUCKET)
prefix_objs = bucket.objects.filter(Prefix=S3_TCGA_RNA_PREFIX)

# Ingest each TCGA RNA star align gene counts file, extract and transpose genes & tpms
def processTCGA_RNA_sample(file_id, rawRNA_df):
    RNA_tpm_df = rawRNA_df[~rawRNA_df['gene_name'].isna()][['gene_name', 'tpm_unstranded']]
    RNA_tpm_df['File ID'] = file_id
    RNA_transposed_df = RNA_tpm_df.pivot_table(index="gene_name", columns="File ID", values="tpm_unstranded").T.reset_index()
    RNA_transposed_df.columns = ["rna_expr_" + c for c in RNA_transposed_df.columns]
    RNA_transposed_df.rename(columns={'rna_expr_File ID': 'File ID'}, inplace=True)
    return RNA_transposed_df


tcgaRNA_df = pd.DataFrame()
for obj in prefix_objs:
    key = obj.key
    key_parts = key.split('/')       
    file_name = key_parts[-1]
    file_id = key_parts[-2]
    
    rna_sample_df = pd.read_csv(os.path.join("s3://", S3_BUCKET, key), delimiter='\t', skiprows=0, header=1)
    rna_sample_transposed_df = processTCGA_RNA_sample(file_id, rna_sample_df)
    tcgaRNA_df = tcgaRNA_df.append(rna_sample_transposed_df)

# Join RNA data with rna sample_sheet to map from file_id to sample_id
tcgaRNA_processed_df = tcgaRNA_df.merge(tcgaRNA_sample_sheet_filtered, on='File ID', how='inner')
tcgaRNA_processed_df.rename(columns={'Sample ID': 'sample_id'}, inplace=True)
tcgaRNA_processed_df.drop(columns="File ID", inplace=True)


# +
def subset_df_by_prefixes(df, prefixes):
    column_list = []
    for prefix in prefixes:
        column_list += df.columns[df.columns.str.startswith(prefix)].tolist()
    return df[set(column_list)]


def eliminate_sparsity(input_dataframe):
    dense_df = input_dataframe.dropna(axis='index', how='all')
    dense_df.dropna(axis='columns', how='all', inplace=True)
    dense_df.fillna(value=0, downcast='infer', inplace=True)
    return dense_df


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

# Recursive Feature Elimination with Random Forest.
def RFE_RF_Model():
    return make_pipeline(
        StandardScaler(),
        RFE(RandomForestClassifier(), step=0.05)
    )


# +
# Merge TCGA analyte sources into multiomic dataset
tcgaMerged_clinical_cnv = tcgaClinicalCleaned.merge(tcgaCNV_processed_df, on='sample_id', how='inner')
tcgaMerged_clinical_cnv_rna = tcgaMerged_clinical_cnv.merge(tcgaRNA_processed_df, on='sample_id', how='inner')
tcgaMerged_clinical_cnv_rna_snv = tcgaMerged_clinical_cnv_rna.merge(tcgaSNV_processed_df, on='sample_id', how='inner')
tcgaMerged_clinical_cnv_rna_snv_indel = tcgaMerged_clinical_cnv_rna_snv.merge(tcgaINDEL_processed_df, on='sample_id', how='inner')

# Drop empty rows, columns and duplicated columns
tcgaMerged = eliminate_sparsity(tcgaMerged_clinical_cnv_rna_snv_indel)
tcgaMerged = tcgaMerged.loc[:,~tcgaMerged.columns.duplicated()]
tcgaMerged.to_csv(f"./molecular_twin_pilot/outputs/multiomic/tcga_harmonized_validation_dataset.csv", index=False)

# +
# Prepare MTPilot dataset
mtpilotDataset = pd.read_csv(os.path.join(OUTPUT_DIR, "multiomic/early_stage_patients_multiomic_dataset.csv"))

# Replace MTPilot clinical variables with the ones harmonized with TCGA data.
mtpilotDataset = mtpilotDataset.loc[:, ~mtpilotDataset.columns.str.startswith('clinical_')]
mtpilotDataset.rename(columns={'Biobank_Number': 'sample_id'}, inplace=True)
mtpilotDataset.drop(columns=({'label_deceased'}), inplace=True)
mtpilotDataset['sample_id'] = mtpilotDataset['sample_id'].apply(lambda x: x.lower())

mtpilotMerged = mtpilotClinicalCleaned.merge(mtpilotDataset, on='sample_id', how='outer')

# Drop empty rows, columns and duplicated columns
mtpilotMerged = eliminate_sparsity(mtpilotMerged)
mtpilotMerged = mtpilotMerged.loc[:,~mtpilotMerged.columns.duplicated()]
mtpilotMerged.to_csv(f"./molecular_twin_pilot/outputs/multiomic/mtpilot_tcga_harmonized_dataset.csv", index=False)

# +
# Inspect RNA skew and normalization between TCGA and MTPilot
tcgaRNA_df = subset_df_by_prefixes(tcgaMerged, ('rna_expr_',))
mtpilotRNA_df = subset_df_by_prefixes(mtpilotMerged, ('rna_expr_',))

common_rna_features = set(np.intersect1d(mtpilotRNA_df.columns, tcgaRNA_df.columns)) 
tcgaRNA_common_df = tcgaRNA_df[common_rna_features]
mtpilotRNA_common_df = mtpilotRNA_df[common_rna_features]

tcga_rna_statistics = tfdv.generate_statistics_from_dataframe(tcgaRNA_common_df)
mtpilot_rna_statistics = tfdv.generate_statistics_from_dataframe(mtpilotRNA_common_df)

tfdv.visualize_statistics(lhs_statistics=tcga_rna_statistics, rhs_statistics=mtpilot_rna_statistics,
                          lhs_name='TCGA RNA', rhs_name='MTPilot RNA')
# -

# Create combination of analyte models, individual single-omic and multi-omic
singleomic_prefixes = ['clinical_', 'CNV_', 'rna_expr_', 'freebayes_SNV_', 'pindel_INDEL_']
analyte_combinations = (list(itertools.combinations(list(set(singleomic_prefixes)), 1)))
analyte_combinations += (list(itertools.combinations(list(set(singleomic_prefixes)), len(singleomic_prefixes))))
analyte_combinations

# +
categorical_label = 'label_deceased'
model_func = L1_Norm_RF_Model #L1_Norm_RF_Model, L1_Norm_MLP_Model, RFE_RF_Model
mtpilotMerged = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/mtpilot_tcga_harmonized_dataset.csv")
tcgaMerged = pd.read_csv(f"./molecular_twin_pilot/outputs/multiomic/tcga_harmonized_validation_dataset.csv")

# if limiting to parsimonious model features, only run multiomic model on restricted column set
if limit_to_parsimonious_features:
    analyte_combinations = (list(itertools.combinations(list(set(singleomic_prefixes)), len(singleomic_prefixes))))
    mtpilotMerged = mtpilotMerged[mtpilot_parsimonious_features + [categorical_label]]
    tcgaMerged = tcgaMerged[mtpilot_parsimonious_features + [categorical_label]]

# Train on MTPilot, validate on TCGA for each single-omic and multi-omic analytes
def test_feature_target_combo(model, X_train_features, y_train_labels, X_test_features, y_test_labels, scoring_metric='balanced_accuracy'):
    #model = RandomForestClassifier()
    model.fit(X_train_features, y_train_labels)

    # Predict
    pred = model.predict(X_test_features)
    
    # Score
    accuracy = model.score(X_test_features, y_test_labels)
    return pred, accuracy


def compute_model_eval_stats(actual_labels, test_labels_correct):
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
        "Accuracy": accuracy,
        "Precision": precision,
        "Sens": recall,
        "Spec": Spec,
        "F1": F1,
        "PPV": PPV,
        "NPV": NPV
    }


def pretty_print_prefix(prefix):
    if len(prefix) == 1:
        if prefix[0] == 'clinical_':
            return 'Clinical'
        elif prefix[0] == 'CNV_':
            return 'CNV'
        elif prefix[0] == 'freebayes_SNV_':
            return 'SNV'
        elif prefix[0] == 'rna_expr_':
            return 'RNA'
        elif prefix[0] == 'pindel_INDEL_':
            return 'INDEL'
        else:
            return 'UNKNOWN'
    else:
        return 'Clinical, DNA, RNA'


def process_prefix(prefix):
    rows = []
    X_train_features_label = subset_df_by_prefixes(mtpilotMerged, (*prefix, categorical_label))
    X_test_features_label = subset_df_by_prefixes(tcgaMerged, (*prefix, categorical_label))

    # Eliminate empty input samples and columns 
    X_train_features_label_dense = eliminate_sparsity(X_train_features_label)
    X_test_features_label_dense = eliminate_sparsity(X_test_features_label)
    
    # Subset to features available in both TCGA and MTPilot datasets
    common_features = set(np.intersect1d(X_train_features_label_dense.columns, X_test_features_label_dense.columns)) 
    X_train_features_subsetted = X_train_features_label_dense[common_features]
    X_test_features_subsetted = X_test_features_label_dense[common_features]

    # Obtain features
    X_train_features = X_train_features_subsetted.drop(columns=categorical_label)
    X_test_features = X_test_features_subsetted.drop(columns=categorical_label)
    
    # Obtain labels
    y_train = X_train_features_label_dense[categorical_label]
    y_test = X_test_features_label_dense[categorical_label]

    num_train_samples = len(X_train_features)
    num_test_samples = len(X_test_features)
    model_dict = {}
    test_metric = -1

    score_dict = {}
    try:
        print(f"Starting prefix: {prefix}, label: {categorical_label}, model: {model_func.__name__}, train shape: {X_train_features.shape}, test shape: {X_test_features.shape}")
        score_dict, test_metric = test_feature_target_combo(
            model=model_func(),
            X_train_features=X_train_features,
            y_train_labels=y_train,
            X_test_features=X_test_features,
            y_test_labels=y_test,
            scoring_metric='balanced_accuracy'
        )

        model_dict = {
            "analytes": pretty_print_prefix(prefix),
            "num_train_samples": num_train_samples,
            "num_test_samples": num_test_samples,
            "target_label": categorical_label,
            "num_features": len(X_train_features.columns),
        }

    except Exception as e:
        print(e)

    stats_dict = compute_model_eval_stats(actual_labels=y_test.values,
                                          test_labels_correct=score_dict)
    combined_dict = {**model_dict, **stats_dict}
    rows.append(combined_dict)
    print(f"Completed prefix: {prefix}, label:{categorical_label}, model: {model_func.__name__}, train shape: {X_train_features.shape}, test shape: {X_test_features.shape}, balanced_accuracy: {test_metric}")     
    return rows


# +
# Execute model training & validation for each analyte combination
flattened_rows = []
for prefix_item in analyte_combinations:
    flattened_rows.extend(process_prefix(prefix_item))

tcga_validated_categorical_df = pd.DataFrame(flattened_rows)
tcga_validated_categorical_df.to_csv(f"./molecular_twin_pilot/outputs/multiomic/tcga_validation_results{model_func.__name__}{'_parsim_cols' if limit_to_parsimonious_features else ''}.csv", index=False)
tcga_validated_categorical_df


