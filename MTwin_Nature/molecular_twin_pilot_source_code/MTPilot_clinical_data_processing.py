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
import pandas as pd
import os, datetime

BASE_DIR = "/molecular_twin_pilot/"
CLINICAL_INPUT_DIR = os.path.join(BASE_DIR, "clinical")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Clinical Study Sources
clinicalSource = pd.read_excel(os.path.join(CLINICAL_INPUT_DIR, "Pancreas_FFPE_Frozen_De_ID_MB.xlsx"))
clinicalAddendumSource = pd.read_excel(os.path.join(CLINICAL_INPUT_DIR, "Added_data_for_MTPC_study_De_ID_MB.xlsx"))
demographicsSource = pd.read_excel(os.path.join(CLINICAL_INPUT_DIR, "MTPC_PT_Demographics_DE_ID_MB.xlsx"))
qcResultsSource = pd.read_excel(os.path.join(CLINICAL_INPUT_DIR, "Cedars_Sinai_Tempus_Results_20201106.xlsx"))
lastFollowupSource = pd.read_excel(os.path.join(CLINICAL_INPUT_DIR, "Final_MRI_CT_W_tmr_information_MTCS_De_ID_MB.xlsx"))
latestPatientUpdate = pd.read_excel("/Users/onikolic/Downloads/New Dates Tumor Registry 2021-10-11 Final De-ID.xlsx")
# -

# Remove duplicate entries in the clinical input sources, each patient should have one entry.
clinicalSource.drop_duplicates(subset=['Biobank Number'], inplace=True)
clinicalAddendumSource.drop_duplicates(subset=['BioBank Number'], inplace=True)
demographicsSource.drop_duplicates(subset=['Biobank ID'], inplace=True)
qcResultsSource.drop_duplicates(subset=['Study Patient ID'], inplace=True)
lastFollowupSource.drop_duplicates(subset=['Biobank ID'], inplace=True)
latestPatientUpdate.drop_duplicates(subset=['Biobank ID'], inplace=True)

# +
from IPython.core.display import display, HTML
from IPython.display import IFrame

import functools
import numpy as np
import os, datetime
import pandas as pd

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))


# +
def cleanDataframe(cleaned_df):
    # Cleanup invalid characters in column names
    cleaned_df.columns = cleaned_df.columns.str.replace(',', '')
    cleaned_df.columns = cleaned_df.columns.str.replace('(', '')
    cleaned_df.columns = cleaned_df.columns.str.replace(')', '')
    cleaned_df.columns = cleaned_df.columns.str.replace('|', '_')
    cleaned_df.columns = cleaned_df.columns.str.replace(' ', '_')
    cleaned_df.columns = cleaned_df.columns.str.replace('/', '_')
    # Lowercase all strings in dataframe
    cleaned_df = cleaned_df.astype(str).apply(lambda x: x.str.lower())
    # Remove errant commas and semi-colons within cells for csv parsing
    cleaned_df.replace(',', '', regex=True, inplace=True)
    cleaned_df.replace(';', '', regex=True, inplace=True)
    cleaned_df.replace('\([0-9]*\)', '', regex=True, inplace=True)
    # Drop empty rows and columns, fill empty cells with appropriate defaults.
    cleaned_df.dropna(axis='index', how='all', inplace=True)
    cleaned_df.dropna(axis='columns', how='all', inplace=True)
    cleaned_df.fillna(value="", inplace=True, downcast='infer')
    return cleaned_df

clinicalSourceCleaned = cleanDataframe(clinicalSource)
clinicalAddendumSourceCleaned = cleanDataframe(clinicalAddendumSource)
demographicsSourceCleaned = cleanDataframe(demographicsSource)
qcResultsSourceCleaned = cleanDataframe(qcResultsSource)
lastFollowupSourceCleaned = cleanDataframe(lastFollowupSource)
latestPatientUpdate = latestPatientUpdate.astype(str).apply(lambda x: x.str.lower())

# +
# Filter for relevant columns and merge clinical, addendum, demographic and last followup data
clinicalFiltered = clinicalSourceCleaned[["Biobank_Number", "Sex", "Age_at_Diagnosis", "Site_-_Primary_ICD-O-3",
                                     "Histology_Behavior_ICD-O-3", "Histology_Text", "TNM_Mixed_Stage_T_Code",
                                     "TNM_Mixed_Stage_N_Code", "TNM_Mixed_Stage", "Grade_Mixed", "Tumor_Size_Summary",
                                     "Surgical_Margins_Summary", "Perineural_Invasion", "Lymphovascular_Invasion"]]
clinicalAddendumSourceCleaned.rename(columns={"BioBank_Number": "Biobank_Number",}, inplace=True)
addendumFiltered = clinicalAddendumSourceCleaned[["Biobank_Number", "Cancer_Status_Summary", "Text_Pathology", "Text_Surgery",
                                                  "Text_Chemotherapy", "Chemotherapy_Summary", "Radiation_Summary"]]
clinicalAddendumMerged = clinicalFiltered.merge(addendumFiltered)

demographicsSourceCleaned.rename(columns={"Biobank_ID": "Biobank_Number"}, inplace=True)
demographicsFiltered = demographicsSourceCleaned[["Biobank_Number", "Race", "Spanish_Hispanic_Origin", "Family_history_1st_any_cancer", "Family_history_1st_this_cancer",
                                                  "Family_history_2nd_any_cancer", "Family_history_2nd_this_cancer", "Height", "Weight", "Patient_History_of_Cancer_Seq_1",
                                                  "Patient_History_of_Cancer_Seq_2", "Patient_History_Alcohol", "Patient_History_Tobacco", "Secondary_Diagnosis_1",
                                                  "Secondary_Diagnosis_2", "Secondary_Diagnosis_3", "Secondary_Diagnosis_4", "Secondary_Diagnosis_5", "Secondary_Diagnosis_6",
                                                  "Secondary_Diagnosis_7", "Secondary_Diagnosis_8", "Secondary_Diagnosis_9", "Secondary_Diagnosis_10"]]
clinicalAddendumDemoMerged = clinicalAddendumMerged.merge(demographicsFiltered)

lastFollowupSourceCleaned.rename(columns={"Biobank_ID": "Biobank_Number", "VS": "Vital_Status"}, inplace=True)
lastFollowupFiltered = lastFollowupSourceCleaned[["Biobank_Number", "Vital_Status", "Date_Last_Contact_Death", "Date_Recur", "Recur_Type", "DATE_LAST_CA_STATUS"]]
clinicalAllMerged = clinicalAddendumDemoMerged.merge(lastFollowupFiltered)

# +
# Correct specific clinical values as per received study corrections.
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-554', 'Tumor_Size_Summary'] = 64
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-456', 'Tumor_Size_Summary'] = 35
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-590', 'Tumor_Size_Summary'] = 12

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-297', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-450', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-760', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-041', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-102', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-370', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-444', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-545', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-664', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-17-480', 'Vital_Status'] = 'dead'
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-17-615', 'Vital_Status'] = 'dead'

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-13-844', 'Height'] = 72
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-13-844', 'Weight'] = 154

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-13-930', 'Height'] = 65
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-13-930', 'Weight'] = 164

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-881', 'Height'] = 65
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-14-881', 'Weight'] = 109

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-336', 'Height'] = 71
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-16-336', 'Weight'] = 171

clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-17-537', 'Height'] = 64
clinicalAllMerged.loc[clinicalAllMerged['Biobank_Number'] == 'gi-17-537', 'Weight'] = 119

clinicalAllMerged[['Height', 'Weight', 'Tumor_Size_Summary']] = clinicalAllMerged[["Height", "Weight", "Tumor_Size_Summary"]].apply(pd.to_numeric)

# Clean Radiation_Summary and Chemotherapy_Summary values: 1) Drop patients with unknown status.  2) consolidate remaining status as appropriate.
clinicalAllMerged = clinicalAllMerged[clinicalAllMerged["Radiation_Summary"] != "9 (unknown)"]
clinicalAllMerged = clinicalAllMerged[clinicalAllMerged["Chemotherapy_Summary"] != "88 (recommendedunkn if given)"]

clinicalAllMerged.replace({'Chemotherapy_Summary': {'87 (refused)': '00 (none not planned)',
                                                    '82 (contraindicated)': '00 (none not planned)',
                                                   }}, inplace=True)

clinicalAllMerged['Chemotherapy_Binary'] = clinicalAllMerged['Chemotherapy_Summary'].map({'00 (none not planned)': 0, '01 (chemo nos)': 1, '02 (single-agent chemo)': 1, '03 (multi-agent chemo)': 1,})
clinicalAllMerged[['Age_at_Diagnosis', 'Tumor_Size_Summary']] = clinicalAllMerged[['Age_at_Diagnosis', 'Tumor_Size_Summary']].apply(pd.to_numeric, errors='coerce')

# Drop empty rows and columns.
clinicalAllMerged.dropna(axis='index', how='all', inplace=True)
clinicalAllMerged.dropna(axis='columns', how='all', inplace=True)

# +
# Derive clinical features

# Compute patient BMI
clinicalAllMerged['BMI'] = 703 * clinicalAllMerged['Weight'] / (clinicalAllMerged['Height'] * clinicalAllMerged['Height'])
# Merge race and hispanic columns
clinicalAllMerged['merged_ethnicity'] = np.where((clinicalAllMerged['Race'] == "white") & (clinicalAllMerged['Spanish_Hispanic_Origin'] != "unknown") & (clinicalAllMerged['Spanish_Hispanic_Origin'] != "non-spanish"), clinicalAllMerged['Spanish_Hispanic_Origin'], clinicalAllMerged['Race'])
clinicalAllMerged['Vital Status'] = clinicalAllMerged['Vital_Status'].map({'dead': 'Deceased', 'alive': 'Alive'})

# Create target labels (alive/dead binary, recurrence binary, alive/recur-alive/recur-dead/dead categorical, time to death, time to recurrence, tumor-stage categorical)
clinicalAllMerged['label_patient_survival'] = clinicalAllMerged['Vital_Status'].map({'dead': '0', 'alive': 1})
clinicalAllMerged['label_recurrence'] = clinicalAllMerged.apply(lambda x: True if x['Date_Recur'].startswith('day') else False, axis=1)

#clinicalAllMerged['label_tnm_sub_stage'] = clinicalAllMerged['TNM_Mixed_Stage'].map({'1a': 0, '1b': 1, '2a': 2, '2b': 3, '3': 4, '4': 5})
clinicalAllMerged['TNM_Mixed_Substage_ord'] = clinicalAllMerged['TNM_Mixed_Stage'].map({'1a': 0, '1b': 1, '2a': 2, '2b': 3, '3': 4, '4': 5})
clinicalAllMerged['TNM_Mixed_Stage_ord'] = clinicalAllMerged['TNM_Mixed_Stage'].map({'1a': 0, '1b': 0, '2a': 1, '2b': 1, '3': 2, '4': 3})

# Drop stage 3 & 4 patients from study (reduces study cohort from 93 to 80 patients)
clinicalAllMerged = clinicalAllMerged[clinicalAllMerged["TNM_Mixed_Stage_ord"] <= 1]

label_columns = ["label_patient_survival", "label_recurrence",] #"label_tnm_stage", "label_outcome_categorical", "label_days_to_death", "label_days_to_recurrence", "label_tnm_sub_stage", ]
clinicalAllMerged[label_columns] = clinicalAllMerged[label_columns].apply(pd.to_numeric, errors='coerce', axis=1)
clinicalAllMerged = clinicalAllMerged.loc[:, ~clinicalAllMerged.columns.duplicated()]
clinicalAllMerged.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_clinical_features_with_raw.csv"), index=False)
# -

clinicalAllMerged['Sex'] = clinicalAllMerged['Sex'].map({'1 (male)': 'Male', '2 (female)': 'Female'})
ax = clinicalAllMerged['Sex'].value_counts().plot(kind='bar', figsize=(10,8), fontsize=14)
ax.set_xlabel("Sex", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_sex.png", bbox_inches='tight')

ax = clinicalAllMerged['Age_at_Diagnosis'].plot.hist(bins=8, figsize=(10,8), fontsize=14)
ax.set_xlabel("Age at Diagnosis", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_age.png", bbox_inches='tight')

ax = clinicalAllMerged['BMI'].plot.hist(bins=10, figsize=(10,8), fontsize=14)
ax.set_xlabel("BMI", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_bmi.png", bbox_inches='tight')

ax = clinicalAllMerged['Vital Status'].value_counts().plot(kind='bar', figsize=(10,8), fontsize=14)
ax.set_xlabel("Disease Survival", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_survival.png", bbox_inches='tight')

clinicalAllMerged['Recurrence'] = clinicalAllMerged['label_recurrence'].map({0:'False', 1: 'True'})
ax = clinicalAllMerged['Recurrence'].value_counts().plot(kind='bar', figsize=(10,8), fontsize=14)
ax.set_xlabel("Disease Recurrence", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_recurrence.png", bbox_inches='tight')

ax = clinicalAllMerged["TNM_Mixed_Stage"].value_counts().sort_values().plot(kind='bar', figsize=(10,8), fontsize=14)
ax.set_xlabel("TNM Stage", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.figure.savefig("MTPilot_tnm_stage.png", bbox_inches='tight')

# +
latestPatientUpdate['Chemo or Surgery Date First'] = np.where(latestPatientUpdate["Day of 1st Chemotherapy"].isna(), 'None', latestPatientUpdate['Chemo or Surgery Date First'])
latestPatientUpdate.rename(columns={'Chemo or Surgery Date First': 'Chemotherapy_Type', 'Biobank ID': 'Biobank_Number'}, inplace=True)
latestPatientUpdate['Chemotherapy_Type'] = latestPatientUpdate['Chemotherapy_Type'].map({'None': 'None', 'surgery': 'Adjuvant', 'chemo': 'Neoadjuvant'})
clinicalAllMergedWithUpdate = clinicalAllMerged.merge(latestPatientUpdate, on='Biobank_Number')

ax = clinicalAllMergedWithUpdate["Chemotherapy_Type"].value_counts().sort_values().plot(kind='bar', figsize=(10,8), fontsize=14)
ax.set_xlabel("Chemotherapy", fontsize=18)
ax.set_ylabel("Number of Patients", fontsize=18)
ax.set_facecolor('white')
ax.figure.savefig("MTPilot_chemotherapy_type.png", bbox_inches='tight')
# -

# Look at distribution of clinical variables
clinicalAllMerged_statistics = tfdv.generate_statistics_from_dataframe(clinicalAllMerged)
tfdv.visualize_statistics(clinicalAllMerged_statistics)

# +
# Ordinal encodings of categorical clinical variables
# Convert all categorical columns into 'category' type then obtain their numeric codes.
clinicalAllMerged['Sex_ord'] = clinicalAllMerged["Sex"].astype('category')
clinicalAllMerged['Site_-_Primary_ICD-O-3_ord'] = clinicalAllMerged["Site_-_Primary_ICD-O-3"].astype('category')
clinicalAllMerged['Histology_Behavior_ICD-O-3_ord'] = clinicalAllMerged["Histology_Behavior_ICD-O-3"].astype('category')
clinicalAllMerged['TNM_Mixed_Stage_ord'] = clinicalAllMerged["TNM_Mixed_Stage"].astype('category')
clinicalAllMerged['Grade_Mixed_ord'] = clinicalAllMerged["Grade_Mixed"].astype('category')
clinicalAllMerged['Surgical_Margins_Summary_ord'] = clinicalAllMerged["Surgical_Margins_Summary"].astype('category')
clinicalAllMerged['Chemotherapy_Summary_ord'] = clinicalAllMerged["Chemotherapy_Summary"].astype('category')
clinicalAllMerged['Radiation_Summary_ord'] = clinicalAllMerged["Radiation_Summary"].astype('category')
clinicalAllMerged['Perineural_Invasion_ord'] = clinicalAllMerged["Perineural_Invasion"].astype('category')
clinicalAllMerged['Lymphovascular_Invasion_ord'] = clinicalAllMerged["Lymphovascular_Invasion"].astype('category')
clinicalAllMerged['Family_history_1st_any_cancer_ord'] = clinicalAllMerged["Family_history_1st_any_cancer"].astype('category')
clinicalAllMerged['Family_history_2nd_any_cancer_ord'] = clinicalAllMerged["Family_history_2nd_any_cancer"].astype('category')
clinicalAllMerged['Family_history_1st_this_cancer_ord'] = clinicalAllMerged["Family_history_1st_this_cancer"].astype('category')
clinicalAllMerged['Family_history_2nd_this_cancer_ord'] = clinicalAllMerged["Family_history_2nd_this_cancer"].astype('category')
clinicalAllMerged['Patient_History_of_Cancer_Seq_1_ord'] = clinicalAllMerged["Patient_History_of_Cancer_Seq_1"].astype('category')
clinicalAllMerged['Patient_History_of_Cancer_Seq_2_ord'] = clinicalAllMerged["Patient_History_of_Cancer_Seq_2"].astype('category')
clinicalAllMerged['Patient_History_Alcohol_ord'] = clinicalAllMerged["Patient_History_Alcohol"].astype('category')
clinicalAllMerged['Patient_History_Tobacco_ord'] = clinicalAllMerged["Patient_History_Tobacco"].astype('category')
clinicalAllMerged['merged_ethnicity_ord'] = clinicalAllMerged["merged_ethnicity"].astype('category')

clinicalAllMerged['Sex_ord'] = clinicalAllMerged["Sex_ord"].cat.codes
clinicalAllMerged['Site_-_Primary_ICD-O-3_ord'] = clinicalAllMerged["Site_-_Primary_ICD-O-3_ord"].cat.codes
clinicalAllMerged['Grade_Mixed_ord'] = clinicalAllMerged["Grade_Mixed_ord"].cat.codes
clinicalAllMerged['Surgical_Margins_Summary_ord'] = clinicalAllMerged["Surgical_Margins_Summary_ord"].cat.codes
clinicalAllMerged['Chemotherapy_Summary_ord'] = clinicalAllMerged["Chemotherapy_Summary_ord"].cat.codes
clinicalAllMerged['Radiation_Summary_ord'] = clinicalAllMerged["Radiation_Summary_ord"].cat.codes
clinicalAllMerged['Perineural_Invasion_ord'] = clinicalAllMerged["Perineural_Invasion_ord"].cat.codes
clinicalAllMerged['Lymphovascular_Invasion_ord'] = clinicalAllMerged["Lymphovascular_Invasion_ord"].cat.codes
clinicalAllMerged['Family_history_1st_any_cancer_ord'] = clinicalAllMerged["Family_history_1st_any_cancer_ord"].cat.codes
clinicalAllMerged['Family_history_2nd_any_cancer_ord'] = clinicalAllMerged["Family_history_2nd_any_cancer_ord"].cat.codes
clinicalAllMerged['Family_history_1st_this_cancer_ord'] = clinicalAllMerged["Family_history_1st_this_cancer_ord"].cat.codes
clinicalAllMerged['Family_history_2nd_this_cancer_ord'] = clinicalAllMerged["Family_history_2nd_this_cancer_ord"].cat.codes
clinicalAllMerged['Patient_History_of_Cancer_Seq_1_ord'] = clinicalAllMerged["Patient_History_of_Cancer_Seq_1_ord"].cat.codes
clinicalAllMerged['Patient_History_of_Cancer_Seq_2_ord'] = clinicalAllMerged["Patient_History_of_Cancer_Seq_2_ord"].cat.codes
clinicalAllMerged['Patient_History_Alcohol_ord'] = clinicalAllMerged["Patient_History_Alcohol_ord"].cat.codes
clinicalAllMerged['Patient_History_Tobacco_ord'] = clinicalAllMerged["Patient_History_Tobacco_ord"].cat.codes
clinicalAllMerged['merged_ethnicity_ord'] = clinicalAllMerged["merged_ethnicity_ord"].cat.codes
# -

# One-hot encode pre-existing conditions Secondary_Diagnosis_[1-10] columns
secondary_diagnosis_cols = ['Secondary_Diagnosis_1', 'Secondary_Diagnosis_2', 'Secondary_Diagnosis_3', 'Secondary_Diagnosis_4',
                            'Secondary_Diagnosis_5', 'Secondary_Diagnosis_6', 'Secondary_Diagnosis_7', 'Secondary_Diagnosis_8',
                            'Secondary_Diagnosis_9', 'Secondary_Diagnosis_10']
one_hot_encoded_df = pd.get_dummies(clinicalAllMerged[secondary_diagnosis_cols], prefix='secondary_diagnosis_onehot_')
clinicalAllMerged = pd.concat([clinicalAllMerged, one_hot_encoded_df], axis=1)

import matplotlib.pyplot as plt
one_hot_encoded_df.plot(kind="hist").hist

fullColumnsClinical = clinicalAllMerged

# +
# Drop all raw clinical inputs after they have been cleaned and featurized.
clinicalAllMerged.drop(columns=[
       'Sex', 'Site_-_Primary_ICD-O-3', 'Histology_Behavior_ICD-O-3', 'Recurrence',
       'Histology_Text', 'TNM_Mixed_Stage_T_Code', 'TNM_Mixed_Stage_N_Code',
       'TNM_Mixed_Stage', 'Grade_Mixed', 'Tumor_Size_Summary', 'Chemotherapy_Summary',
       'Radiation_Summary', 'Perineural_Invasion', 'Surgical_Margins_Summary',
       'Lymphovascular_Invasion', 'Cancer_Status_Summary', 'Text_Pathology',
       'Text_Surgery', 'Text_Chemotherapy', 'Race', 'Spanish_Hispanic_Origin',
       'Family_history_1st_any_cancer', 'Family_history_1st_this_cancer',
       'Family_history_2nd_any_cancer', 'Family_history_2nd_this_cancer',
       'Patient_History_of_Cancer_Seq_1', 'Patient_History_Alcohol',
       'Patient_History_of_Cancer_Seq_2', 'Patient_History_Tobacco',
       'Secondary_Diagnosis_1', 'Secondary_Diagnosis_2', 'Secondary_Diagnosis_3',
       'Secondary_Diagnosis_4', 'Secondary_Diagnosis_5', 'Secondary_Diagnosis_6',
       'Secondary_Diagnosis_7', 'Secondary_Diagnosis_8', 'Secondary_Diagnosis_9',
       'Secondary_Diagnosis_10', 'Vital_Status', 'Date_Last_Contact_Death', 'Date_Recur',
       'Recur_Type', 'DATE_LAST_CA_STATUS', 'merged_ethnicity'], inplace=True)

clinicalAllMerged = clinicalAllMerged.add_prefix("clinical_")
clinicalAllMerged.rename(columns={"clinical_Biobank_Number": "Biobank_Number",
                                  "clinical_label_patient_survival": "label_patient_survival",
                                  "clinical_label_recurrence": "label_recurrence",
                                  "clinical_label_tnm_stage": "label_tnm_stage"}, inplace=True)
# -

clinicalAllMerged.columns

# Save featurized, clinical baseline dataset
CLINICAL_OUTPUT_DIR = os.path.join(BASE_DIR, "clinical")

clinicalAllMerged.to_csv(os.path.join(CLINICAL_OUTPUT_DIR, "early_stage_patients_clinical_features_with_metadata.csv"), index=False)
clinicalLabels = clinicalAllMerged[["Biobank_Number"] + label_columns]
clinicalLabels["Biobank_Number"] = clinicalLabels['Biobank_Number'].str.upper()
clinicalLabels.to_csv(os.path.join(OUTPUT_DIR, "early_stage_patients_clinical_labels.csv"), index=False)


fullColumnsClinical.columns

eda_num(fullColumnsClinical)

eda_cat(fullColumnsClinical, x='Vital_Status', y='TNM_Mixed_Stage')

eda_cat(fullColumnsClinical, x='Vital_Status', y='Sex')

eda_num(fullColumnsClinical, method="correlation")

eda_cat(fullColumnsClinical, x='Age_at_Diagnosis', y='Sex_ord')

# Feature Importance
eda_numcat(fullColumnsClinical, method='pps', x='label_patient_survival')

eda_numcat(fullColumnsClinical, method='pps', x='label_recurrence')


