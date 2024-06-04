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
import pandas as pd
import dataset_manager.workflow_engine.argo.engine as engine
import dataset_manager.workflow_engine.argo.translation as translation
import sister
import tensorflow as tf
import tensorflow_data_validation as tfdv

from IPython.core.display import display, HTML
display(HTML("<style>.container { width:100% !important; }</style>"))

BASE_DIR = "./molecular_twin_pilot/"
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs/multiomic") #, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
# -

# # Declare workflow inputs.

# +
cedarsMTWorkflow, inputSources = createWorkflow("cedars-mt-multiomic") 

# Multi-omic Tabular Sources
studyLabels = createSource("study_labels", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "studyLabels.csv"))
clinicalMetadata = createSource("clinical_metadata", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "clinicalMetadata.csv"))
clinicalFeatures = createSource("clinical_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "clinicalNormalizedFeatures.csv"))
notesNLPFeatures = createSource("notes_nlp_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "clinicalNLPEmbeddings.csv"))
plasmaProteinFeatures = createSource("plasma_protein_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "plasmaProteinFeaturesCleaned.csv"))
tissueProteinFeatures = createSource("tissue_protein_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "tissueProteinFeaturesCleaned.csv"))
plasmaLipidFeatures = createSource("plasma_lipid_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/clinical", "plasmaLipidFeaturesCleaned.csv"))
pathologyFeatures = createSource("pathology_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "outputs/pathology", "pathologyFeatures.csv"))
cnvFeatures = createSource("cnv_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "DNA/test-output", "process-cnv-files.csv"))
snvFeatures = createSource("snv_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "DNA/test-output", "process-freebayes-variants.csv"))
indelsFeatures = createSource("indels_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "DNA/test-output", "process-pindel-variants.csv"))
rnaFusionFeatures = createSource("rna_fusion_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "RNA/test-output", "process-rna-fusion.csv"))
rnaGeneExprFeatures = createSource("rna_gene_expr_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "RNA/tx_to_gene", "tx-to-gene-parsed.csv"))
#rnaTransExprFeatures = createSource("rna_trans_expr_features", wf.Data.Format.CSV, os.path.join(BASE_DIR, "RNA/kallisto", "quantify-paired-end.csv"))
        
inputSources.input_sources.extend([studyLabels, clinicalMetadata, clinicalFeatures, notesNLPFeatures, plasmaProteinFeatures, tissueProteinFeatures, plasmaLipidFeatures, pathologyFeatures, cnvFeatures, snvFeatures, indelsFeatures, rnaFusionFeatures, rnaGeneExprFeatures])
# -

inputSources

# # Define data preparation workflow.

# +
# Remove duplicate entries in the *omics input sources, each patient should have one entry.
studyLabelsDedup = createOperation(op_name="studyLabelsDedup", input_sources = inputSources, 
                                   op_type="drop_duplicates", inputs=["study_labels"],
                                   op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
clinicalMetadataDedup = createOperation(op_name="clinicalMetadataDedup", input_sources = inputSources, 
                                        op_type="drop_duplicates", inputs=["clinical_metadata"],
                                        op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
clinicalFeaturesDedup = createOperation(op_name="clinicalFeaturesDedup", input_sources = inputSources, 
                                       op_type="drop_duplicates", inputs=["clinical_features"],
                                       op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
notesNLPDedup = createOperation(op_name="notesNLPDedup", input_sources = inputSources, 
                                op_type="drop_duplicates", inputs=["notes_nlp_features"],
                                op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
plasmaProteinFeaturesDedup = createOperation(op_name="plasmaProteinFeaturesDedup", input_sources = inputSources, 
                                       op_type="drop_duplicates", inputs=["plasma_protein_features"],
                                       op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
tissueProteinFeaturesDedup = createOperation(op_name="tissueProteinFeaturesDedup", input_sources = inputSources, 
                                       op_type="drop_duplicates", inputs=["tissue_protein_features"],
                                       op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
plasmaLipidFeaturesDedup = createOperation(op_name="plasmaLipidFeaturesDedup", input_sources = inputSources, 
                                       op_type="drop_duplicates", inputs=["plasma_lipid_features"],
                                       op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
pathologyFeaturesDedup = createOperation(op_name="pathologyFeaturesDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["pathology_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["Biobank_Number"])
cnvFeaturesDedup = createOperation(op_name="cnvFeaturesDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["cnv_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["sample"])
snvFeaturesDedup = createOperation(op_name="snvFeaturesDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["snv_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["sample"])
indelsFeaturesDedup = createOperation(op_name="indelsFeaturesDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["indels_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["sample"])
rnaFusionsDedup = createOperation(op_name="rnaFusionsDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["rna_fusion_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["sample"])
rnaGeneExprDedup = createOperation(op_name="rnaGeneExprDedup", input_sources = inputSources, 
                                         op_type="drop_duplicates", inputs=["rna_gene_expr_features"],
                                         op_sub_type=wf.DropDuplicates.RowToKeep.FIRST, duplicate_cols=["sample"])

# Study Labels cleaning: 1) convert NAs to 0; 2) Columns already prefixed with 'label_'
studyLabelsCleaned = createOperation(op_name = "studyLabelsCleaned", input_sources = inputSources, op_type = "python_script",
                                  inputs = [studyLabelsDedup.name], expr="""
import re
studyLabelsDedup.fillna(0, downcast='infer', inplace=True)
studyLabelsDedup["Biobank_Number"] = studyLabelsDedup["Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
studyLabelsDedup["Biobank_Number"] = studyLabelsDedup["Biobank_Number"].str.upper()
output = studyLabelsDedup
""")

# Clinical metadata cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'metadata_'
clinicalMetadataCleaned = createOperation(op_name = "clinicalMetadataCleaned", input_sources = inputSources, op_type = "python_script",
                                          inputs = [clinicalMetadataDedup.name], expr="""
import re
clinicalMetadataDedup.fillna(0, downcast='infer', inplace=True)
clinicalMetadataDedup = clinicalMetadataDedup.add_prefix('metadata_')
clinicalMetadataDedup["Biobank_Number"] = clinicalMetadataDedup["metadata_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
clinicalMetadataDedup["Biobank_Number"] = clinicalMetadataDedup["Biobank_Number"].str.upper()
clinicalMetadataDedup.drop(["metadata_Biobank_Number"], axis=1, inplace=True)
output = clinicalMetadataDedup
""")

# Clinical features cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'clinical_'
clinicalFeaturesCleaned = createOperation(op_name = "clinicalFeaturesCleaned", input_sources = inputSources, op_type = "python_script",
                                          inputs = [clinicalFeaturesDedup.name], expr="""
import re
clinicalFeaturesDedup.fillna(0, downcast='infer', inplace=True)
clinicalFeaturesDedup = clinicalFeaturesDedup.add_prefix('clinical_')
clinicalFeaturesDedup["Biobank_Number"] = clinicalFeaturesDedup["clinical_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
clinicalFeaturesDedup["Biobank_Number"] = clinicalFeaturesDedup["Biobank_Number"].str.upper()
clinicalFeaturesDedup.drop(["clinical_Biobank_Number"], axis=1, inplace=True)
output = clinicalFeaturesDedup
""")

# NotesNLP cleaning: 1) convert NAs to 0; 2) Columns already prefixed with '{surgery|chemotherapy|pathology}_embed_'
notesNLPCleaned = createOperation(op_name = "notesNLPCleaned", input_sources = inputSources, op_type = "python_script",
                                  inputs = [notesNLPDedup.name], expr="""
import re
notesNLPDedup.fillna(0, downcast='infer', inplace=True)
notesNLPDedup["Biobank_Number"] = notesNLPDedup["Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
notesNLPDedup["Biobank_Number"] = notesNLPDedup["Biobank_Number"].str.upper()
output = notesNLPDedup
""")

# Plasma Protein cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'plasma_protein_'
plasmaProteinCleaned = createOperation(op_name = "plasmaProteinCleaned", input_sources = inputSources, op_type = "python_script",
                                 inputs = [plasmaProteinFeaturesDedup.name], expr="""
import re
plasmaProteinFeaturesDedup.fillna(0, downcast='infer', inplace=True)
plasmaProteinFeaturesDedup = plasmaProteinFeaturesDedup.add_prefix('plasma_protein_')
plasmaProteinFeaturesDedup["Biobank_Number"] = plasmaProteinFeaturesDedup["plasma_protein_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
plasmaProteinFeaturesDedup["Biobank_Number"] = plasmaProteinFeaturesDedup["Biobank_Number"].str.upper()
plasmaProteinFeaturesDedup.drop(["plasma_protein_Biobank_Number"], axis=1, inplace=True)
output = plasmaProteinFeaturesDedup
""")

# Tissue Protein cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'tissue_protein_'
tissueProteinCleaned = createOperation(op_name = "tissueProteinCleaned", input_sources = inputSources, op_type = "python_script",
                                       inputs = [tissueProteinFeaturesDedup.name], expr="""
import re
tissueProteinFeaturesDedup.fillna(0, downcast='infer', inplace=True)
tissueProteinFeaturesDedup = tissueProteinFeaturesDedup.add_prefix('tissue_protein_')
tissueProteinFeaturesDedup.rename(columns={"tissue_protein_Biobank_Number": "Biobank_Number"}, inplace=True)

output = tissueProteinFeaturesDedup
""")

# Plasma Lipid cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'lipid_'
plasmaLipidCleaned = createOperation(op_name = "plasmaLipidCleaned", input_sources = inputSources, op_type = "python_script",
                                       inputs = [plasmaLipidFeaturesDedup.name], expr="""
import re
plasmaLipidFeaturesDedup.fillna(0, downcast='infer', inplace=True)
plasmaLipidFeaturesDedup = plasmaLipidFeaturesDedup.add_prefix('plasma_lipid_')
plasmaLipidFeaturesDedup.rename(columns={"plasma_lipid_Biobank_Number": "Biobank_Number"}, inplace=True)

output = plasmaLipidFeaturesDedup
""")

# Pathology cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'pathology_'
pathologyCleaned = createOperation(op_name = "pathologyCleaned", input_sources = inputSources, op_type = "python_script",
                                   inputs = [pathologyFeaturesDedup.name], expr="""
import re
pathologyFeaturesDedup.fillna(0, downcast='infer', inplace=True)
pathologyFeaturesDedup = pathologyFeaturesDedup.add_prefix('pathology_')
pathologyFeaturesDedup["Biobank_Number"] = pathologyFeaturesDedup["pathology_Biobank_Number"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
pathologyFeaturesDedup["Biobank_Number"] = pathologyFeaturesDedup["Biobank_Number"].str.upper()
pathologyFeaturesDedup.drop(["pathology_Biobank_Number"], axis=1, inplace=True)
output = pathologyFeaturesDedup
""")

# DNA CNVs cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'CNV_'
cnvCleaned = createOperation(op_name = "cnvCleaned", input_sources = inputSources, op_type = "python_script",
                             inputs = [cnvFeaturesDedup.name], expr="""
import re
cnvFeaturesDedup.fillna(0, downcast='infer', inplace=True)
cnvFeaturesDedup["Biobank_Number"] = cnvFeaturesDedup["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
cnvFeaturesDedup["Biobank_Number"] = cnvFeaturesDedup["Biobank_Number"].str.upper()
cnvFeaturesDedup.drop(["sample"], axis=1, inplace=True)
output = cnvFeaturesDedup
""")

# DNA SNVs cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'freebayes_SNV_'
snvCleaned = createOperation(op_name = "snvCleaned", input_sources = inputSources, op_type = "python_script",
                             inputs = [snvFeaturesDedup.name], expr="""
import re
snvFeaturesDedup.fillna(0, downcast='infer', inplace=True)
snvFeaturesDedup["Biobank_Number"] = snvFeaturesDedup["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
snvFeaturesDedup["Biobank_Number"] = snvFeaturesDedup["Biobank_Number"].str.upper()
snvFeaturesDedup.drop(["sample"], axis=1, inplace=True)
output = snvFeaturesDedup
""")

# DNA Indels cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'pindel_INDEL_'
indelsCleaned = createOperation(op_name = "indelsCleaned", input_sources = inputSources, op_type = "python_script",
                                inputs = [indelsFeaturesDedup.name], expr="""
import re
indelsFeaturesDedup.fillna(0, downcast='infer', inplace=True)
indelsFeaturesDedup["Biobank_Number"] = indelsFeaturesDedup["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
indelsFeaturesDedup["Biobank_Number"] = indelsFeaturesDedup["Biobank_Number"].str.upper()
indelsFeaturesDedup.drop(["sample"], axis=1, inplace=True)
output = indelsFeaturesDedup
""")

# RNA Fusions cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Columns already prefixed with 'AF4_'
rnaFusionsCleaned = createOperation(op_name = "rnaFusionsCleaned", input_sources = inputSources, op_type = "python_script",
                                    inputs = [rnaFusionsDedup.name], expr="""
import re
rnaFusionsDedup.fillna(0, downcast='infer', inplace=True)
rnaFusionsDedup["Biobank_Number"] = rnaFusionsDedup["sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
rnaFusionsDedup["Biobank_Number"] = rnaFusionsDedup["Biobank_Number"].str.upper()
rnaFusionsDedup.drop(["sample"], axis=1, inplace=True)
output = rnaFusionsDedup
""")

# RNA Gene Expr cleaning: 1) convert NAs to 0; 2) Extract Biobank_Number; 3) Prefix columns with 'rna_gene_expr_'
rnaGeneExprCleaned = createOperation(op_name = "rnaGeneExprCleaned", input_sources = inputSources, op_type = "python_script",
                                     dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python:latest",
                                     inputs = [rnaGeneExprDedup.name], expr="""
import numpy as np
import re

rnaGeneExprDedup.fillna(0, downcast='infer', inplace=True)
rnaGeneExprDedup = rnaGeneExprDedup.add_prefix('rna_gene_expr_')
rnaGeneExprDedup["Biobank_Number"] = rnaGeneExprDedup["rna_gene_expr_sample"].str.extract(r'(GI-\d{2}-\d{3})', flags=re.I).astype(str)
rnaGeneExprDedup["Biobank_Number"] = rnaGeneExprDedup["Biobank_Number"].str.upper()
rnaGeneExprDedup.drop(["rna_gene_expr_sample"], axis=1, inplace=True)

# Log transform RNA Gene Expression features.
# TODO(oggie): parameterize if RNA inputs are transcripts then log-normalize, if gene_expr, then it is normalized already.
#subsetted_rna_columns = rnaGeneExprDedup.columns[rnaGeneExprDedup.columns.str.contains("rna_gene_expr_")]
#rnaGeneExprDedup[subsetted_rna_columns] = np.log(rnaGeneExprDedup[subsetted_rna_columns] + 1)

output = rnaGeneExprDedup
""")

# Merge all input sources.
mergeLabelsMetadata = createOperation(op_name="mergeLabelsMetadata", input_sources = inputSources,
                                       inputs=[studyLabelsCleaned.name, clinicalMetadataCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeClinical = createOperation(op_name="mergeClinical", input_sources = inputSources,
                                       inputs=[mergeLabelsMetadata.name, clinicalFeaturesCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeNotesNLP = createOperation(op_name="mergeNotesNLP", input_sources = inputSources,
                                       inputs=[mergeClinical.name, notesNLPCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergePlasmaProtein = createOperation(op_name="mergePlasmaProtein", input_sources = inputSources,
                                       inputs=[mergeNotesNLP.name, plasmaProteinCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeTissueProtein = createOperation(op_name="mergeTissueProtein", input_sources = inputSources,
                                       inputs=[mergePlasmaProtein.name, tissueProteinCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergePlasmaLipid = createOperation(op_name="mergePlasmaLipid", input_sources = inputSources,
                                       inputs=[mergeTissueProtein.name, plasmaLipidCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergePathology = createOperation(op_name="mergePathology", input_sources = inputSources,
                                       inputs=[mergePlasmaLipid.name, pathologyCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeCNV = createOperation(op_name="mergeCNV", input_sources = inputSources,
                                       inputs=[mergePathology.name, cnvCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeSNV = createOperation(op_name="mergeSNV", input_sources = inputSources,
                                       inputs=[mergeCNV.name, snvCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeIndels = createOperation(op_name="mergeIndels", input_sources = inputSources,
                                       inputs=[mergeSNV.name, indelsCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeRnaFusions = createOperation(op_name="mergeRnaFusions", input_sources = inputSources,
                                       inputs=[mergeIndels.name, rnaFusionsCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])
mergeRnaGeneExpr = createOperation(op_name="mergeRnaGeneExpr", input_sources = inputSources,
                                       inputs=[mergeRnaFusions.name, rnaGeneExprCleaned.name], op_type="merge",
                                       op_sub_type=wf.MergeDatasOp.LEFT,  join_cols=[])

# Apply feature normalization, rescaling, feature pruning.
multiomicNormalizedFeatures = createOperation(op_name = "multiomicNormalizedFeatures", input_sources = inputSources, op_type = "python_script",
                                              dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-expanded:latest",
                                              inputs = [mergeRnaGeneExpr.name], materialize_output=True, expr="""
from sklearn.feature_selection import VarianceThreshold

def pruneLowVarianceFeatures(features, threshold=0.05):
    feature_selector = VarianceThreshold(threshold)

    metadataColumns = features.columns[features.columns.str.startswith("metadata_")].tolist() + \
                      features.columns[features.columns.str.startswith("label_")].tolist() + ["Biobank_Number"]
    metadataFeatures = features.loc[:, metadataColumns]
    numericFeatures = features.drop(metadataColumns, axis=1)
    numericPrunedFeatures = pd.DataFrame(feature_selector.fit_transform(numericFeatures))
    numericPrunedFeatures['Biobank_Number'] = features['Biobank_Number']
    return numericPrunedFeatures

# Remove all constant columns.
import pandas as pd
output = mergeRnaGeneExpr.loc[:, mergeRnaGeneExpr.nunique(axis=0, dropna=True) != 1]
""")

# Prune out highly correlated features within each singleomics source.
multiomicPruneCorrelatedFeatures = createOperation(op_name = "multiomicPruneCorrelatedFeatures", input_sources = inputSources, op_type = "python_script",
                                          dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-expanded:latest",
                                          inputs = [multiomicNormalizedFeatures.name], resource_size=wf.Operation.OperationSize.MEDIUM, materialize_output=True, expr="""
import numpy as np

# TODO(oggie): raise minimum column threshold to 31K, currently OOM but larger nodes can't be scheduled.
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

# Drop correlated features within each singleomic source.
column_prefixes = ["plasma_protein_", "tissue_protein_", "plasma_lipid_", "pathology_NF", "CNV_", "freebayes_SNV_", "pindel_INDEL_", "AF4_", "rna_gene_expr_"]

columns_to_drop = []
for prefix in column_prefixes:
    singleomic_columns = multiomicNormalizedFeatures.columns[multiomicNormalizedFeatures.columns.str.contains(prefix)].tolist()
    singleomic_dataset = multiomicNormalizedFeatures.loc[:, singleomic_columns]
    singleomic_dataset.dropna(axis=0, how='all', inplace=True)
    singleomic_dataset.dropna(axis=1, how='all', inplace=True)
    columns_to_drop += get_correlated_columns(singleomic_dataset, 0.95)

output = multiomicNormalizedFeatures.drop(columns_to_drop, axis=1)
""")
# -

# ## Identify top predictive biomarkers (feature importance) for merged dataset.

# Determine and extract top predictive biomarkers (feature importance) for each target objective.
multiomicPredictiveBiomarkers = createOperation(op_name="multiomicPredictiveBiomarkers", input_sources = inputSources, op_type="python_script",
                                               inputs = [multiomicNormalizedFeatures.name], materialize_output=True,
                                               dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python:latest",
                                               expr="""
# Train independent models to identify predictive biomarker importance for target objectives.

import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier

target_labels = ["label_outcome_binary", "label_recurrence_binary", "label_outcome_categorical", "label_tnm_sub_stage", "label_tnm_stage"]

filtered_dataset = multiomicNormalizedFeatures.fillna(0, downcast='infer')
dataset = filtered_dataset.select_dtypes(['number'])
dataset = dataset[dataset.columns.drop(list(dataset.filter(regex='label_')))]

d = {}
for label_column in target_labels:
    d[label_column] = {}
    if label_column in dataset.columns:
        X = dataset.drop(label_column, axis=1)
    else:
        X = dataset
    y = filtered_dataset[label_column].astype('int')


    # Build a forest and compute the impurity-based feature importances
    forest = ExtraTreesClassifier(n_estimators=250,
                                  random_state=0,
                                  n_jobs=10)

    forest.fit(X, y)
    importances = forest.feature_importances_
    std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                 axis=0)
    indices = np.argsort(importances)[::-1]
    
    top_feature_count = min(200, len(X.columns))

    for f in range(top_feature_count):
        d[label_column][f] = {X.columns[indices[f]]: importances[indices[f]]}

output = pd.DataFrame(d)
""")

# ## Transform merged dataset into Embedding Projector ready representation for clustering & dimensionality reduction exploration.

multiomicEmbeddingProjectorTensors = createOperation(op_name="multiomicEmbeddingProjectorTensors", input_sources = inputSources, op_type="python_script",
                                                     inputs = [multiomicNormalizedFeatures.name], expr=""" 
import os

# Create TF Embedding Projector metadata dataset, with human readable clinical columns.
metadata_columns = multiomicNormalizedFeatures.columns[multiomicNormalizedFeatures.columns.str.startswith("metadata_")].tolist() + \
                   multiomicNormalizedFeatures.columns[multiomicNormalizedFeatures.columns.str.startswith("label_")].tolist() + ["Biobank_Number"]
metadata_dataset = multiomicNormalizedFeatures.loc[:, metadata_columns]
metadata_tsv = metadata_dataset.to_csv(os.path.join(artifacts, 'CS_MT_embedding_projector_multiomic_metadata.tsv'), sep='\t', index=False, header=True)

# Create TF Embedding Projector tensor dataset for multi-omic dataset.
multiomic_train_dataset = multiomicNormalizedFeatures.select_dtypes(['number'])
multiomic_train_dataset = multiomic_train_dataset[multiomic_train_dataset.columns.drop(list(multiomic_train_dataset.filter(regex='label_')) + \
                                                                                       list(multiomic_train_dataset.filter(regex='clinical_')) + \
                                                                                       list(multiomic_train_dataset.filter(regex='surgery_embed_')) + \
                                                                                       list(multiomic_train_dataset.filter(regex='pathology_embed_')) + \
                                                                                       list(multiomic_train_dataset.filter(regex='chemotherapy_embed_')))]
dataset_tsv = multiomic_train_dataset.to_csv(os.path.join(artifacts, 'CS_MT_embedding_projector_multiomic_dataset.tsv'), sep='\t', na_rep='0.0', index=False, header=False)

# Create TF Embedding Projector tensor dataset for each single-omic dataset.
column_prefixes = ["clinical_", "surgery_embed_", "pathology_embed_", "chemotherapy_embed_", "plasma_protein_", "tissue_protein_", "plasma_lipid_", "pathology_NF", "CNV_", "freebayes_SNV_", "pindel_INDEL_", "AF4_", "rna_gene_expr_"]
for prefix in column_prefixes:
   singleomic_columns = multiomicNormalizedFeatures.columns[multiomicNormalizedFeatures.columns.str.contains(prefix)].tolist()
   singleomic_dataset = multiomicNormalizedFeatures.dropna(axis=0, how='all', subset=singleomic_columns)
   singleomic_metadata = singleomic_dataset.loc[:, metadata_columns]
   singleomic_train_dataset = singleomic_dataset.loc[:, singleomic_columns]
   singleomic_metadata_tsv = singleomic_metadata.to_csv(os.path.join(artifacts, f'CS_MT_embedding_projector_{prefix}metadata.tsv'), sep='\t', index=False, header=True)
   singleomic_dataset_tsv = singleomic_train_dataset.to_csv(os.path.join(artifacts, f'CS_MT_embedding_projector_{prefix}dataset.tsv'), sep='\t', na_rep='0.0', index=False, header=False)

output = multiomicNormalizedFeatures
"""
)

# # Add all workflow operations into a workflow DAG.

cedarsMTWorkflow.operations.extend([studyLabelsDedup, clinicalMetadataDedup, clinicalFeaturesDedup, notesNLPDedup,
                                    plasmaProteinFeaturesDedup, tissueProteinFeaturesDedup, plasmaLipidFeaturesDedup, pathologyFeaturesDedup, cnvFeaturesDedup, snvFeaturesDedup,
                                    indelsFeaturesDedup, rnaFusionsDedup, rnaGeneExprDedup,
                                    studyLabelsCleaned, clinicalMetadataCleaned, clinicalFeaturesCleaned,
                                    notesNLPCleaned, plasmaProteinCleaned, tissueProteinCleaned, plasmaLipidCleaned, pathologyCleaned, cnvCleaned, snvCleaned,
                                    indelsCleaned, rnaFusionsCleaned, rnaGeneExprCleaned, mergeLabelsMetadata,
                                    mergeClinical, mergeNotesNLP, mergePlasmaProtein, mergeTissueProtein, mergePlasmaLipid, mergePathology,
                                    mergeCNV, mergeSNV, mergeIndels, mergeRnaFusions, mergeRnaGeneExpr,
                                    multiomicNormalizedFeatures, multiomicPruneCorrelatedFeatures, multiomicEmbeddingProjectorTensors, #multiomicPredictiveBiomarkers,
                                   ])

# # Translate workflow into YAML for Argo execution.

yamlWorkflow = translation.construct_tabular_argo_workflow(cedarsMTWorkflow, inputSources, output=OUTPUT_DIR)
print(yamlWorkflow)

# # Print constructed workflow.

cedarsMTWorkflow

# # Serialize and persist Datasource Inputs and Workflow.

with open("../dataset_manager/sample_data/cedars_mt_multiomic_input_sources.json", "w") as f:
    f.write(json_format.MessageToJson(inputSources))
with open("../dataset_manager/sample_data/cedars_mt_multiomic_wf.json", "w") as f:
    f.write(json_format.MessageToJson(cedarsMTWorkflow))
with open("../dataset_manager/sample_data/cedars_mt_multiomic_wf.yaml", "w") as f:
    f.write(yamlWorkflow)

# # Execute Workflow

# Execute workflow on Argo framework.
submit_response = engine.submit_workflow(yamlWorkflow)

submit_response.text

response = engine.get_workflow_response(submit_response)

response.json()

artifact_list = engine.get_materialized_artifacts(cedarsMTWorkflow, response.json(), OUTPUT_DIR)
output_sample_sheets = engine.create_output_sample_sheets(artifacts=artifact_list, output=OUTPUT_DIR)
output_dict = engine.construct_output_dict(cedarsMTWorkflow, OUTPUT_DIR)

output_dict

# # Visualize merged dataset statistics.

multiomic_dataset = pd.read_csv(os.path.join(OUTPUT_DIR, "multiomicNormalizedFeatures.csv"))
multiomic_dataset

print(os.path.join(OUTPUT_DIR, "multiomicNormalizedFeatures.csv"))

# Show top multi-omic predictive biomarkers for each study objective
predictive_biomarkers = pd.read_csv(os.path.join(OUTPUT_DIR, "multiomicPredictiveBiomarkers.csv"))
predictive_biomarkers.head(50)

merged_statistics = tfdv.generate_statistics_from_dataframe(merged_dataset)
tfdv.visualize_statistics(merged_statistics)

# # Infer Schema from merged dataset.

infered_schema = tfdv.infer_schema(merged_statistics)
tfdv.display_schema(infered_schema)

# # Visualize prepared study dataset using facet-dive to slice and dice

dataset_json = merged_dataset
dataset_json = dataset_json.sample(n=5).to_json(orient='records')
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

# # Cluster prepared study dataset using embedding-projector to visualize, manipulate and analyze

artifacts = pd.read_csv(os.path.join(OUTPUT_DIR, "clinicalEmbeddingProjectorTensors-artifacts-sample-sheet.csv"))
artifacts

# Download pre-generated embedding projector dataset and metadata to local storage (due to tool's cross-site scripting work-around).
projector_dataset = pd.read_csv(os.path.join(OUTPUT_DIR, "CS_MT_embedding_projector_multiomic_dataset.tsv"))
projector_metadata = pd.read_csv(os.path.join(OUTPUT_DIR, "CS_MT_embedding_projector_multiomic_metadata.tsv"))
dataset_tsv = projector_dataset.to_csv('CS_MT_embedding_projector_multiomic_dataset.tsv', sep='\t', na_rep='0.0', index=False, header=False)
metadata_tsv = projector_metadata.to_csv('CS_MT_embedding_projector_multiomic_metadata.tsv', sep='\t', index=False, header=True)

# Embed embedding projector in an iFrame inside Jupyter Notebook
IFrame('https://projector.tensorflow.org/', width=1700, height=1000)

