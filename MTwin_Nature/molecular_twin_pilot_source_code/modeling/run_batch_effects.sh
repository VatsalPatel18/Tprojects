#!/bin/bash

set -exou pipefail

# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/multiomicNormalizedFeatures.csv .

# clinical features
# mkdir -p clinical
# python3 batch_effects.py -i clinicalNormalizedFeatures.csv \
#     -x ".*Histology_Behavior_ICD|.*diagnosis|.*outcome|.*Patient_History_of_Cancer_Seq.*|.*Biobank_Number|Sex.*" \
#     -o clinical \
#     -v "label_outcome_binary|label_tnm_stage"

# # # pathology data
# mkdir -p pathology
# # aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/pathology/pathologyFeatures.csv .
# python3 batch_effects.py \
#     -i pathologyFeatures.csv \
#     -x "Biobank_Number" \
#     -o pathology

# # # indel data
# mkdir -p indels
# # aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/DNA/test-output/process-pindel-variants.csv .
# python3 batch_effects.py \
#     -i process-pindel-variants.csv \
#     -x "Biobank_Number|sample" \
#     -o indels

# # # RNA-seq data
# mkdir -p rnaseq_tx
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/kallisto/quantify-paired-end.csv .
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/tx_to_gene/tx-to-gene-parsed.csv .
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/kallisto/quantify-paired-end.csv .
# mkdir -p rnaseq
# # aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/kallisto/quantify-paired-end.csv .
# python3 batch_effects.py \
#     -i quantify-paired-end.csv \
#     -x "Biobank_Number|sample|Unnamed" \
#     --log \
#     -o rnaseq

# mkdir -p rnaseq_multi
# python3 batch_effects.py -i multiomicNormalizedFeatures.csv \
#     -n "rna_gene_expr_.*" \
#     -o rnaseq_multi \
#     --log \
#     -v "label_outcome_binary|label_tnm_stage"

# # other data
# mkdir -p cnvs
# python3 batch_effects.py \
#     -i multiomicNormalizedFeatures.csv \
#     -n "CNV_.*" \
#     -v "label_outcome_binary|label_tnm_stage" \
#     -x "Biobank_Number|sample|clinical_.*|metadata_.*|pathology_.*|surgery_.*|chemotherapy_.*|protein_.*|freebayes_.*|pindel_.*|rna_gene_expr_.*|label_.*|AF4_.*" \
#     -o cnvs

# mkdir -p fusions
# # aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/multiomicPruneCorrelatedFeatures.csv .
# python3 batch_effects.py \
#     -i multiomicPruneCorrelatedFeatures.csv \
#     -n "^AF4_.*" \
#     -x "Biobank_Number|sample|clinical_.*|metadata_.*|pathology_.*|surgery_.*|chemotherapy_.*|protein_.*|freebayes_.*|pindel_.*|rna_gene_expr_.*|label_.*|CNV_.*" \
#     -v "label_outcome_binary|label_tnm_stage" \
#     -o fusions

# plasma proteins
mkdir -p plasma_protein
python3 batch_effects.py \
    -i multiomicNormalizedFeatures.csv \
    -n "protein_.*" \
    -v "label_outcome_binary|label_tnm_stage" \
    -x "Biobank_Number|sample|clinical_.*|metadata_.*|pathology_.*|surgery_.*|chemotherapy_.*|freebayes_.*|pindel_.*|rna_gene_expr_.*|label_.*|AF4_.*|CNV_.*" \
    -o plasma_protein
