# Codebase Layout

Raw study data pre-processing and pipelines for respective single-omic data sources. RNA, DNA and Pathology pipelines with estimated 2-6hrs runtime on 8 m5.xlarge AWS instances.
```
./MTPilot_Genomic_data_processing.py
./MTPilot_clinical_data_processing.py
./MTPilot_differential_expression_correlations_multiomic_merge.py
./MTPilot_merge_multiomic_dataset.py
./MTPilot_pathology_processing.py
./MTPilot_protein_tumor_normal_differential_expression.py
./MTPilot_rna_fusions_processing.py
./rnaseq/*
```

Study model training pipeline with cross-validation using merged MTPilot multiomic dataset; estimated runtime ~4-5hrs on m5.2xlarge AWS instance.
```
./MTPilot_multiomic_model_training.py
./modeling/*
```

Study Analysis and Validation source code from MTPilot merged study dataset and TCGA.

```
./MTPilot_generate_study_analysis_plots.py
./MTPilot_parsimonious_curves_compare_with_TCGA.py
./MTPilot_recursive_feature_elimination_parsimonious_plots.py
./MTPilot_validation_train_on_MTPilot_predict_on_TCGA.py
``` 

# Environment Setup
```
# create new python environment
$ conda create --name py38 python=3.8.13

# activate python env
$ conda activate py38
```

# Create source code directory and unzip source code
```
$ mkdir molecular_twin_pilot_study
$ cd molecular_twin_pilot_study
$ unzip molecular_twin_pilot_source_code.zip
```

# Setup Python dependencies
```
$ pip install -r requirements.txt
```


# Download study dataset
```
# 1) Login/Register with NIH at https://www.ncbi.nlm.nih.gov/bioproject/
# 2) Download study dataset via download link (https://www.ncbi.nlm.nih.gov/bioproject/PRJNA889519) or search for BioProject Accession Number: PRJNA889519
# 3) Stage study data inside code repository as sub-directory 'molecular_twin_study'
```

# Run study analyses
```
# Train cross-validated MTPilot disease survival and disease recurrence models
$ python MTPilot_multiomic_model_training.py

# Generate study analysis and plots
$ python MTPilot_generate_study_analysis_plots.py

# Train disease survival on MTPilot dataset, predict/validate on TCGA
$ python MTPilot_validation_train_on_MTPilot_predict_on_TCGA.py

# Create parsimonous model learning curves using recursive feature elimination for MTPilot
$ python MTPilot_recursive_feature_elimination_parsimonious_plots.py

# Create parsimonious model learning curves using recursive feature elimination for TCGA and MTPilot
# Set MTP_OR_TCGA='MTPilot' to generate MTPilot results; set MTP_OR_TCGA='TCGA' to generate TCGA results
$ python MTPilot_parsimonious_curves_compare_with_TCGA.py

```
