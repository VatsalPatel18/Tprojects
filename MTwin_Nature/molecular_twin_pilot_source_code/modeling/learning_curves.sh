#!/bin/bash

# data from Jan 2021
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=True_results.csv smaller_data.csv
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=False_results.csv larger_data.csv

# data from Feb 18 2021
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=True_results_corr_pruned.csv smaller_data.csv
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=False_results_corr_pruned.csv larger_data.csv

aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=True_results_pair_wise.csv smaller_data.csv
aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/multiomic/categorical_model_grid_search_intersection=False_results_pair_wise.csv larger_data.csv

Rscript learning_curves_by_analyte.R
