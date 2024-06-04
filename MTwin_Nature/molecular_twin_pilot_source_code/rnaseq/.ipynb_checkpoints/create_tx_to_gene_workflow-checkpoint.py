# +
import os

import json

from common.common import better_home
import data_factory.api.test_client.workflow_utils as workflow_utils
import dataset_manager.proto.workflow_pb2 as wf
import dataset_manager.workflow_engine.argo.engine as engine
import dataset_manager.workflow_engine.argo.translation as translation
from dataset_manager.workflow_engine.utils import (createDirectorySource,
                                                   createOperation,
                                                   createSource,
                                                   createWorkflow)


PREFIX = 's3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA'
OUTPUT_DIR = f'{PREFIX}/tx_to_gene/'
BIOR_DOCKER_IMAGE = '965350412536.dkr.ecr.us-west-2.amazonaws.com/tximport:f749dca5-dirty-11'
TX_TO_GENES_PATH = 's3://betteromics-data-platform/references/kallisto/homo_sapiens/transcripts_to_genes.txt'
CONVERTER_PATH = 's3://betteromics-data-platform/references/'
ABUNDANCES_DIR = "s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/kallisto/quantify-paired-end/"


def custom_naming_function(path):
    """
    Extract sample name from the s3 path
    """
    sample_name = path.split("/")[-2]
    return sample_name.split("_")[0]


################################################################################
# create a new workflow
# ###############################################################################
workflow_graph, inputSources = createWorkflow("tx-to-gene-workflow")

tx_mapping = createSource(
    source_name="tx_mapping",
    source_format=wf.Data.Format.CSV,  # format does not matter and not enforced
    source_loc_ref=TX_TO_GENES_PATH)

# upload R script to s3 and will use it as an reference file
script_path = os.path.join(
    better_home(),
    "root",
    "dataset_manager",
    "external_projects",
    "cedars",
    "rnaseq",
    "tx_to_genes.R")
fname = os.path.basename(script_path)
CONVERTER_PATH = f's3://betteromics-data-platform/references/{fname}'
workflow_utils.write_file_to_s3(script_path, CONVERTER_PATH)
# can probably link to a (customer) git repo instead of s3?
aggregate_script = createSource(
    source_name="tx_to_gene_script",
    source_format=wf.Data.Format.CSV,  # format does not matter and not enforced
    source_loc_ref=CONVERTER_PATH)

abundance_files = createDirectorySource(
    # num_samples=20,  # fewer samples for testing
    name="abundance_files",
    s3_dir=ABUNDANCES_DIR,
    suffix=".tsv",
    source_format=wf.Data.Format.CSV,
    custom_naming_function=custom_naming_function,
    sample_sheet=True)

inputSources.input_sources.extend([abundance_files, tx_mapping, aggregate_script])

sharded_tx_to_gene_op = createOperation(
    op_name="tx-to-gene",
    input_sources=inputSources,
    reference_files=["tx_mapping", "tx_to_gene_script"],
    op_type="sharded_op",
    input_list="abundance_files",
    resource_size=wf.Operation.OperationSize.SMALL,
    materialize_output=True,
    dockerImage=BIOR_DOCKER_IMAGE,
    expr="""
#!/bin/bash

set -exu

ls $INPUT
cp $INPUT abundance.tsv
Rscript {{tx_to_gene_script}} -i abundance.tsv -m {{tx_mapping}}

mv gene_level_abundances.csv $OUTPUT
"""
)

workflow_graph.operations.extend(
    [
        sharded_tx_to_gene_op
    ]
)


# +
yaml_workflow = translation.construct_tabular_argo_workflow(
    workflow_graph,
    inputSources,
    output=OUTPUT_DIR)

workflow_path = os.path.join(
    better_home(),
    'root',
    'dataset_manager',
    'external_projects',
    'cedars',
    'rnaseq',
    'tx_to_genes_workflow.yaml')
with open(workflow_path, 'w') as f:
    f.write(yaml_workflow)

submission = engine.submit_workflow(yaml_workflow)
# submission.text
response = engine.get_workflow_response(submission)
engine.get_materialized_artifacts(workflow_graph, response.json(), OUTPUT_DIR)

""
OUTPUT_DIR

""

