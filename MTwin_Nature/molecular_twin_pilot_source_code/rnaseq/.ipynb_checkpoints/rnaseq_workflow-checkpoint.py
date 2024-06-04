# +
import os

import json

from common.common import better_home
import dataset_manager.proto.workflow_pb2 as wf
import dataset_manager.workflow_engine.argo.engine as engine
import dataset_manager.workflow_engine.argo.translation as translation
from dataset_manager.workflow_engine.utils import (createDirectorySource,
                                                   createOperation,
                                                   createSource,
                                                   createWorkflow)


# +
PREFIX = 's3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA'
workflow_graph, inputSources = createWorkflow("rnaseq-workflow")
INPUT_DIR = f'{PREFIX}/'
OUTPUT_DIR = f'{PREFIX}/kallisto/'
KALLISTO_DOCKER_IMAGE = '965350412536.dkr.ecr.us-west-2.amazonaws.com/kallisto:cedars_rna_seq_v2'
TRANSCRIPT_PATH = 's3://betteromics-data-platform/references/kallisto/homo_sapiens/transcriptome.idx'
CONVERTER_PATH = 's3://betteromics-data-platform/references/tsv_to_csv.py'


def custom_naming_function(s):
    return s.split("/")[-1].split(".")[0]


def get_script(script_path):
    with open(script_path, 'r') as f:
        script_code = f.read()
        return script_code


transcriptome_index = createSource(
    source_name="transcriptome_index",
    source_format=wf.Data.Format.CSV,  # format does not matter here?
    source_loc_ref=TRANSCRIPT_PATH)

# can probably link to a (customer) git repo instead of s3?
tsv_to_csv = createSource(
    source_name="tsv_to_csv",
    source_format=wf.Data.Format.CSV,  # format does not matter here?
    source_loc_ref=CONVERTER_PATH)

fastq_files = createDirectorySource(
    # num_samples=20,  # fewer samples for testing
    name="fastq_files",
    s3_dir=INPUT_DIR,
    suffix=".tar.gz",
    source_format=wf.Data.Format.CSV,
    custom_naming_function=custom_naming_function,
    sample_sheet=True)

inputSources.input_sources.extend([fastq_files, transcriptome_index, tsv_to_csv])


# +
quant_script_path = os.path.join(
    better_home(),
    'root',
    'dataset_manager',
    'external_projects',
    'cedars',
    'rnaseq',
    'quantify_kallisto.sh')

quantify_rnaseq_paired_end = createOperation(
    op_name="quantify-paired-end",
    input_sources=inputSources,
    reference_files=['transcriptome_index', 'tsv_to_csv'],
    op_type="sharded_op",
    input_list="fastq_files",
    resource_size=wf.Operation.OperationSize.MEDIUM,
    materialize_output=True,
    dockerImage=KALLISTO_DOCKER_IMAGE,
    expr=get_script(quant_script_path)
)

workflow_graph.operations.extend(
    [
        quantify_rnaseq_paired_end
    ]
)


# +
yaml_workflow = translation.construct_tabular_argo_workflow(
    workflow_graph,
    inputSources,
    output=OUTPUT_DIR)
base_path = os.path.join(
    better_home(),
    'root',
    'dataset_manager',
    'external_projects',
    'cedars',
    'rnaseq')
with open(os.path.join(base_path, "rnaseq_input_sources.json"), 'w') as f:
    f.write(json_format.MessageToJson(inputSources))
with open(os.path.join(base_path, "rnaseq_workflow.json"), 'w') as f:
    f.write(json_format.MessageToJson(workflow_graph))
with open(os.path.join(base_path, "rnaseq_workflow.yaml"), 'w') as f:
    f.write(yamlWorkflow)

submission = engine.submit_workflow(yaml_workflow)
# submission.text
response = engine.get_workflow_response(submission)
# -

engine.get_materialized_artifacts(workflow_graph, response.json(), OUTPUT_DIR)

OUTPUT_DIR
