# Copyright (C) 2022 - Betteromics Inc.

# +
import datetime
import os

import dataset_manager.proto.workflow_pb2 as wf
import dataset_manager.workflow_engine.argo.engine as engine
import dataset_manager.workflow_engine.argo.translation as translation
import google.protobuf.json_format as json_format
import numpy as np
import pandas as pd
from dataset_manager.workflow_engine.utils import (createDirectorySource,
                                                   createOperation,
                                                   createSampleSheet,
                                                   createSource,
                                                   createWorkflow,
                                                   sanitize_colname)
from google.protobuf.json_format import MessageToJson
from IPython.core.display import HTML, display

display(HTML("<style>.container { width:95% !important; }</style>"))


# +
VCFWorkflow, inputSources = createWorkflow("cedars-genomic-workflow")
INPUT_DIR = "./molecular_twin_pilot/DNA/"
CNV_INPUT_DIR = "./molecular_twin_pilot/CNV/"
OUTPUT_DIR = "/molecular_twin_pilot/DNA/test-output/"


BED_FILE = "molecular_twin_pilot/clinical/xTv4_panel_probe_gene_targets.bed"


def custom_naming_function(s):
    return s.split("/")[-1].split(".")[0]


bed_file = createSource(source_name="bed_file", source_format=wf.Data.Format.CSV, source_loc_ref=BED_FILE)

somatic_freebayes_vcf = createDirectorySource(name="somatic_freebayes_vcf", s3_dir=INPUT_DIR, suffix="soma.freebayes.vcf",
                                              source_format=wf.Data.Format.CSV,
                                              custom_naming_function=custom_naming_function, sample_sheet=True)

somatic_pindel_vcf = createDirectorySource(name="somatic_pindel_vcf", s3_dir=INPUT_DIR, suffix="soma.pindel.vcf",
                                           source_format=wf.Data.Format.CSV,
                                           custom_naming_function=custom_naming_function, sample_sheet=True)

cnv_files = createDirectorySource(name="cnv_files", s3_dir=CNV_INPUT_DIR, suffix=".txt",
                                  source_format=wf.Data.Format.CSV,
                                  custom_naming_function=custom_naming_function, sample_sheet=True)


inputSources.input_sources.extend([somatic_freebayes_vcf, somatic_pindel_vcf, cnv_files, bed_file])


# -

def get_cnv_script():
    return """
set -eoux
cd /tmp

cat <<EOF>> parse_vcf.py
#!/usr/bin/python

import argparse
import pandas as pd

def process_cnv(input, output, bed_file):
    df = pd.read_csv(input, sep="\t")
    # Only count gain or loss CNVs
    df = df[df["GainLoss"] !="Neutral"]
    bed_file = pd.read_csv(bed_file, sep="\t", names=["Chr", "Start", "End", "gene"])

    # Merge bed file regions
    def f(gene_df, axis=None):
        min_start = gene_df.Start.min()
        max_end = gene_df.End.max()
        last_entry = gene_df.iloc[-1]
        last_entry.Start = min_start
        last_entry.End = max_end
        return last_entry
    merged_bed_file = bed_file.groupby("gene").agg(axis="columns", func=f).reset_index().sort_values(['Chr', 'Start'])


    merged_df = df.merge(merged_bed_file, how='left', on ='Chr', suffixes=("_cnv", "_bed"))

    # Filter for partial overlap
    partial_overlap = merged_df[((merged_df.Start_cnv <= merged_df.Start_bed)  & (merged_df.Start_bed <= merged_df.End_cnv)) | ((
        merged_df.Start_cnv <= merged_df.End_bed)  & (merged_df.End_bed <= merged_df.End_cnv))]

    final_df = partial_overlap[["gene", "Log2.Coverage.Ratio"]].pivot_table(index="gene", values="Log2.Coverage.Ratio").T
    final_df.columns = ["CNV_" + c for c in final_df.columns]
    final_df.to_csv(output, index=False)


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error {message}")
        self.print_help()
        sys.exit(2)
    
def main():
    parser = ArgParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--bed', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    
    process_cnv(input=args.input, bed_file=args.bed, output=args.output)   
if __name__ == "__main__":
    main()
EOF

python parse_vcf.py --input $INPUT --output $OUTPUT --bed {{bed_file}}

"""


def get_common_script(prefix):
    return """
set -eoux
cd /tmp

cat <<EOF>> parse_vcf.py
#!/usr/bin/python
import argparse
import pandas as pd
import vcf
from collections import Counter
def read(f: str):
    reader = vcf.Reader(open(f))
    df = pd.DataFrame([vars(r) for r in reader])
    # return empty df if no variants found
    if len(df) == 0:
        return df
    out = df.merge(pd.DataFrame(df.INFO.tolist()),
                   left_index=True, right_index=True)
    return out

def extract_gene_name(df: pd.DataFrame) -> dict:
    # ##INFO=<ID=ANN,Number=.,Type=String,Description="Functional annotations: 'Allele | Annotation | Annotation_Impact | Gene_Name | Gene_ID | Feature_Type | Feature_ID | Transcript_BioType | Rank | HGVS.c | HGVS.p | cDNA.pos / cDNA.length | CDS.pos / CDS.length | AA.pos / AA.length | Distance | ERRORS / WARNINGS / INFO' ">
    # First pass, just counts the number of variants associated with each 
    return dict(Counter(df["ANN"].map(lambda x: x[0]).str.split("|").str[4]))

def write_dict_to_csv(d: dict, output: str, prefix: str):
    prefixed_dict = {prefix + str(key): val for key, val in d.items()} 

    pd.DataFrame.from_dict(prefixed_dict,orient='index').T.to_csv(output, index=False)


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error {message}")
        self.print_help()
        sys.exit(2)
    
def main():
    parser = ArgParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    
    f = read(args.input)
    # Write empty csv if no variants found
    if len(f) == 0:
        f.to_csv(args.output, index=False)
        return 
    gene_count_dict = extract_gene_name(f)""" + f"""
    write_dict_to_csv(d=gene_count_dict, output=args.output, prefix="{prefix}")

if __name__ == "__main__":
    main()
EOF

# Few files fail with:
# SyntaxError: One of the FILTER lines is malformed: ##FILTER=All filters passed
# Remove Filter Line
cat $INPUT | grep -v "##FILTER" > working.vcf

python parse_vcf.py --input working.vcf --output $OUTPUT
"""


# +
process_freebayes_variants = createOperation(op_name="process-freebayes-variants", input_sources=inputSources, op_type="sharded_op",
                                             input_list="somatic_freebayes_vcf",
                                             resource_size=wf.Operation.OperationSize.SMALL, materialize_output=True,
                                             dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:cedars_genomics",
                                             expr=get_common_script("freebayes_SNV_")
                                             )


process_pindel_variants = createOperation(op_name="process-pindel-variants", input_sources=inputSources, op_type="sharded_op",
                                          input_list="somatic_pindel_vcf",
                                          resource_size=wf.Operation.OperationSize.SMALL, materialize_output=True,
                                          dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:cedars_genomics",
                                          expr=get_common_script("pindel_INDEL_")
                                          )

process_cnv_files = createOperation(op_name="process-cnv-files", input_sources=inputSources, op_type="sharded_op",
                                    input_list="cnv_files",
                                    reference_files=["bed_file"],
                                    resource_size=wf.Operation.OperationSize.SMALL, materialize_output=True,
                                    dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:cedars_genomics",
                                    expr=get_cnv_script()
                                    )

VCFWorkflow.operations.extend([process_freebayes_variants, process_pindel_variants, process_cnv_files])
# -


with open("./Cedars_Genomics_Sources.json", "w") as f:
    f.write(MessageToJson(inputSources, preserving_proto_field_name=True, including_default_value_fields=True))
with open("./Cedars_Genomics_wf.json", "w") as f:
    f.write(MessageToJson(VCFWorkflow, preserving_proto_field_name=True, including_default_value_fields=True))


# +
yaml_workflow = translation.construct_tabular_argo_workflow(VCFWorkflow, inputSources, output=OUTPUT_DIR)

submission = engine.submit_workflow(yaml_workflow)
submission.text
response = engine.get_workflow_response(submission)

# -

engine.get_materialized_artifacts(VCFWorkflow, response.json(), OUTPUT_DIR)

OUTPUT_DIR


