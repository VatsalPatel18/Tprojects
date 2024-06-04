# +
# https://github.com/grailbio/bio/tree/master/fusion


# wget http://hgdownload.cse.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz
# wget ftp://ftp.ebi.ac.uk/pub/databases/gencode/Gencode_human/release_26/gencode.v26.annotation.gtf.gz

# bio-fusion -generate-transcriptome -output gencode.v26.whole_genes.fa -keep-mitochondrial-genes -keep-readthrough-transcripts -keep-pary-locus-transcripts -keep-versioned-genes gencode.v26.annotation.gtf hg38.fa

# fasta file: ./molecular_twin_pilot/bio-fusion-reference-files/gencode.v26.whole_genes.fa


# #!wget https://fusionhub.persistent.co.in/out/global/Fusionhub_global_summary.txt
# import pandas as pd
# df = pd.read_csv("Fusionhub_global_summary.txt", sep="\t")
# df['Genes'] = df["Fusion_gene"].str.replace("--", "/")
# df = df[df["COSMIC"] == "+"]
# df[["Genes"]].to_csv("./molecular_twin_pilot/bio-fusion-reference-files/fusion_hub_gene_pairs.tsv", sep="\t", index=False)


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
VCFWorkflow, inputSources = createWorkflow("cedars-rna-fusion-workflow")
OUTPUT_DIR = "./molecular_twin_pilot/RNA/test-output/"


def custom_naming_function(s):
    return s.split("/")[-1].split(".")[0]


transcriptome = createSource(source_name="transcriptome",
                             source_format=wf.Data.Format.CSV,
                             source_loc_ref="./molecular_twin_pilot/bio-fusion-reference-files/gencode.v26.whole_genes.fa")
cosmic_fusion = createSource(source_name="cosmic_fusion",
                             source_format=wf.Data.Format.CSV,
                             source_loc_ref="./molecular_twin_pilot/bio-fusion-reference-files/fusion_hub_gene_pairs.tsv")


rna_fusion = createDirectorySource(name="rna_fusion",
                                   s3_dir="s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/RNA/",
                                   suffix=".tar.gz",
                                   source_format=wf.Data.Format.CSV,
                                   custom_naming_function=custom_naming_function, sample_sheet=True)


inputSources.input_sources.extend([transcriptome, rna_fusion, cosmic_fusion])


# -

def get_rna_fusion_script():
    return """
set -eoux
cd /tmp

cat <<EOF>> parse_rna_fusion_script.py
#!/usr/bin/python

import argparse
import pandas as pd

from collections import defaultdict 


class ArgParser(argparse.ArgumentParser):
    def error(self, message):
        sys.stderr.write(f"error {message}")
        self.print_help()
        sys.exit(2)


def process_rna_fusion(input, output):

    gene_counts = defaultdict(int)
    with open(input, "r") as f:
        while True:
            line = f.readline()
            if not line:
                break
            if line.startswith(">"):
                gene_counts[line.split("|")[1]]+=1
    prefixed_dict = {"AF4_" + str(key): val for key, val in gene_counts.items()} 
    pd.DataFrame(prefixed_dict, index=[0]).to_csv(output, index=False)


def main():
    parser = ArgParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)

    args = parser.parse_args()
    
    process_rna_fusion(input=args.input, output=args.output)   
if __name__ == "__main__":
    main()
    
EOF

FASTQ_1=$(ls $INPUT/*_[1].fastq.gz)
FASTQ_2=$(ls $INPUT/*_[23].fastq.gz)

bio-fusion \
    -r1 $FASTQ_1 \
    -r2 $FASTQ_2 \
    -transcript={{transcriptome}} \
    -cosmic-fusion={{cosmic_fusion}}
    
python parse_rna_fusion_script.py --input all-outputs.fa --output $OUTPUT
cp *.fa  $ARTIFACTS/

"""


# +

process_rna_fusion_files = createOperation(op_name="process-rna-fusion", input_sources=inputSources, op_type="sharded_op",
                                           input_list="rna_fusion",
                                           reference_files=["transcriptome", "cosmic_fusion"],
                                           resource_size=wf.Operation.OperationSize.MEDIUM, materialize_output=True,
                                           dockerImage="965350412536.dkr.ecr.us-west-2.amazonaws.com/betteromics-python-slim:cedars-bio-fusion",
                                           expr=get_rna_fusion_script()
                                           )


# process_freebayes_variants, process_pindel_variants, process_cnv_files])
VCFWorkflow.operations.extend([process_rna_fusion_files])
# -


with open("./Cedars_RNA_Fusion_Sources.json", "w") as f:
    f.write(MessageToJson(inputSources, preserving_proto_field_name=True, including_default_value_fields=True))
with open("./Cedars_RNA_Fusion_wf.json", "w") as f:
    f.write(MessageToJson(VCFWorkflow, preserving_proto_field_name=True, including_default_value_fields=True))


# +
yaml_workflow = translation.construct_tabular_argo_workflow(VCFWorkflow, inputSources, output=OUTPUT_DIR)

submission = engine.submit_workflow(yaml_workflow)
submission.text
response = engine.get_workflow_response(submission)
# -

engine.get_materialized_artifacts(VCFWorkflow, response.json(), OUTPUT_DIR)



