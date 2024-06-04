#!/bin/bash

set -exu

# $INPUT is already unzipped from tar.gz to *fastq.gz files

FILE1=$(ls $INPUT/*_1.fastq.gz)
FILE2=$(ls $INPUT/*_[23].fastq.gz)
time kallisto quant \
        -i {{transcriptome_index}} \
        -t 4 \
        -b 100 \
        -o $ARTIFACTS \
        $FILE1 $FILE2

ls -l $ARTIFACTS

echo "TEST\nTEST2" > $OUTPUT

# convert tsv to csv, pivot
python {{tsv_to_csv}} $ARTIFACTS/abundance.tsv $OUTPUT

# csvtk abundances.tsv abundances.csv