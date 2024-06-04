# convert tsv to csv
import pandas as pd
import sys


def convert(input_path: str, output_path: str) -> None:
    '''
    Input file format:

target_id       length  eff_length      est_counts      tpm
ENST00000631435.1       12      9       0       0
ENST00000434970.2       9       6       0       0
ENST00000448914.1       13      10      0       0

    Output file format:
ENST00000631435.1,ENST00000434970.2
0,0
    '''
    data_frame = pd.read_table(input_path, sep='\t')
    # remove transcripts w/ counts below 1 (assume that should have observed at least 1 transcript)
    data_frame = data_frame[data_frame['est_counts'] > 0]
    # keep only the TPM -- transcripts per million since it normalizes for the
    # length of a transcript
    wide_data = pd.pivot_table(
        data_frame,
        values='tpm',
        columns='target_id')
    wide_data.to_csv(output_path, sep=',', index=False)


if __name__ == "__main__":
    # expects two command line parameters -- (1) path to tsv and (2) path to csv
    convert(sys.argv[1], sys.argv[2])
