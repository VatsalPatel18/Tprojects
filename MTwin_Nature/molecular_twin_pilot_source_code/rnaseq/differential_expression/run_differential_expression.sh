#/bin/bash
set -exou pipefail

# download cancer data
CANCERS=tx-to-gene-parsed.csv
# aws s3 cp s3://betteromics-data-platform/data_manager/cedars_sinai/molecular_twin_pilot/outputs/RNA/tx_to_gene/$CANCERS .

# non-cancer data is in the email from Ren
NONCANCERS="~/Downloads/noncancer_gene_level_abundances.csv"

time Rscript differential_expression.R -c $CANCERS -n $NONCANCERS > run_diff_exp.log 2>&1

time Rscript select_genes.R > select_genes.log 2>&1
