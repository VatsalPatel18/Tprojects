# Read in abundances and a tx -> gene mapping, use tximport to go from
# transcripts to genes; then filter out genes w/ 0 expression; pivot the
# dataframe to be wide (genes as columns) and write that as output.
# Following this tutorial: https://bioconductor.org/packages/3.12/bioc/vignettes/tximport/inst/doc/tximport.html#kallisto

parse_args <- function() {
  parser <- argparse::ArgumentParser()
  # expected file format:
  # target_id       length  eff_length      est_counts      tpm
  # ENST00000631435.1       12      9       0       0
  # ENST00000434970.2       9       6       0       0
  parser$add_argument(
    "-i",
    "--input",
    required=TRUE,
    help="Transcript abundances in a tsv file"
  )
  # expected file format:
  # ENST00000456328.2       ENSG00000223972.5       DDX11L1
  # ENST00000450305.2       ENSG00000223972.5       DDX11L1
  # ENST00000488147.1       ENSG00000227232.5       WASH7P
  parser$add_argument(
    "-m",
    "--mapping",
    required=TRUE,
    help="Mapping from transcripts to genes"
  )
  args <- parser$parse_args()
  return(args)
}

main <- function() {
  args <- parse_args()
  `%>%` <- dplyr::`%>%`
  
  # read the mapping file, does not have a header
  mapping <- readr::read_tsv(
    args$mapping,
    col_names=FALSE,
    col_types = "ccc") # parse as strings
  # set column names
  colnames(mapping) <- c("transcript", "gene", "gene_name")
  # keep only the tx name and gene name
  tx2gene <- mapping %>%
    dplyr::select(transcript, gene_name)
  print(head(mapping))
  
  # create a list of files w/ a single file
  files <- c(args$input)
  
  txi <- tximport::tximport(
    files,
    type="kallisto",
    tx2gene=tx2gene,
    txOut=FALSE) # generate gene-level summaries
  message("Gene count: ", length(txi$abundance))
  
  # convert numerical vector to a data frame
  abundances <- data.frame(
    abundance=txi$abundance,
    stringsAsFactors=FALSE
  )
  abundances <- tibble::rownames_to_column(abundances, "gene_name")

  # remove 0s
  abundances <- abundances %>%
    dplyr::filter(abundance > 0)
  message("Genes w/ non-zero expression: ", nrow(abundances))
  
  # convert to a wide format where each column is a gene
  wide <- reshape2::dcast(
    abundances,
    . ~ gene_name,
    value.var="abundance"
  )
  # drop a silly artifact column
  wide['.'] <- NULL
  print(head(wide[, 1:10]))
  readr::write_csv(wide, "gene_level_abundances.csv")
}

# invoke main
main()
