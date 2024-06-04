# DESeq2
# dplyr
# readr
# reshape2

library("tximport")
library("readr")
#library("tximportData")

# Define some string constants
SAMPLE <- "sample"
GENE_NAME <- "gene_name"
GENE_EXP <- "gene_expression"
CANCER <- "cancer"
NONCANCER <- "noncancer"

parse_args <- function() {
  parser <- argparse::ArgumentParser()
  # expected file format:
  # A1BG,A1CF,A2M, ...,sample
  # 3.254874,14.887599999999999,170.566297,...,"cancer-sample-1"
  parser$add_argument(
    "-c",
    "--cancers",
    required=TRUE,
    help="Cancer gene expression data"
  )
  # expected file format:
  # "GTEX_NORMAL_1","GTEX_NORMAL_2","GTEX_NORMAL_3"
  # "A1BG",2.441618,7.11739
  # "A1CF",0.418921300456528,0.54037560552484
  # "A2M",35.9645663,57.709455
  parser$add_argument(
    "-n",
    "--non-cancers",
    required=TRUE,
    help="Non-cancer gene expression data"
  )
  args <- parser$parse_args()
  return(args)
}

filter_sparse_data <- function(data, sparsity_cutoff, low_value_cutoff) {
  message("Removing low counts and data-sparse genes")
  # replace low values w/ NA
  data[data <= low_value_cutoff] <- NA
  nas_per_column <- apply(data, 2, function(col) {
    total_nas <- sum(is.na(col))
    return(total_nas/length(col))
  })
  keep_columns_index <- which(nas_per_column < sparsity_cutoff)
  message("Will keep columns: ", length(keep_columns_index))
  filtered_data <- data[, keep_columns_index, drop=FALSE]
  return(filtered_data)
}

get_uncorrelated_genes <- function(data, correlation_cutoff=0.80) {
  message("Looking for uncorrelated genes")
  `%>%` <- dplyr::`%>%`
  # correlations between columns, in chunks
  uncorrelated_genes <- vector()
  chunk_size <- 1000
  for (chunk in 1 : ceiling(ncol(data) / chunk_size) ) {
    leftmost_column <- (chunk - 1) * chunk_size + 1
    rightmost_column <- min(ncol(data), chunk * chunk_size)
    message("Chunk ", chunk, " [", leftmost_column, ";", rightmost_column, "]")
    data_subset <- data[, leftmost_column : rightmost_column, drop=FALSE]
    correlations <- cor(
      data_subset,
      method="spearman",
      use = "pairwise.complete.obs", # use only pairwise non-NA observations
    )
    correlations <- as.data.frame(correlations)
    correlations <- tibble::rownames_to_column(correlations, var=GENE_NAME)
    long_correlations <- reshape2::melt(
      correlations,
      id.vars=c(GENE_NAME),
      variable.name='second_gene_name',
      value.name='correlation'
    )
    # plt <- ggplot2::ggplot(long_correlations, ggplot2::aes(abs(correlation))) +
    #   ggplot2::geom_freqpoly(bins=1000)
    # fname <- sprintf("chunk_%s.pdf", chunk)
    # ggplot2::ggsave(fname, plot=plt)
    # keep only the genes with correlation lower than cutoff
    long_correlations <- long_correlations %>%
      dplyr::mutate(correlation = abs(correlation)) %>%
      dplyr::filter(correlation < correlation_cutoff)
    chunk_uncorrelated_genes <- unique(as.vector(long_correlations$second_gene_name))
    message("Uncorrelated genes in chunk: ", length(chunk_uncorrelated_genes))
    uncorrelated_genes <- c(uncorrelated_genes, chunk_uncorrelated_genes)
  }
  message("Uncorrelated genes: ", length(uncorrelated_genes))
  return(uncorrelated_genes)
}

filter_sparse_data_pivot <- function(data, sparsity_cutoff, low_value_cutoff) {
  message("Removing low counts and data-sparse genes")
  # replace low values w/ NA
  data[data <= low_value_cutoff] <- NA
  nas_per_row <- apply(data, 1, function(row) {
    total_nas <- sum(is.na(row))
    return(total_nas/length(row))
  })
  keep_rows_index <- which(nas_per_row < sparsity_cutoff)
  message("Will keep rows (genes): ", length(keep_rows_index))
  filtered_data <- data[keep_rows_index, , drop=FALSE]
  return(filtered_data)
}

prepare_cancers <- function(cancers_path, sparsity_cutoff=0.75, low_value_cutoff=2, correlation_cutoff=0.8) {
  `%>%` <- dplyr::`%>%`
  message("Reading cancer data")
  cancers <- readr::read_csv(
    cancers_path,
    col_names=TRUE,
  )
  # drop row indices that come from pandas
  cancers[['index']] <- NULL
  message("Cancers: ", nrow(cancers))
  
  # filter out genes that are missing data for more than 25% samples
  # -> 23611 genes
  cancers <- filter_sparse_data(cancers, sparsity_cutoff, low_value_cutoff)
  
  # replace NAs w/ 0 (0 count -- did not observe)
  cancers[is.na(cancers)] <- 0
  
  # print("Cancers matrix: ", dim(cancers))
  
  message("Genes in cancer sampes after removing sparse genes or low genes: ", ncol(cancers) - 1)
  n_cancers <- nrow(cancers)
  cancer_gene_set <- colnames(cancers)
  
  # remove tumor genes that are correlated
  # expression_values <- cancers %>%
  #   dplyr::select(-sample)
  # uncorrelated_genes <- get_uncorrelated_genes(expression_values, correlation_cutoff=correlation_cutoff)
  
  long_cancers <- reshape2::melt(
    cancers,
    id.vars=c(SAMPLE),
    variable.name=GENE_NAME,
    value.name=GENE_EXP
  )
  print(head(long_cancers))
  # add a generic label "cancer", only keep values for the uncorrelated genes
  long_cancers <- long_cancers %>%
    dplyr::mutate(label=CANCER)
    # dplyr::filter(gene_name %in% uncorrelated_genes)
  print(head(long_cancers))
  
  cancer_gene_set <- unique(long_cancers[[GENE_NAME]])
  message("Cancer genes after removing correlated genes: ", length(cancer_gene_set))
  # data is now in the format of:
  #   gene_name gene_expression  label  sample
  # 1      A1BG        3.254874 cancer   abc1
  # 2      A1CF       14.887600 cancer   abc1
  # 3       A2M      170.566297 cancer   abc1
  return(list(data=long_cancers, gene_set=cancer_gene_set))
}

prepare_noncancers <- function(noncancers_path, sparsity_cutoff=0.75, low_value_cutoff=2) {
  `%>%` <- dplyr::`%>%`
  # ATTENTION: reading only 5 rows for testing
  message("Reading non-cancer data")
  non_cancers <- readr::read_csv(
    noncancers_path,
  )
  
  # remove low counts, drop sparse genes
  non_cancers <- filter_sparse_data_pivot(non_cancers, sparsity_cutoff, low_value_cutoff)
  
  # replace NAs w/ 0 (0 count -- did not observe)
  non_cancers[is.na(non_cancers)] <- 0
  n_non_cancers <- ncol(non_cancers) - 1
  message("Non-cancers: ", n_non_cancers)
  message("Genes in non-cancer sampes: ", nrow(non_cancers))
  print(head(non_cancers))
  
  non_cancer_gene_set <- non_cancers[[GENE_NAME]]
  
  # pivot so that rows are samples, columns are genes
  long_non_cancers <- reshape2::melt(
    non_cancers,
    id.vars=c(GENE_NAME),
    variable.name=SAMPLE,
    value.name=GENE_EXP
  )
  long_non_cancers <- long_non_cancers %>%
    dplyr::mutate(label=NONCANCER)
  
  return(list(data=long_non_cancers, gene_set=non_cancer_gene_set))
}

main <- function() {
  args <- parse_args()
  `%>%` <- dplyr::`%>%`

  cancer_data <- prepare_cancers(args$cancers)
  noncancer_data <- prepare_noncancers(args$non_cancers)
  
  # find overlap between cancer and non-cancer gene sets
  overlapping_genes <- intersect(unique(cancer_data$gene_set), unique(noncancer_data$gene_set))
  message("Genes that overlap between cancers and non-cancers: ", length(overlapping_genes))
  
  # filter cancer and non-cancer data to only contain genes that are present in both set
  # TODO(darya, oggie): should we keep all genes instead?
  long_non_cancers <- noncancer_data$data %>%
    dplyr::filter(gene_name %in% overlapping_genes)
  
  # filter cancer genes to the same set
  long_cancers <- cancer_data$data %>%
    dplyr::filter(gene_name %in% overlapping_genes)
  
  # stitch together; pivot to wide where rows are samples, columns are genes
  all_samples <- rbind(long_cancers, long_non_cancers)
  gene_expression_estimates <- reshape2::dcast(
    all_samples,
    gene_name ~ sample + label,
    value.var=GENE_EXP,
  )
  # gene_expression_estimates data is now in the form of:
  #   gene_name GI-13-844_cancer GI-13-930_cancer ...
  # 1   A3GALT2         0.535891         0.429237 ...
  # 2      A1CF        14.878770         7.791570 ...
  # 3    A4GALT         5.915170        14.656810 ...
  # 4 ....
  readr::write_csv(gene_expression_estimates, "gene_expression_all_samples.csv", col_names=TRUE)
  
  gene_expression_estimates <- tibble::column_to_rownames(
    gene_expression_estimates,
    var=GENE_NAME
  )
  # gene_expression_estimates data is now in the form of:
  #         GI-13-844_cancer GI-13-930_cancer
  # A3GALT2         0.535891         0.429237
  # A1CF           14.878770         7.791570
  # A4GALT          5.915170        14.656810
  
  # assign random labels
  # labels <- data.frame(
  #   label=rep(c(CANCER, NONCANCER), ncol(gene_expression_estimates) / 2 + 1),
  #   stringsAsFactors=FALSE
  # )
  # labels <- labels[1:ncol(gene_expression_estimates), , drop=FALSE]

  labels <- data.frame(
    label=colnames(gene_expression_estimates),
    stringsAsFactors=FALSE
  )
  labels <- labels %>%
    dplyr::mutate(label = dplyr::if_else(grepl(NONCANCER, label), NONCANCER, CANCER)) %>%
    dplyr::mutate(label = factor(label, levels=c(NONCANCER, CANCER)))
  print(head(labels))
  print(dim(labels))
  
  # expects integer counts, so we need to round our floats:
  # as per https://www.biostars.org/p/429819/#438369
  gene_expression_estimates <- round(gene_expression_estimates)
  deseq_dataset <- DESeq2::DESeqDataSetFromMatrix(
    countData=gene_expression_estimates,
    colData=labels,
    design= ~ label,
  )
  differential_analysis <- DESeq2::DESeq(deseq_dataset)
  saveRDS(differential_analysis, file="differential_analysis_data.rds")
}

# invoke main
main()