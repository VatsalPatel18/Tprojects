library("apeglm")
library("ggplot2")
library("dplyr")

GENE_NAME <- "gene_name"

`%>%` <- dplyr::`%>%`

differential_analysis <- readRDS(file="differential_analysis_data.rds")
names <- DESeq2::resultsNames(differential_analysis)
print(names)
res <- DESeq2::results(differential_analysis)
length(res$padj)
        
resLFC <- DESeq2::lfcShrink(
  differential_analysis, 
  coef=names[2], 
  type="apeglm"
)

adjusted_p_values <- data.frame(
  p_value=c(res$padj, resLFC$padj),
  gene_name=c(rownames(res), rownames(resLFC)),
  label=c(rep("before", length(res$padj)), rep("after", length(resLFC$padj))),
  stringsAsFactors=FALSE)

ggplot2::ggplot(adjusted_p_values, aes(p_value, color=label)) +
  geom_freqpoly(alpha = 0.5, bins=1000) +
  xlim(0, 0.0001) +
  labs(title="Distro of adjusted p-value before/after shrinkage")

# genes before adjusting for low counts
DESeq2::plotMA(res, ylim=c(-2,2))
# genes after adjusting for low counts
DESeq2::plotMA(resLFC, ylim=c(-2,2))

plot_data <- DESeq2::plotCounts(
  differential_analysis, 
  gene=which.min(res$padj), 
  intgroup="label", 
  returnData=TRUE
)
ggplot2::ggplot(plot_data, ggplot2::aes(x=label, y=count)) + 
  ggplot2::geom_point(position=ggplot2::position_jitter(w=0.1,h=0)) + 
  ggplot2::scale_y_log10(breaks=c(25,100,400))

# select a 1000 most significant genes
shrunk_adjusted_p_values <- tibble::rownames_to_column(as.data.frame(resLFC), var=GENE_NAME)
shrunk_adjusted_p_values <- shrunk_adjusted_p_values %>%
  dplyr::arrange(padj)
top_n <- 2000
most_significant <- shrunk_adjusted_p_values[1:top_n, ]
readr::write_csv(most_significant, sprintf("differentially_expressed_genes_%s.csv", top_n))

ggplot2::ggplot(most_significant, 
                ggplot2::aes(baseMean, log2FoldChange, size=-log10(padj), label=gene_name)) +
  ggplot2::geom_point(alpha=0.1) +
  ggplot2::scale_x_log10() +
  ggplot2::geom_text(size=3, alpha=0.4, nudge_y = 0.3) +
  ggplot2::labs(title=sprintf("%s most significant differentially expressed genes", top_n))
