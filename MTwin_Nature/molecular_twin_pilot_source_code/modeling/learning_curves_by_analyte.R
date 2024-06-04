library(dplyr)
library(ggplot2)

drop_feature_columns <- function(df) {
  col_idx <- grep("top_10_*|X1", names(df))
  df <- df[, -col_idx]
  return(df)
}

prepare_data <- function(df) {
  df <- drop_feature_columns(df)
  print(table(df$model_type))
  df <- df %>%
    mutate(num_analytes = stringr::str_count(feature_prefix, "'") / 2) %>%
    filter(num_analytes == 1) %>%
    filter(grepl("\\(", feature_prefix)) %>%
    filter(target_label != 'label_tnm_stage') %>%
    # filter(model_type == 'SVM_Model' | model_type == 'PCA_LR_Model') %>%
    # filter(model_type == 'SVM_Model') %>%
    mutate(analyte = gsub("\\[|\\]|'|\\(|\\)|,", "", feature_prefix)) %>%
    mutate(analyte = gsub("_$", "", analyte)) %>%
    rename(acc = test_loo_accuracy) %>%
    select(analyte, num_samples, target_label, TP, FP, FN, TN, acc, F1, model_type) %>%
    # compute our own F1
    mutate(F1 = 2 * TP / (2 * TP + FP + FN)) %>%
    select(analyte, num_samples, target_label, acc, F1, model_type)
  print(head(df))
  print(unique(df$analyte))
  # rename analytes
  df <- df %>%
    mutate(analyte = case_when(
      analyte == 'clinical' ~ "CLIN",
      analyte == 'pindel_INDEL' ~ "INDELs",
      analyte == 'pathology_embed' ~ "PATH",
      analyte == 'rna_gene_expr' ~ "EXPR",
      analyte == 'freebayes_SNV' ~ "SNVs",
      analyte == 'tissue_protein' ~ "TISS_PROT",
      analyte == 'plasma_protein' ~ "PLASM_PROT",
      analyte == 'pathology_NF' ~ "PATH_NF",
      analyte == 'tissue_lipid' ~ "LIPIDS",
      analyte == 'surgery_embed' ~ "SURG",
      analyte == 'chemotherapy_embed' ~ "CHEMO",
      analyte == 'AF4' ~ "FUSION",
      TRUE ~ as.character(analyte)
    ))
  print(unique(df$analyte))
  # filter out non-molecular analytes
  df <- df %>%
    filter(analyte != 'CLIN') %>%
    filter(analyte != 'PATH') %>%
    filter(analyte != 'SURG') %>%
    filter(analyte != 'CHEMO')
  print("Model names")
  print(unique(df$model_type))
  # rename models
  df <- df %>%
    mutate(model_type = case_when(
      model_type == 'SVM_Model' ~ "SVM",
      model_type == 'PCA_LR_Model' ~ "PCA+LR",
      model_type == 'L1_Norm_SVM_Model' ~ "L1_SVM",
      model_type == 'L1_Norm_RF_Model' ~ "L1_RF",
      model_type == 'L1_Norm_MLP_Model' ~ "L1_MLP",
      TRUE ~ as.character(model_type)
    ))
  return(df)
}

smaller_data <- readr::read_csv("smaller_data.csv")
dim(smaller_data)

larger_data  <- readr::read_csv("larger_data.csv")
dim(larger_data)

smaller_data <- prepare_data(smaller_data)
larger_data <- prepare_data(larger_data)

joined_data <- rbind(smaller_data, larger_data)
readr::write_csv(joined_data, "learning_curves_plot_data.csv")

# joined_data <- joined_data %>%
#  mutate(analyte = factor(as.character(analyte),
#                             ordered = TRUE,
#                             levels = c("CNV", "LIPIDS", "PATH_NF", "SNVs",
#                                        "PLASM_PROT", "TISS_PROT", "EXPR", "FUSION",
#                                        "INDELs")))

plt <- ggplot(joined_data, aes(num_samples, F1, color = target_label)) +
  geom_line() +
  geom_point() +
  # facet_wrap(vars(analyte)) +
  facet_grid(model_type ~ analyte, scales="free_x") +
  labs(title="Model performance with increased sample size, per analyte")

ggsave("learning_curves_by_analyte.png", plot = plt, width=10, height=5)