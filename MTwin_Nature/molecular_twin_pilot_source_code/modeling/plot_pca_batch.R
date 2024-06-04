library(ggplot2)
library(dplyr)

pca_data <- readr::read_csv("projected_data.csv")
head(pca_data)
predictors <- readr::read_csv("predictive_labels.csv")
predictors <- predictors %>%
  mutate(label_outcome_categorical = as.factor(label_outcome_categorical)) %>%
  mutate(label_recurrence_binary = as.factor(label_recurrence_binary)) %>%
  mutate(label_outcome_binary = as.factor(label_outcome_binary))
head(predictors)

weird_batch <- c(
  "GI-14-389",
  "GI-16-234",
  "GI-16-444",
  "GI-16-664",
  "GI-17-182",
  "GI-17-513",
  "GI-17-537",
  "GI-17-816",
  "GI-18-352",
  "GI-19-234",
  "GI-14-881",
  "GI-17-404",
  "GI-18-611",
  "GI-18-665"
)
pca_data <- pca_data %>%
  mutate(t_sort_batch = case_when(
    sample %in% weird_batch ~ "NovaSeq",
    TRUE ~ "NextSeq"))

pca_data <- merge(pca_data,
                  predictors,
                  by.x = "sample",
                  by.y = "Biobank_Number")
head(pca_data)

labels <- c("t_sort_batch",
            "label_outcome_categorical",
            "label_recurrence_binary",
            "label_outcome_binary")
for (col in labels) {
  print(col)
  print(head(pca_data))
  plt <- ggplot(pca_data,
         aes_string("pca_x",
                    "pca_y",
                    label="sample",
                    color=col)) +
    geom_point() +
    geom_text(nudge_y=1, size = 3)
  path <- sprintf("rna_seq_batch_by_%s.png", col)
  ggsave(path, plot=plt, width=5, height=4)
}
