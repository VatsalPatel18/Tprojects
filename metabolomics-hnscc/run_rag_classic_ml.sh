#!/bin/bash

# Default parameters
DATA_DIR="./data"
RESULT_DIR="./final_result"
PRELIM_RESULT_DIR="./prelim_result"
CLINICAL_FILE="./data/survival.data.csv"
LATENT_DIM=2
EPOCHS=10
BATCH_SIZE=32
AUTOENCODER_MODEL_DIR="$RESULT_DIR/autoencoder"
CLUSTER_MODEL_DIR="$RESULT_DIR/clustering_model"
EVALUATION_RESULT_DIR="./evaluation_results"
CLUSTER_PREDICTION_DIR="./cluster_predictions"

# Custom parameters from command-line arguments
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --data_dir) DATA_DIR="$2"; shift ;;
        --result_dir) RESULT_DIR="$2"; shift ;;
        --prelim_result_dir) PRELIM_RESULT_DIR="$2"; shift ;;
        --clinical_file) CLINICAL_FILE="$2"; shift ;;
        --latent_dim) LATENT_DIM="$2"; shift ;;
        --epochs) EPOCHS="$2"; shift ;;
        --batch_size) BATCH_SIZE="$2"; shift ;;
        --autoencoder_model_dir) AUTOENCODER_MODEL_DIR="$2"; shift ;;
        --cluster_model_dir) CLUSTER_MODEL_DIR="$2"; shift ;;
        --evaluation_result_dir) EVALUATION_RESULT_DIR="$2"; shift ;;
        --cluster_prediction_dir) CLUSTER_PREDICTION_DIR="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

# Step 1: Run Autoencoder
echo "Running autoencoder..."
rag-classic-ml autoencoder \
    --data "$DATA_DIR/metabolic_genes.csv" \
    --sampleID 'PatientID' \
    --output_dir "$RESULT_DIR" \
    --prelim_output "$PRELIM_RESULT_DIR" \
    --latent_dim "$LATENT_DIM" \
    --epochs "$EPOCHS" \
    --batch_size "$BATCH_SIZE"

# Step 2: Run Survival Clustering
echo "Running survival clustering..."
rag-classic-ml survival_clustering \
    --data_path "$RESULT_DIR/latent_features.csv" \
    --clinical_df_path "$CLINICAL_FILE" \
    --save_dir "$RESULT_DIR"

# Step 3: Evaluate Autoencoder
echo "Evaluating autoencoder..."
rag-classic-ml evaluate_autoencoder \
    --data "$DATA_DIR/metabolic_genes.csv" \
    --sampleID 'PatientID' \
    --model_dir "$AUTOENCODER_MODEL_DIR" \
    --output_dir "$EVALUATION_RESULT_DIR" \
    --batch_size "$BATCH_SIZE"

# Step 4: Predict Clusters
echo "Predicting clusters..."
rag-classic-ml predict_clusters \
    --data "$EVALUATION_RESULT_DIR/latent_features.csv" \
    --model_dir "$CLUSTER_MODEL_DIR" \
    --clinical_data "$CLINICAL_FILE" \
    --output_dir "$CLUSTER_PREDICTION_DIR"

echo "All steps completed."
