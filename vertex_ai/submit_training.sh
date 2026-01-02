#!/bin/bash
# Script to submit training job to Vertex AI
# Usage: ./submit_training.sh [config_file] [job_name]

set -e

# Configuration
PROJECT_ID="${GOOGLE_CLOUD_PROJECT:-your-project-id}"
REGION="${GOOGLE_CLOUD_REGION:-us-central1}"
IMAGE_URI="${IMAGE_URI:-gcr.io/${PROJECT_ID}/tablesense2:latest}"
CONFIG_FILE="${1:-configs/detector_config_minimal.yaml}"
JOB_NAME="${2:-detector-training-$(date +%Y%m%d-%H%M%S)}"
MACHINE_TYPE="${MACHINE_TYPE:-n1-standard-4}"
GPU_TYPE="${GPU_TYPE:-NVIDIA_TESLA_T4}"
GPU_COUNT="${GPU_COUNT:-1}"

echo "Submitting Vertex AI training job..."
echo "  Project: ${PROJECT_ID}"
echo "  Region: ${REGION}"
echo "  Job Name: ${JOB_NAME}"
echo "  Config: ${CONFIG_FILE}"
echo "  Machine: ${MACHINE_TYPE}"
echo "  GPU: ${GPU_COUNT}x ${GPU_TYPE}"

# Submit the training job
gcloud ai custom-jobs create \
  --project="${PROJECT_ID}" \
  --region="${REGION}" \
  --display-name="${JOB_NAME}" \
  --worker-pool-spec=machine-type="${MACHINE_TYPE}",replica-count=1,container-image-uri="${IMAGE_URI}",accelerator-type="${GPU_TYPE}",accelerator-count="${GPU_COUNT}" \
  --args="--config=${CONFIG_FILE}"

echo "Job submitted! Monitor it at:"
echo "https://console.cloud.google.com/vertex-ai/training/custom-jobs?project=${PROJECT_ID}"
