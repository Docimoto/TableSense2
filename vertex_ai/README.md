# Vertex AI Training Setup

This directory contains files and instructions for running training on Google Vertex AI GPUs.

## Prerequisites

1. **Google Cloud SDK**: Install and authenticate
   ```bash
   gcloud auth login
   gcloud config set project YOUR_PROJECT_ID
   ```

2. **Enable APIs**:
   ```bash
   gcloud services enable aiplatform.googleapis.com
   gcloud services enable containerregistry.googleapis.com
   ```

3. **Docker**: For building container images

## Setup Steps

### 1. Build and Push Docker Image

```bash
# Build the Docker image
docker build -t gcr.io/YOUR_PROJECT_ID/tablesense2:latest -f vertex_ai/Dockerfile .

# Push to Google Container Registry
docker push gcr.io/YOUR_PROJECT_ID/tablesense2:latest
```

### 2. Upload Training Data

You have two options:

**Option A: Use Cloud Storage (Recommended)**
```bash
# Create a bucket
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME

# Upload training data
gsutil -m cp -r training_data/tablesense gs://YOUR_BUCKET_NAME/training_data/
```

Then modify the code to read from GCS paths.

**Option B: Include in Docker Image**
- Add training data to the Docker image (larger image size)

### 3. Submit Training Job

**Using the provided script:**
```bash
chmod +x vertex_ai/submit_training.sh
export GOOGLE_CLOUD_PROJECT="your-project-id"
export GOOGLE_CLOUD_REGION="us-central1"
./vertex_ai/submit_training.sh configs/detector_config_vertex_ai.yaml
```

**Or manually using gcloud:**
```bash
gcloud ai custom-jobs create \
  --project=YOUR_PROJECT_ID \
  --region=us-central1 \
  --display-name=detector-training \
  --worker-pool-spec=machine-type=n1-standard-4,replica-count=1,\
container-image-uri=gcr.io/YOUR_PROJECT_ID/tablesense2:latest,\
accelerator-type=NVIDIA_TESLA_T4,accelerator-count=1 \
  --args="--config=configs/detector_config_vertex_ai.yaml"
```

## GPU Options

Available GPU types on Vertex AI:
- `NVIDIA_TESLA_T4` - Cost-effective, good for development
- `NVIDIA_TESLA_V100` - High performance
- `NVIDIA_TESLA_P100` - Older but cheaper
- `NVIDIA_A100` - Latest, most powerful (most expensive)

## Machine Types

Common options:
- `n1-standard-4` - 4 vCPUs, 15GB RAM (good for T4)
- `n1-standard-8` - 8 vCPUs, 30GB RAM (for V100/A100)
- `n1-highmem-4` - 4 vCPUs, 26GB RAM (if you need more memory)

## Monitoring

- View jobs: https://console.cloud.google.com/vertex-ai/training/custom-jobs
- View logs: Click on your job → View logs
- W&B dashboard: Your W&B project will show metrics from cloud training

## Cost Considerations

- T4 GPU: ~$0.35/hour
- V100 GPU: ~$2.48/hour
- A100 GPU: ~$3.67/hour

Monitor costs: https://console.cloud.google.com/billing

## Troubleshooting

1. **Permission errors**: Ensure your account has Vertex AI User role
2. **Image not found**: Make sure you pushed the image to GCR
3. **Data not found**: Verify data paths in your code match GCS paths
4. **Out of memory**: Reduce batch_size or use a GPU with more memory

## Alternative: Vertex AI Workbench

For interactive development:

```bash
# Create a Workbench instance with GPU
gcloud notebooks instances create detector-notebook \
  --project=YOUR_PROJECT_ID \
  --location=us-central1-b \
  --machine-type=n1-standard-4 \
  --accelerator-type=NVIDIA_TESLA_T4 \
  --accelerator-count=1 \
  --install-gpu-driver
```

Then you can run training interactively in Jupyter.
