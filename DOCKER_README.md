# CRR RAG Docker Deployment

## Quick Start

### Build and Run with Docker Compose (Recommended)

```bash
# Build and start the container
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the container
docker-compose down
```

Access the app at: http://localhost:8501

### Build and Run with Docker

```bash
# Build the image
docker build -t crr-rag-app .

# Run the container
docker run -d \
  --name crr-rag-streamlit \
  -p 8501:8501 \
  --env-file .env \
  crr-rag-app

# View logs
docker logs -f crr-rag-streamlit

# Stop the container
docker stop crr-rag-streamlit
docker rm crr-rag-streamlit
```

## Environment Variables

Make sure your `.env` file contains:

```env
NVIDIA_API_KEY=your_nvidia_api_key
GEMINI_API_KEY=your_gemini_api_key
ASTRA_DB_TOKEN=your_astra_db_token
ASTRA_DB_API_ENDPOINT=your_astra_db_endpoint
ASTRA_DB_ID=your_astra_db_id
```

## Production Deployment

### Remove Development Volume Mount

Edit `docker-compose.yml` and comment out the volumes section:

```yaml
    # volumes:
    #   - ./st_crr.py:/app/st_crr.py
```


### Deploy to Cloud



#### Google Cloud Run
```bash
# Build and push to Google Container Registry
gcloud builds submit --tag gcr.io/PROJECT-ID/crr-rag-app

# Deploy to Cloud Run
gcloud run deploy crr-rag-app \
  --image gcr.io/PROJECT-ID/crr-rag-app \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
```

## Troubleshooting

### Check container status
```bash
docker ps
```

### View logs
```bash
docker logs crr-rag-streamlit
```

### Enter container shell
```bash
docker exec -it crr-rag-streamlit bash
```

### Rebuild after code changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```

## Health Check

The container includes a health check endpoint:
- URL: http://localhost:8501/_stcore/health
- Interval: 30 seconds
- Timeout: 10 seconds

## Resource Requirements

- **Memory**: 1GB minimum, 2GB recommended
- **CPU**: 1 core minimum, 2 cores recommended
- **Storage**: 500MB for image + dependencies

## Security Notes

⚠️ **Important**: Never commit `.env` file to version control!

Add to `.gitignore`:
```
.env
.env.local
```

## Monitoring

### View resource usage
```bash
docker stats crr-rag-streamlit
```

### Set resource limits
Edit `docker-compose.yml`:
```yaml
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 2G
        reservations:
          cpus: '1'
          memory: 1G
```
