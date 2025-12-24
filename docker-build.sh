#!/bin/bash

# Docker build and run script for CRR RAG Streamlit App

set -e

echo "🐳 Building CRR RAG Docker Image..."

# Build the Docker image
docker build -t crr-rag-app:latest .

echo "✅ Build complete!"
echo ""
echo "📦 Image size:"
docker images crr-rag-app:latest

echo ""
echo "🚀 Starting container..."

# Stop and remove existing container if it exists
docker stop crr-rag-streamlit 2>/dev/null || true
docker rm crr-rag-streamlit 2>/dev/null || true

# Run the container
docker run -d \
  --name crr-rag-streamlit \
  -p 8501:8501 \
  --env-file .env \
  --restart unless-stopped \
  crr-rag-app:latest

echo "✅ Container started!"
echo ""
echo "📊 Container status:"
docker ps | grep crr-rag-streamlit

echo ""
echo "🌐 Access the app at: http://localhost:8501"
echo ""
echo "📝 View logs with: docker logs -f crr-rag-streamlit"
echo "🛑 Stop container with: docker stop crr-rag-streamlit"
