# LungCare Triage: Lung Cancer Detection with Ensemble CNN Models

LungCare Triage is an advanced, full-stack predictive web application designed to aid medical professionals and radiologists in early lung cancer detection from 3D CT Scans. 

The core predictive engine leverages an **Ensemble of 3D Convolutional Neural Networks (ResNet-3D, DenseNet-3D, VGG-3D)**, aggregated via soft-voting probabilities to maximize sensitivity and reduce false positives across malignant nodules.

## Features
- **Ensemble Voting Mechanism**: Aggregates the probability scores of multiple CNN architectures to yield a robust final prediction.
- **3D Visualization**: Generates real-time, interactive 3D visualizations from raw slices.
- **Data Brutalism UI**: A modern, healthcare-focused dark theme interface built with Next.js and TailwindCSS.
- **RESTful Backend**: High-performance FastAPI backend processing `.mhd` files.
- **Containerized Workloads**: Fully deployable via Docker multi-stage builds.

## Setup Instructions

### 1. The Easy Way (Docker Compose)
Ensure Docker Desktop is running on your machine, then execute:
```bash
docker-compose up --build
```
- Frontend will be live at `http://localhost:3000`
- Backend API Docs at `http://localhost:8000/docs`

### 2. Manual Setup
If you prefer not to use Docker:
**Backend System:**
```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

**Frontend User Interface:**
```bash
cd frontend-nextjs
npm install
npm run dev
```

Navigate to `http://localhost:3000` to interact with the Ensemble Dashboard.
