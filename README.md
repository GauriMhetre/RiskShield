# RiskShield

![CI](https://github.com/<YOUR_GITHUB_USERNAME>/<YOUR_REPO_NAME>/actions/workflows/ci.yml/badge.svg)

A real-time, ML-powered fraud detection platform.

**[Live Demo](https://riskshield-frontend-YOUR-URL.onrender.com)** (Replace with your actual Render URL!)

## Overview

RiskShield is an end-to-end machine learning application that detects fraudulent transactions in real-time. It features:
- **Machine Learning Model**: XGBoost model trained on historical transaction data to predict fraud risk.
- **Backend API**: FastAPI application serving the ML model, connected to a PostgreSQL database for historical context.
- **Frontend Dashboard**: React/Vite dashboard providing real-time visibility into flagged transactions and risk scores.
- **Dockerized Deployment**: Fully containerized using Docker and Docker Compose for seamless local development and cloud deployment.

## Tech Stack

- **Frontend**: React, Vite, Tailwind CSS, Recharts
- **Backend**: FastAPI, SQLAlchemy, PostgreSQL, Pydantic
- **Machine Learning**: XGBoost, Scikit-learn, Pandas
- **Infrastructure**: Docker, Docker Compose, Render (Cloud Hosting)

## Local Setup

1. **Clone the repository**
2. **Set up your environment variables**
   Copy `.env.example` to `.env` and fill in the necessary fields (e.g. `POSTGRES_PASSWORD`).
3. **Start the application**
   ```bash
   docker-compose up --build
   ```
4. **Access the services**
   - Frontend Dashboard: `http://localhost:5173`
   - Backend API Docs: `http://localhost:8000/docs`

## Repository Structure

- `backend/`: FastAPI application and database models
- `frontend/`: React dashboard application
- `ml/`: Machine learning training pipeline and notebooks
- `scripts/`: Utility scripts (e.g., database backfill)
- `db/`: Database schema definitions