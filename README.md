# Project Name - MLOps Trustpilot sentiment analysis System

This project is a starter pack for a complete **MLOps pipeline**, based on a movie Trustpilot sentiment analysis System. It is designed to demonstrate best practices in development, versioning, containerization, CI/CD, and production deployment.

## 🎯 Objectives

- Build an end-to-end movie recommendation pipeline
- Integrate DVC for data and model versioning
- Track experiments using MLflow
- Containerize the application using Docker
- Deploy the FastAPI + ML model app on Render
- Use GitHub Actions for automated testing and deployment (CI/CD)

---

## 🔧 Tech Stack

| Tool         | Role                                                       |
|--------------|------------------------------------------------------------|
| **FastAPI**  | Serves the recommendation model via a REST API             |
| **MLflow**   | Tracks experiments and metrics                             |
| **DVC**      | Version control for datasets and models                    |
| **Docker**   | Containerizes the API and model                            |
| **GitHub Actions** | Automates CI/CD: testing, building, deployment       |
| **Render**   | Hosts the FastAPI app and the MLflow tracking server       |

---

## 🛠️ Project Structure

```bash
├── data/                # Raw, intermediate, and processed datasets
├── models/              # Trained models (versioned using DVC)
├── notebooks/           # Jupyter notebooks for prototyping and exploration
├── reports/figures/     # Graphs, charts, and the MLOps diagram
├── logs/                # Logs from training and prediction
├── requirements.txt     # Python dependencies
├── src/                 # Source code
│   ├── tests/           # Test code
│   ├── app/             # FastAPI app
│   ├── Dockerfile           # Dockerfile for containerizing the API
│   ├── docker-compose           # Docker-compose
│   ├── requirements.txt         # requirements.txt
├── .github/workflows/   # GitHub Actions workflows (CI/CD)
