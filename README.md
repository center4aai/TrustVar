# TrustVar: A Dynamic Framework for Trustworthiness Evaluation and Task Variation Analysis in Large Language Models

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/Akhmetov-VA/TrustVar)

## Project Description

**TrustVar** is a framework built on our previous LLM trustworthiness testing system. While we previously focused on how LLMs handle tasks, we now rethink the evaluation procedure itself. TrustVar shifts the focus: we investigate the quality of tasks themselves, not just model behavior.

### Key Innovation

Unlike traditional frameworks that test models through tasks, TrustVar tests tasks through models. We analyze tasks as research objects, measuring their ambiguity, sensitivity, and structure, then examine how these parameters influence model behavior.

### Core Features

- **Task Variation Generation**: Automatically creates families of task reformulations
- **Model Robustness Testing**: Evaluates model stability under formulation changes
- **Task Sensitivity Index (TSI)**: Measures how strongly formulations affect model success
- **Multi-language Support**: English and Russian tasks with extensible architecture
- **Interactive Pipeline**: Unified system for data loading, task generation, variation, model evaluation, and visual analysis

## Table of Contents

- [Project Architecture](#project-architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Metrics](#metrics)

## Project Architecture

![TrustVar Architecture](assets/TrustVar_Pipeline.jpg)

### Core Components

- **Data Ingestion** - accepts preformatted datasets in CSV, JSON, Excel, and Parquet formats, supporting both user uploads and built-in collections like SLAVA, RuBia, etc;
- **Task Generator** - applies five controlled transformations: lexico-syntactic paraphrasing, length variation, stylistic shifts, synonym substitution, and word reordering to create semantically equivalent variants;
- **Perturbation Settings** - sets up each transformation with user-configurable parameters (10 by default);
- **Task Pool** - serves as a persistent repository organizing tasks by six trustworthiness dimensions (truthfulness, safety, fairness, robustness, privacy, ethics) and maintaining evaluation queues;
- **LLM Tester** - executes inference on both local models via Ollama and remote APIs, recording outputs with complete metadata for reproducibility;
- **Analyzer** - measures response stability using coefficient of variation, feeding instability flags back for task refinement;
- **Task Meta-Evaluator** - computes the Task Sensitivity Index (TSI) across all model-task pairs, flagging high-TSI items for revision;
- **Evaluator & Visualizer** - computes RtAR, TFNR, Accuracy, and Pearson correlation metrics;
- **Dashboard and Leaderboard** - combine Metrics with Analyser data and display the results for user convenience


## Project Structure

```
trustvar
├── src
│   ├── api
│   │   ├── routes
│   │   │   ├── datasets.py
│   │   │   ├── models.py
│   │   │   └── tasks.py
│   │   └── main.py
│   ├── config
│   │   ├── constants.py
│   │   └── settings.py
│   ├── core
│   │   ├── schemas
│   │   │   ├── dataset.py
│   │   │   ├── model.py
│   │   │   └── task.py
│   │   ├── services
│   │   │   ├── ab_test_analyzer.py
│   │   │   ├── dataset_service.py
│   │   │   ├── eval_service.py
│   │   │   ├── include_exclude_evaluator.py
│   │   │   ├── judge_service.py
│   │   │   ├── model_service.py
│   │   │   ├── rta_evaluator.py
│   │   │   └── task_service.py
│   │   ├── tasks
│   │   │   ├── celery_app.py
│   │   │   ├── health_check_task.py
│   │   │   ├── inference_task.py
│   │   │   ├── model_download_task.py
│   ├── ui
│   │   ├── components
│   │   │   ├── dataset_uploader.py
│   │   │   ├── datasets_section.py
│   │   │   ├── general_section.py
│   │   │   ├── models_section.py
│   │   │   ├── results_section.py
│   │   │   ├── spider_chart_variations.py
│   │   │   ├── task_monitor.py
│   │   │   └── tasks_section.py
│   │   ├── api_client.py
│   │   └── app.py
├── Dockerfile.celery
├── Dockerfile.streamlit
├── README.md
├── docker-compose.dev.yml
├── docker-compose.yml
├── pyproject.toml
```

## Quick Start

### Requirements

- **Docker** and **Docker Compose**
- **Python 3.11+** (for local development)

### Launch with Docker

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd trustvar
   ```

2. **Create `.env` file with environment variables:**
   ```env
    HF_TOKEN=<your-hf-token>
    OPENAI_API_KEY=<your-openai-key>
    OPENAI_BASE_URL=<base_url>
    FRONTEND_PORT=27364
   ```

3. **Launch all services:**
   ```bash
   docker-compose up -d
   ```

4. **Download datasets and auxiliary information:**
   
   After running `docker-compose up`, you need to download the datasets and auxiliary information from our Google Drive and upload them to MongoDB:
   
   **[📥 Download Datasets](https://drive.google.com/drive/folders/1jvBWvAc9JcjLYQ8T09xoDKUkwjCf7tiI?usp=sharing)**
   
   The drive contains:
   - **Accuracy_Groups.json** - Accuracy metrics grouped by categories
   - **Accuracy.json** - Main accuracy dataset
   - **Correlation.json** - Correlation metrics
   - **IncludeExclude.json** - Include/Exclude analysis data
   - **RtAR.json** - Refuse to Answer metrics
   - **TFNR.json** - True False Negative Rate metrics
   - **jailbreak.json** - Jailbreak detection tasks
   - **ood_detection.json** - Out-of-distribution detection
   - **privacy_assessment.json** - Privacy assessment tasks
   - **stereotypes_detection_3.json** - Stereotype detection
   - **tasks.json** - Task definitions
   - And many more specialized datasets...


5. **Open the web interface:**
   - Monitoring: http://localhost:27366
   - MongoDB Express: http://localhost:27374

## Metrics

Supported metric types:

- **Accuracy** - Response accuracy
- **RtA (Refuse to Answer)** - Analysis of answer refusals
- **Correlation** - Correlation with reference answers
- **Include/Exclude** - Analysis of element inclusion/exclusion

## License

This project is licensed under the MIT License.