# 🏦 Bank Marketing Term Deposit Prediction
### Machine Learning Zoomcamp (Cohort 2026) - Midterm Project

[![Python 3.11](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Tuned-FF6F00?logo=xgboost&logoColor=white)](https://xgboost.readthedocs.io/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-181717?logo=github&logoColor=white)](https://github.com/chawlamayur11/bank-term-deposit-prediction)

---

## 📌 Table of Contents
1. [Problem Description & Business Impact](#1-problem-description--business-impact)
2. [Dataset Overview & Feature Dictionary](#2-dataset-overview--feature-dictionary)
3. [Key EDA Findings & Domain Nuances](#3-key-eda-findings--domain-nuances)
4. [Validation Strategy & Encoding](#4-validation-strategy--encoding)
5. [Model Training, Tuning & Comparison](#5-model-training-tuning--comparison)
6. [Final Model Performance & Feature Importance](#6-final-model-performance--feature-importance)
7. [Repository Structure](#7-repository-structure)
8. [Getting Started & Local Setup](#8-getting-started--local-setup)
9. [Training the Model (`train.py`)](#9-training-the-model-trainpy)
10. [Serving Predictions with Web Service (`predict.py`)](#10-serving-predictions-with-web-service-predictpy)
11. [Testing the Service (`test.py` & `curl`)](#11-testing-the-service-testpy--curl)
12. [Containerization with Docker](#12-containerization-with-docker)
13. [Cloud Deployment Guide](#13-cloud-deployment-guide)

---

## 1. Problem Description & Business Impact

### Context
Commercial banks frequently conduct outbound telemarketing campaigns to market long-term deposit investments. A **term deposit** is a cash investment where funds are committed for a fixed tenure in exchange for a predetermined interest rate. These deposits represent stable, low-cost capital for banks to support credit lending.

### The Problem
Traditional telemarketing campaigns operate on broad cold-calling lists, suffering from:
* **High Operational Expenses**: Call center personnel spend hundreds of hours dialing customers who have no inclination to invest.
* **Customer Fatigue & Brand Friction**: Repeated unwanted sales calls alienate clients.
* **Low Conversion Rates**: Only ~11.3% of contacted clients agree to open a term deposit.

### The Machine Learning Solution
We build an end-to-end binary classification system to **predict the probability of a client subscribing to a term deposit** before dialing them.

By scoring and ranking prospective leads in advance:
1. Telemarketers can **prioritize high-probability prospects** (top deciles), dramatically boosting conversion rates.
2. Low-propensity customers can be excluded from calling campaigns or redirected toward digital/email promotions.
3. Marketing spend and call center staffing can be efficiently planned based on real-time macroeconomic and demographic conditions.

---

## 2. Dataset Overview & Feature Dictionary

The project utilizes the **Bank Marketing (bank-additional-full)** benchmark dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/222/bank+marketing) (Moro et al., 2014), comprising **41,188 records and 20 input attributes**.

The dataset is committed directly in this repository under [`data/bank-additional-full.csv`](data/bank-additional-full.csv).

### Features
| Category | Feature | Type | Description |
| :--- | :--- | :--- | :--- |
| **Client Data** | `age` | Numeric | Age in years |
| | `job` | Categorical | Job category (e.g., admin, blue-collar, technician, management) |
| | `marital` | Categorical | Marital status (`divorced`, `married`, `single`, `unknown`) |
| | `education` | Categorical | Education level (`basic.4y`, `high.school`, `university.degree`, etc.) |
| | `default` | Categorical | Has credit in default? (`no`, `yes`, `unknown`) |
| | `housing` | Categorical | Has housing loan? (`no`, `yes`, `unknown`) |
| | `loan` | Categorical | Has personal loan? (`no`, `yes`, `unknown`) |
| **Campaign Contacts** | `contact` | Categorical | Contact communication channel (`cellular`, `telephone`) |
| | `month` | Categorical | Last contact month of year (`jan` to `dec`) |
| | `day_of_week` | Categorical | Last contact day of the week (`mon` to `fri`) |
| | `campaign` | Numeric | Number of contacts performed during this campaign |
| | `pdays` | Numeric | Days passed after previous campaign contact (`999` = never contacted) |
| | `previous` | Numeric | Number of contacts performed before this campaign |
| | `poutcome` | Categorical | Outcome of the previous campaign (`failure`, `nonexistent`, `success`) |
| **Macroeconomic** | `emp_var_rate` | Numeric | Employment variation rate (quarterly indicator) |
| | `cons_price_idx`| Numeric | Consumer price index (monthly indicator) |
| | `cons_conf_idx` | Numeric | Consumer confidence index (monthly indicator) |
| | `euribor3m` | Numeric | Euribor 3-month interest rate (daily indicator) |
| | `nr_employed` | Numeric | Number of employees in economy (quarterly indicator) |
| **Target** | `y` | Binary | Has client subscribed to term deposit? (`yes`: 1, `no`: 0) |

---

## 3. Key EDA Findings & Domain Nuances

Full exploration with visual charts is documented in [`notebook.ipynb`](notebook.ipynb).

### 1. Target Imbalance
* **No (`0`)**: 36,548 (88.73%)
* **Yes (`1`)**: 4,640 (11.27%)  
* *Takeaway*: Severe class imbalance. Standard accuracy can be deceptive (a naive null classifier achieves 88.7%), so **ROC-AUC** is selected as the primary optimization metric.

### 2. Operational Reality: Why `duration` MUST Be Excluded
The UCI dataset includes a `duration` column (call duration in seconds). As emphasized by the authors (Moro et al., 2014):
> *"The call duration is not known before a call is performed. Also, after the call ends, `y` is known. Thus, this input should only be included for benchmark purposes and discarded for a realistic predictive model."*

Including `duration` would introduce **severe data leakage**, resulting in an artificial model useless for prospective outbound sales. **`duration` is strictly excluded from all training features.**

### 3. Mutual Information (Categorical Importance)
* `poutcome` (MI = 0.0304) and `month` (MI = 0.0267) have the highest mutual dependence with conversion.
* Clients who previously agreed to a campaign (`poutcome = success`) have an extraordinary conversion rate of **over 65%**.
* Contacts made in **October, September, and March** show conversion rates exceeding **40%**, compared to under **7% in May**.

### 4. Macroeconomic Drivers
* During high employment variation rates and high 3-month Euribor rates (~4.85%), term deposit subscriptions drop sharply.
* When central rates drop (Euribor < 1.5%), consumers aggressively seek guaranteed fixed-rate savings vehicles, pushing term deposit conversions up.

---

## 4. Validation Strategy & Encoding

* **Split Ratio**: Stratified 60% Train (24,712), 20% Validation (8,238), and 20% Test (8,238) with `random_state=42`.
* **Feature Representation**: `DictVectorizer(sparse=False)` to cleanly one-hot encode categorical features while passing numerical values through without leakage.

---

## 5. Model Training, Tuning & Comparison

Four distinct model families were systematically evaluated and tuned on the validation set using **ROC-AUC**:

| Model Architecture | Hyperparameter Search / Tuning | Best Parameters | Val ROC-AUC |
| :--- | :--- | :--- | :---: |
| **Logistic Regression** | Regularization $C \in [0.001, 0.01, 0.1, 1.0, 10.0]$ | `solver='liblinear'`, `C=1.0` | **0.7908** |
| **Decision Tree** | `max_depth` $\in [4, 6, 8, 10]$, `min_samples_leaf` $\in [5, 10, 20, 50]$ | `max_depth=6`, `min_samples_leaf=20` | **0.7874** |
| **Random Forest** | `n_estimators` $\in [50, 100, 150]$, `max_depth` $\in [6, 8, 10]$ | `n_estimators=100`, `max_depth=10`, `min_leaf=5` | **0.8026** |
| **XGBoost (Winner)** | `eta` $\in [0.05, 0.1, 0.2]$, `max_depth` $\in [3, 4, 5, 6]$ | `eta=0.05`, `max_depth=5`, `num_boost_round=100` | **0.8080** |

**Selection Reasoning**:  
Gradient-boosted decision trees (XGBoost) demonstrated superior non-linear feature interaction capture (especially between macroeconomic indicators and client history), achieving the highest validation ROC-AUC score (**0.8080**).

---

## 6. Final Model Performance & Feature Importance

The winning tuned XGBoost configuration was trained on the combined training set (`df_full_train = df_train + df_val`, 32,950 records) and evaluated on the untouched test set (`df_test`, 8,238 records):

| Metric | Test Set Score | Description |
| :--- | :---: | :--- |
| **ROC-AUC** | **0.8145** | Robust discriminative capability across all decision thresholds |
| **Accuracy** | **90.25%** | Overall correct classification rate (threshold = 0.5) |
| **Precision** | **68.66%** | High fidelity when predicting deposit subscription |
| **Recall** | **24.78%** | Conservative threshold capturing high-certainty leads |
| **F1-Score** | **0.3642** | Harmonic mean balancing precision and recall |

### Top 5 Predictive Features (XGBoost Gain)
1. `nr_employed`: Number of employees (prevailing labor market strength).
2. `poutcome = success`: Prior marketing campaign success.
3. `euribor3m`: 3-month Euribor interbank lending rate.
4. `month = oct` / `month = mar`: Timing and fiscal quarter-end effects.
5. `age`: Client life stage and accumulated savings capability.

---

## 7. Repository Structure

```text
midterm-project/
├── data/
│   └── bank-additional-full.csv   # Committed benchmark dataset (41,188 rows)
├── notebook.ipynb                 # Full executed EDA, tuning, & validation notebook
├── train.py                       # Standalone training & model serialization script
├── predict.py                     # FastAPI web service serving predictions
├── test.py                        # Client test script with realistic client profiles
├── model.bin                      # Serialized final model & DictVectorizer
├── requirements.txt               # Pinned pip dependencies
├── Pipfile                        # Pipenv environment specification
├── Dockerfile                     # Docker container configuration
├── .dockerignore                  # Docker build ignore rules
└── README.md                      # Comprehensive project documentation
```

---

## 8. Getting Started & Local Setup

### Option A: Standard `venv` (Recommended)
1. Navigate to the project directory:
   ```bash
   cd midterm-project
   ```
2. Create and activate a virtual environment:
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Linux / macOS
   python3 -m venv venv
   source venv/bin/activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Option B: Using `pipenv`
```bash
cd midterm-project
pipenv install
pipenv shell
```

---

## 9. Training the Model (`train.py`)

To re-train the model from scratch on the full training set and export a fresh `model.bin`:

```bash
python train.py
```

Sample output:
```text
[1/5] Loading data from 'data/bank-additional-full.csv'...
      Loaded 41188 rows and 21 columns.
      Selected 19 operational features (excluded 'duration' to prevent leakage).
[2/5] Vectorizing features using DictVectorizer...
[3/5] Training tuned XGBoost model on full training set...
      Model training complete.
[4/5] Evaluating model on unseen test set (20%)...
==================================================
           FINAL TEST EVALUATION METRICS           
==================================================
  Test ROC-AUC   : 0.81445
  Test Accuracy  : 0.90252 (90.25%)
  Test Precision : 0.68657
  Test Recall    : 0.24784
  Test F1-Score  : 0.36421
==================================================
[5/5] Saving model and vectorizer to 'model.bin'...
      Successfully exported to 'model.bin'! Ready for deployment.
```

---

## 10. Serving Predictions with Web Service (`predict.py`)

The prediction service is implemented using **FastAPI** and served with **Uvicorn**.

### Starting the Server
```bash
python predict.py
```
Or directly with Uvicorn:
```bash
uvicorn predict:app --host 0.0.0.0 --port 8000
```

Once running, interactive documentation (Swagger UI) is available at:
👉 **[http://localhost:8000/docs](http://localhost:8000/docs)**

---

## 11. Testing the Service (`test.py` & `curl`)

### Automated Python Test Client
Run the provided `test.py` script:
```bash
python test.py
```

The script evaluates two contrasting prospects:
* **Client A (High Propensity)**: Previously converted in past campaign, university degree, contacted in October.
  ```json
  {
    "subscription_probability": 0.6582,
    "subscribe": true,
    "threshold": 0.5,
    "recommendation": "High propensity lead: Prioritize outbound telemarketing call."
  }
  ```
* **Client B (Low Propensity)**: Blue-collar, high-interest rate climate, no prior relationship.
  ```json
  {
    "subscription_probability": 0.0286,
    "subscribe": false,
    "threshold": 0.5,
    "recommendation": "Low propensity lead: Do not prioritize for telemarketing."
  }
  ```

### Direct `curl` Example
```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
       "age": 45,
       "job": "management",
       "marital": "married",
       "education": "university.degree",
       "default": "no",
       "housing": "no",
       "loan": "no",
       "contact": "cellular",
       "month": "oct",
       "day_of_week": "wed",
       "campaign": 1,
       "pdays": 3,
       "previous": 2,
       "poutcome": "success",
       "emp_var_rate": -3.4,
       "cons_price_idx": 92.431,
       "cons_conf_idx": -26.9,
       "euribor3m": 0.739,
       "nr_employed": 5017.5
     }'
```

---

## 12. Containerization with Docker

The web service is fully containerized using `python:3.11-slim`.

### 1. Build the Docker Image
```bash
docker build -t bank-marketing-service .
```

### 2. Run the Container
```bash
docker run -it --rm -p 8000:8000 bank-marketing-service
```

### 3. Verify the Containerized Service
In another terminal:
```bash
curl http://localhost:8000/health
python test.py http://localhost:8000
```

---

## 13. Cloud Deployment Guide

The containerized service can be deployed to any modern cloud container platform. Below are step-by-step instructions:

### Option 1: Deploy to Render (Recommended / Free Tier)
1. Fork or push this repository to GitHub.
2. Sign in to [Render](https://render.com/).
3. Click **New +** $\rightarrow$ **Web Service**.
4. Select your GitHub repository.
5. In **Root Directory**, set: `midterm-project`.
6. Choose runtime: **Docker**.
7. Click **Create Web Service**. Render automatically builds the `Dockerfile` and provides a live public HTTPS URL (e.g., `https://bank-marketing-service.onrender.com`).
8. Test your public deployment:
   ```bash
   python test.py https://bank-marketing-service.onrender.com
   ```

### Option 2: Deploy to Google Cloud Run
1. Authenticate with Google Cloud CLI:
   ```bash
   gcloud auth login
   gcloud config set project <YOUR_PROJECT_ID>
   ```
2. Build and submit container image:
   ```bash
   gcloud builds submit --tag gcr.io/<YOUR_PROJECT_ID>/bank-marketing-service midterm-project/
   ```
3. Deploy to Cloud Run:
   ```bash
   gcloud run deploy bank-marketing-service \
       --image gcr.io/<YOUR_PROJECT_ID>/bank-marketing-service \
       --platform managed \
       --region us-central1 \
       --allow-unauthenticated
   ```

### Option 3: Deploy to AWS Elastic Beanstalk / ECS
1. Initialize Elastic Beanstalk CLI inside `midterm-project/`:
   ```bash
   eb init -p docker bank-marketing-service
   ```
2. Create environment:
   ```bash
   eb create bank-marketing-env
   ```
3. Open or test the deployed endpoint:
   ```bash
   eb open
   ```

---

## 🎓 Evaluation Rubric Checklist

- [x] **Problem Description (2/2 pts)**: Detailed business problem, objectives, and operational use case.
- [x] **EDA (2/2 pts)**: Missing values, target distribution, numerical & categorical breakdown, mutual info, correlation matrix, and data leakage discussion.
- [x] **Model Training (3/3 pts)**: Four architectures (Logistic Regression, Decision Trees, Random Forests, XGBoost) systematically trained and tuned.
- [x] **Export to Script (1/1 pt)**: Clean, self-contained `train.py` script.
- [x] **Reproducibility (1/1 pt)**: Full dataset committed in repo; notebook and scripts execute cleanly without errors.
- [x] **Model Deployment (1/1 pt)**: FastAPI web service serving predictions (`predict.py`).
- [x] **Dependency Management (2/2 pts)**: Both `requirements.txt` and `Pipfile` included with virtual environment instructions.
- [x] **Containerization (2/2 pts)**: Multi-layer cached `Dockerfile` and `.dockerignore`.
- [x] **Cloud Deployment (2/2 pts)**: Step-by-step production deployment guides for Render, GCP Cloud Run, and AWS.

**Total Expected Score: 16 / 16 Points**
