# Resume Screening & Employee Sentiment Analysis

A two‑part ML project that

1. ranks résumés against a job description, and
2. predicts employee‑attrition risk from free‑text feedback and recommends an engagement strategy via a Flask API.

---

## 📂 Project Layout

```
.
├── app.py                     # Flask API for Task 2
├── train_model.py             # Model‑training script for Task 2
├── notebooks/
│   ├── preprocess.ipynb               # text cleaning helpers (résumés)
│   ├── resume_screening.ipynb         # TF‑IDF résumé ranking demo
│   ├── resume_screening2.ipynb        # Embeddings + NER résumé ranking demo
│   ├── preporcess_EmpData.ipynb       # data wrangling for employee dataset
│   └── EmployeeSentiment.ipynb        # model exploration + evaluation
├── data/
│   ├── resumes.csv
│   └── combined_employee_data.csv
├── model/
│   ├── model.pkl              # Random‑Forest attrition model
│   └── label_encoder.pkl      # Saved LabelEncoder
└── requirements.txt
```

---

## 1  Resume Screening (Task 1)

### 1.1  Quick‑Start

```bash
# Run the TF‑IDF demo
jupyter notebook notebooks/resume_screening.ipynb

# Run the Embedding + NER demo
jupyter notebook notebooks/resume_screening2.ipynb
```

### 1.2  Pipelines ⏩

| Step | TF‑IDF Approach                         | Embedding + NER Approach                          |
| ---- | --------------------------------------- | ------------------------------------------------- |
| 1    | Load `resumes.csv`                      | Load `resumes.csv`                                |
| 2    | Add **job description** to corpus       | Add **job description** to corpus                 |
| 3    | `TfidfVectorizer(stop_words='english')` | Clean text ➜ spaCy NER ➜ extract skills           |
| 4    | Cosine similarity (`sklearn.metrics`)   | Embed with **BAAI/bge‑small‑en v1.5** (LangChain) |
| 5    | Rank & sort by similarity               | Cosine similarity (`scipy`) & rank                |

> **Output:** `top_k` résumés most relevant to the description.

---

## 2  Employee Sentiment API (Task 2)

An HTTP endpoint that returns attrition‑risk predictions and engagement advice.

### 2.1  Train the Model

```bash
python train_model.py          # creates model/model.pkl & model/label_encoder.pkl
```

Pipeline (condensed):
1. Load `combined_employee_data.csv`.
2. Extract `cessation_year` (regex).
3. Compute `sentiment_score` (TextBlob polarity).
4. Label `attrition_risk` (keyword rule).
5. Train‑test split 60/40.
6. RandomForest on `cessation_year + sentiment_score`.
7. Persist artefacts with `joblib`.

### 2.2  Run Locally

```bash
pip install -r requirements.txt
python -m textblob.download_corpora   # one‑time for sentiment

# Dev server
python app.py                         # listens on :5000

# Prod
gunicorn app:app -b 0.0.0.0:8000 -w 4
```

### 2.3  API Reference

| Method | Endpoint   | Body                                               | Description                       |
| ------ | ---------- | -------------------------------------------------- | --------------------------------- |
| POST   | `/predict` | `{"feedback":"The workload is very comfortable…"}` | Returns JSON with risk & strategy |

Sample Response

```json
{
  "feedback": "The workload is very comfortable and I often don't have to stay late.",
  "sentiment_score": 0.35,
  "model_probability": 0.11,
  "predicted_attrition_risk": false,
  "engagement_strategy": "Maintain current engagement and recognition programs."
}
```

A live deployment is available at **[https://employee-feedback-ww3a.onrender.com/predict](https://employee-feedback-ww3a.onrender.com/predict)**.

---

## 🛠️ Installation

```bash
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

> If using the résumé‑ranking notebooks, you may also need to download spaCy & HF assets:
>
> ```bash
> python -m spacy download en_core_web_sm
> ```

---

## 🚀 Deployment on Render

1. **Pre‑Deploy Command**: `python -m textblob.download_corpora`
2. **Start Command**: `gunicorn app:app`
   (Configure these in the Render dashboard or `render.yaml`.)

---

## ✍️ Author & License

Created by **Dilshad**. Licensed under the MIT License.
