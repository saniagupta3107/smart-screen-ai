import joblib, re, numpy as np, pandas as pd
from textblob import TextBlob
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# ---------- load & clean ----------
df = pd.read_csv("../data/combined_employee_data.csv")     # record_id, reason_and_factors, cessation_year
def extract_year(s):
    m = re.search(r"(\d{4})", str(s))
    return int(m.group(1)) if m else np.nan
df["cessation_year"]  = df["cessation_year"].apply(extract_year)
df["sentiment_score"] = df["reason_and_factors"].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

def label_attrition_risk(txt):
    txt = str(txt).lower()
    return int("resignation" in txt or "ill health" in txt)
df["attrition_risk"] = df["reason_and_factors"].apply(label_attrition_risk)

# ---------- encode & train ----------
le = LabelEncoder()
df["reason_encoded"] = le.fit_transform(df["reason_and_factors"].fillna(""))

X = df[["cessation_year", "sentiment_score"]]
y = df["attrition_risk"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

print("Train accuracy:", model.score(X_train, y_train))
print("Test  accuracy:", model.score(X_test,  y_test))

# ---------- persist ----------
joblib.dump(model, "model.pkl")
joblib.dump(le,    "label_encoder.pkl")
print("Saved model.pkl and label_encoder.pkl")
