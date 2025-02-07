import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, precision_recall_curve, auc
import lime
import lime.lime_tabular

# Load Dataset
file_path = '/content/test_Y3wMUE5_7gLdaTN.xlsx'
df = pd.read_excel(file_path)

# Data Preprocessing
print("Initial dataset shape:", df.shape)
df.dropna(inplace=True)  # Drop missing values
df = df.select_dtypes(include=[np.number])  # Keep only numeric columns
print("Shape after preprocessing:", df.shape)

# Splitting Features and Target
X = df.iloc[:, :-1]  # Features
y = df.iloc[:, -1]   # Target

# Encoding categorical target
if y.dtype == 'object':
    le = LabelEncoder()
    y = le.fit_transform(y)

# Train-test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Feature Scaling
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Handling Class Imbalance
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
under_sampler = RandomUnderSampler()
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

# Compute Class Weights
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {i: class_weights[i] for i in range(len(class_weights))}

# Model Training
models = {
    "Logistic Regression": LogisticRegression(class_weight='balanced'),
    "Random Forest": RandomForestClassifier(class_weight='balanced', n_estimators=100),
    "XGBoost": XGBClassifier(scale_pos_weight=class_weights[1])
}

results = {}
for name, model in models.items():
    model.fit(X_train_smote, y_train_smote)
    y_pred = model.predict(X_test)
    results[name] = {
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    print(f"\n{name} Performance:\n", results[name]["classification_report"])

# Model Evaluation
plt.figure(figsize=(10, 5))
for name, model in models.items():
    y_scores = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.plot(fpr, tpr, label=name)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.show()

# Explainable AI with LIME
explainer = lime.lime_tabular.LimeTabularExplainer(X_train_smote, feature_names=X.columns.tolist(), class_names=["Class 0", "Class 1"], discretize_continuous=True)
idx = np.random.randint(0, X_test.shape[0])
exp = explainer.explain_instance(X_test[idx], models["Random Forest"].predict_proba, num_features=5)
exp.show_in_notebook()
