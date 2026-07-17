import json
import os

notebook = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Phishing URL Detector: Machine Learning Pipeline\n",
                "\n",
                "This notebook walks through the development, training, and evaluation of a machine learning system to classify URLs as either **phishing** or **legitimate**.\n",
                "\n",
                "## Pipeline Sections:\n",
                "1. **Environment Setup & Data Loading**\n",
                "2. **Exploratory Data Analysis (EDA)**\n",
                "3. **Feature Scaling & Preprocessing**\n",
                "4. **Baseline & Advanced Model Training** (Logistic Regression, Random Forest, Gradient Boosting)\n",
                "5. **Precision/Recall Tradeoff & Threshold Tuning** (maximizing Recall for Phishing detection)\n",
                "6. **Global Interpretability & Feature Importance**\n",
                "7. **Live Inference Example**"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 1. Environment Setup & Data Loading"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "import os\n",
                "import pandas as pd\n",
                "import numpy as np\n",
                "import matplotlib.pyplot as plt\n",
                "import seaborn as sns\n",
                "from sklearn.model_selection import train_test_split\n",
                "from sklearn.preprocessing import StandardScaler\n",
                "from sklearn.linear_model import LogisticRegression\n",
                "from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier\n",
                "from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, f1_score\n",
                "import joblib\n",
                "\n",
                "# Set plot style\n",
                "sns.set_theme(style=\"whitegrid\")\n",
                "plt.rcParams[\"figure.figsize\"] = (10, 6)\n",
                "plt.rcParams[\"font.size\"] = 12"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Load train and test sets\n",
                "train_df = pd.read_csv(\"data/train_data.csv\")\n",
                "test_df = pd.read_csv(\"data/test_data.csv\")\n",
                "\n",
                "print(f\"Train Data shape: {train_df.shape}\")\n",
                "print(f\"Test Data shape: {test_df.shape}\")\n",
                "train_df.head()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 2. Exploratory Data Analysis (EDA)\n",
                "Let's look at the label distribution and see how some key features behave across legitimate vs phishing URLs."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Check class balance\n",
                "print(\"Class distribution in train set:\")\n",
                "print(train_df['label'].value_counts())\n",
                "\n",
                "# Plot distribution of URL length\n",
                "plt.figure()\n",
                "sns.histplot(data=train_df, x='url_length', hue='label', kde=True, bins=50, multiple='stack')\n",
                "plt.title('Distribution of URL Length by Class')\n",
                "plt.xlabel('URL Length')\n",
                "plt.ylabel('Count')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Let's look at average values of numerical features by class\n",
                "numerical_cols = ['url_length', 'num_dots', 'num_hyphens', 'num_slashes', 'subdomain_count', 'num_suspicious_keywords']\n",
                "mean_features = train_df.groupby('label')[numerical_cols].mean().reset_index()\n",
                "mean_features_melted = pd.melt(mean_features, id_vars=['label'], value_vars=numerical_cols)\n",
                "\n",
                "plt.figure(figsize=(14, 7))\n",
                "sns.barplot(data=mean_features_melted, x='variable', y='value', hue='label')\n",
                "plt.title('Comparison of Feature Means by Class (0 = Legit, 1 = Phishing)')\n",
                "plt.ylabel('Mean Value')\n",
                "plt.xlabel('Features')\n",
                "plt.xticks(rotation=15)\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Plot categorical presence features like HTTPS presence and SSL validity\n",
                "fig, axes = plt.subplots(1, 2, figsize=(16, 6))\n",
                "\n",
                "sns.countplot(data=train_df, x='has_https', hue='label', ax=axes[0])\n",
                "axes[0].set_title('HTTPS Presence (0 = HTTP, 1 = HTTPS)')\n",
                "\n",
                "sns.countplot(data=train_df, x='cert_valid', hue='label', ax=axes[1])\n",
                "axes[1].set_title('SSL Certificate Validity (1=Valid, 0=Invalid, -1=No connection/HTTP)')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 3. Feature Preprocessing & Scaling\n",
                "We split our datasets into features ($X$) and labels ($y$) and standardize features using `StandardScaler`."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "feature_cols = [col for col in train_df.columns if col not in ['url', 'label']]\n",
                "\n",
                "X_train = train_df[feature_cols]\n",
                "y_train = train_df['label']\n",
                "X_test = test_df[feature_cols]\n",
                "y_test = test_df['label']\n",
                "\n",
                "scaler = StandardScaler()\n",
                "X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=feature_cols)\n",
                "X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=feature_cols)\n",
                "\n",
                "print(f\"Preprocessed {len(feature_cols)} features:\")\n",
                "print(feature_cols)"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 4. Model Training & Baseline Comparison\n",
                "We will compare three models: \n",
                "1. **Logistic Regression** (Linear baseline)\n",
                "2. **Random Forest** (Tree-based ensemble)\n",
                "3. **Gradient Boosting** (HistGradientBoostingClassifier)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Fit Models\n",
                "lr = LogisticRegression(max_iter=1000, random_state=42)\n",
                "lr.fit(X_train_scaled, y_train)\n",
                "\n",
                "rf = RandomForestClassifier(n_estimators=100, random_state=42)\n",
                "rf.fit(X_train_scaled, y_train)\n",
                "\n",
                "gb = HistGradientBoostingClassifier(random_state=42)\n",
                "gb.fit(X_train_scaled, y_train)"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Evaluate on test set\n",
                "models = {'Logistic Regression': lr, 'Random Forest': rf, 'Gradient Boosting': gb}\n",
                "results = {}\n",
                "\n",
                "for name, model in models.items():\n",
                "    probs = model.predict_proba(X_test_scaled)[:, 1]\n",
                "    preds = model.predict(X_test_scaled)\n",
                "    \n",
                "    report = classification_report(y_test, preds, output_dict=True)\n",
                "    auc = roc_auc_score(y_test, probs)\n",
                "    \n",
                "    phish_rep = report.get('1', report.get('1.0', {}))\n",
                "    results[name] = {\n",
                "        'Accuracy': report['accuracy'],\n",
                "        'Precision (Phish)': phish_rep['precision'],\n",
                "        'Recall (Phish)': phish_rep['recall'],\n",
                "        'F1-Score (Phish)': phish_rep['f1-score'],\n",
                "        'ROC-AUC': auc,\n",
                "        'probs': probs\n",
                "    }\n",
                "\n",
                "results_df = pd.DataFrame(results).T\n",
                "results_df[['Accuracy', 'Precision (Phish)', 'Recall (Phish)', 'F1-Score (Phish)', 'ROC-AUC']]"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 5. Precision-Recall Tradeoff & Threshold Tuning\n",
                "Phishing detection places a high cost on missed phishing attempts (false negatives). To minimize this cost, we can lower the classification threshold to increase **Recall** for the phishing class. Let's plot the Precision-Recall curve and ROC curve."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "fig, axes = plt.subplots(1, 2, figsize=(18, 7))\n",
                "\n",
                "# Plot ROC curves\n",
                "for name, model in models.items():\n",
                "    probs = results[name]['probs']\n",
                "    fpr, tpr, _ = roc_curve(y_test, probs)\n",
                "    axes[0].plot(fpr, tpr, label=f\"{name} (AUC = {results[name]['ROC-AUC']:.4f})\")\n",
                "\n",
                "axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.5)\n",
                "axes[0].set_xlabel('False Positive Rate')\n",
                "axes[0].set_ylabel('True Positive Rate')\n",
                "axes[0].set_title('ROC Curve')\n",
                "axes[0].legend()\n",
                "\n",
                "# Plot Precision-Recall curves\n",
                "for name, model in models.items():\n",
                "    probs = results[name]['probs']\n",
                "    precision, recall, _ = precision_recall_curve(y_test, probs)\n",
                "    axes[1].plot(recall, precision, label=f\"{name}\")\n",
                "\n",
                "axes[1].set_xlabel('Recall')\n",
                "axes[1].set_ylabel('Precision')\n",
                "axes[1].set_title('Precision-Recall Curve')\n",
                "axes[1].legend()\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Let's explore metrics at different thresholds for our best model (Random Forest / Gradient Boosting)\n",
                "best_model_name = results_df['F1-Score (Phish)'].idxmax()\n",
                "best_model = models[best_model_name]\n",
                "probs = results[best_model_name]['probs']\n",
                "\n",
                "print(f\"Threshold analysis for best model: {best_model_name}\")\n",
                "thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]\n",
                "thresh_records = []\n",
                "\n",
                "for t in thresholds:\n",
                "    preds = (probs >= t).astype(int)\n",
                "    p, r, f1, _ = precision_recall_curve(y_test, probs)\n",
                "    # find closest indices for calculation\n",
                "    report = classification_report(y_test, preds, output_dict=True)\n",
                "    phish_rep = report.get('1', report.get('1.0', {}))\n",
                "    thresh_records.append({\n",
                "        'Threshold': t,\n",
                "        'Precision (Phish)': phish_rep['precision'],\n",
                "        'Recall (Phish)': phish_rep['recall'],\n",
                "        'F1-Score (Phish)': phish_rep['f1-score'],\n",
                "        'False Negatives (Missed Phish)': confusion_matrix(y_test, preds)[1, 0]\n",
                "    })\n",
                "\n",
                "thresh_df = pd.DataFrame(thresh_records)\n",
                "thresh_df"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 6. Feature Importance & Interpretation\n",
                "Let's look at the global feature importances of the best-performing model to understand which signals are most indicative of phishing."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "if best_model_name == 'Logistic Regression':\n",
                "    importances = best_model.coef_[0]\n",
                "    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})\n",
                "    imp_df['AbsImportance'] = imp_df['Importance'].abs()\n",
                "    imp_df = imp_df.sort_values(by='AbsImportance', ascending=False)\n",
                "else:\n",
                "    importances = best_model.feature_importances_\n",
                "    imp_df = pd.DataFrame({'Feature': feature_cols, 'Importance': importances})\n",
                "    imp_df = imp_df.sort_values(by='Importance', ascending=False)\n",
                "\n",
                "plt.figure(figsize=(12, 6))\n",
                "sns.barplot(data=imp_df.head(10), x='Importance', y='Feature')\n",
                "plt.title(f'Top Features in URL Phishing Detection ({best_model_name})')\n",
                "plt.xlabel('Importance/Weight Value')\n",
                "plt.ylabel('Feature Name')\n",
                "plt.show()"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "### 7. Live Inference Example\n",
                "Let's run our saved inference bundle on a test URL and look at the output."
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": [
                "# Run a prediction and view its breakdown\n",
                "!python predict.py \"https://www.google.com\"\n",
                "print(\"\\n\" + \"=\"*80 + \"\\n\")\n",
                "!python predict.py \"http://verify-secure-paypal.update-account.com/login\""
            ]
        }
    ],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 2
}

with open("/Users/manishteotia/.gemini/antigravity/scratch/phishing_detector/phishing_detector.ipynb", "w") as f:
    json.dump(notebook, f, indent=2)
print("Created phishing_detector.ipynb successfully!")
