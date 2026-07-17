import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
import joblib

def load_data():
    train_df = pd.read_csv("data/train_data.csv")
    test_df = pd.read_csv("data/test_data.csv")
    
    X_train = train_df.drop(columns=['url', 'label'])
    y_train = train_df['label']
    
    X_test = test_df.drop(columns=['url', 'label'])
    y_test = test_df['label']
    
    return X_train, X_test, y_train, y_test

def evaluate_model(name, model, X_test, y_test, threshold=0.5):
    # Predict probabilities
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)
        
    preds = (probs >= threshold).astype(int)
    
    # Calculate metrics
    report = classification_report(y_test, preds, output_dict=True)
    auc = roc_auc_score(y_test, probs)
    
    # Focus on phishing class (1)
    phish_metrics = report.get('1', report.get('1.0', {}))
    precision = phish_metrics.get('precision', 0)
    recall = phish_metrics.get('recall', 0)
    f1 = phish_metrics.get('f1-score', 0)
    accuracy = report.get('accuracy', 0)
    
    print(f"\n=================== {name} (Threshold={threshold:.2f}) ===================")
    print(classification_report(y_test, preds))
    print(f"ROC-AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, preds))
    
    return {
        'name': name,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'auc': auc,
        'probs': probs,
        'preds': preds
    }

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Features list: {list(X_train.columns)}")
    print(f"Training set size: {X_train.shape[0]} samples")
    print(f"Test set size: {X_test.shape[0]} samples")
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert to DataFrames to retain feature names
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    # 1. Logistic Regression
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_scaled_df, y_train)
    lr_eval = evaluate_model("Logistic Regression", lr, X_test_scaled_df, y_test)
    
    # 2. Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled_df, y_train)
    rf_eval = evaluate_model("Random Forest", rf, X_test_scaled_df, y_test)
    
    # 3. Gradient Boosting (HistGradientBoostingClassifier)
    gb = HistGradientBoostingClassifier(random_state=42)
    gb.fit(X_train_scaled_df, y_train)
    gb_eval = evaluate_model("Gradient Boosting (HistGB)", gb, X_test_scaled_df, y_test)
    
    # Analyze threshold adjustment to emphasize Recall for Gradient Boosting
    print("\n=================== Threshold Analysis for Gradient Boosting ===================")
    thresholds = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    
    for t in thresholds:
        eval_dict = evaluate_model("Gradient Boosting (HistGB)", gb, X_test_scaled_df, y_test, threshold=t)
        print(f"Threshold: {t:.1f} -> Recall: {eval_dict['recall']:.4f}, Precision: {eval_dict['precision']:.4f}, F1: {eval_dict['f1']:.4f}")
    
    # Select the best model based on default F1-score
    models_evals = [
        (lr, lr_eval, "Logistic Regression"),
        (rf, rf_eval, "Random Forest"),
        (gb, gb_eval, "Gradient Boosting")
    ]
    
    best_model_obj, best_eval, best_model_name = max(models_evals, key=lambda x: x[1]['f1'])
    print(f"\n>>> Best model selected based on default F1-score: {best_model_name} (F1: {best_eval['f1']:.4f}, Recall: {best_eval['recall']:.4f})")
    
    # Inspect feature importances / coefficients for explanation
    print(f"\n=================== Feature Importance / Coefficients for {best_model_name} ===================")
    if best_model_name == "Logistic Regression":
        importances = best_model_obj.coef_[0]
        imp_df = pd.DataFrame({'Feature': X_train.columns, 'Coefficient': importances})
        imp_df['Absolute_Weight'] = imp_df['Coefficient'].abs()
        imp_df = imp_df.sort_values(by='Absolute_Weight', ascending=False)
        print(imp_df[['Feature', 'Coefficient']])
    elif best_model_name == "Random Forest":
        importances = best_model_obj.feature_importances_
        imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': importances})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        print(imp_df)
    else:
        # HistGradientBoostingClassifier has permutation importance or standard feature attributes (only in newer versions, or we can use permutation importance)
        # To avoid complex computation, we'll use permutation_importance
        from sklearn.inspection import permutation_importance
        print("Computing permutation importance for Gradient Boosting (this may take a few seconds)...")
        result = permutation_importance(best_model_obj, X_test_scaled_df, y_test, n_repeats=5, random_state=42)
        imp_df = pd.DataFrame({'Feature': X_train.columns, 'Importance': result.importances_mean})
        imp_df = imp_df.sort_values(by='Importance', ascending=False)
        print(imp_df)
        
    # We will adjust threshold to 0.3 to favor recall for the phishing class
    final_threshold = 0.3
    print(f"\nSetting final classification threshold for prediction: {final_threshold}")
    
    # Save the bundle
    model_bundle = {
        'model': best_model_obj,
        'scaler': scaler,
        'feature_cols': list(X_train.columns),
        'model_name': best_model_name,
        'threshold': final_threshold
    }
    
    bundle_path = "model_bundle.joblib"
    joblib.dump(model_bundle, bundle_path)
    print(f"Saved model bundle (model, scaler, metadata) to {bundle_path}")

if __name__ == '__main__':
    main()
