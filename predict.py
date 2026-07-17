import os
import sys
import pandas as pd
import numpy as np
import joblib
from features import extract_features

def main():
    if len(sys.argv) < 2:
        print("Usage: python predict.py <URL>")
        print("Example: python predict.py \"https://verify-paypal-login.com\"")
        sys.exit(1)
        
    url = sys.argv[1]
    bundle_path = "model_bundle.joblib"
    
    if not os.path.exists(bundle_path):
        print(f"Error: Model bundle not found at {bundle_path}. Please run train.py first.")
        sys.exit(1)
        
    # Load model bundle
    bundle = joblib.load(bundle_path)
    model = bundle['model']
    scaler = bundle['scaler']
    feature_cols = bundle['feature_cols']
    model_name = bundle['model_name']
    threshold = bundle['threshold']
    
    print(f"URL to classify: {url}")
    print(f"Using trained model: {model_name} (Threshold: {threshold})")
    print("Extracting features (running live SSL checks)...")
    
    # Extract features (enable network check for live SSL validation)
    raw_feats = extract_features(url, check_network=True)
    
    # Create DataFrame for scaling (ensuring correct column order)
    df_raw = pd.DataFrame([raw_feats])[feature_cols]
    
    # Scale features
    scaled_feats = scaler.transform(df_raw)
    df_scaled = pd.DataFrame(scaled_feats, columns=feature_cols)
    
    # Predict probability
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(df_scaled)[0, 1]
    else:
        prob = model.predict(df_scaled)[0]
        
    # Classification decision
    is_phishing = prob >= threshold
    
    print("\n" + "="*50)
    print("                    RESULTS")
    print("="*50)
    print(f"Phishing Probability: {prob * 100:.2f}%")
    if is_phishing:
        print("Status: RED / DANGER - Suspicious URL (High Phishing Risk)")
    else:
        print("Status: GREEN / SAFE - Legitimate URL")
    print("="*50)
    
    # Calculate feature contributions
    contributions = []
    
    if model_name == "Logistic Regression":
        coefs = model.coef_[0]
        intercept = model.intercept_[0]
        
        # Contribution to log-odds: coefficient * scaled_value
        for i, col in enumerate(feature_cols):
            scaled_val = scaled_feats[0, i]
            raw_val = df_raw.iloc[0, i]
            contrib = coefs[i] * scaled_val
            contributions.append({
                'feature': col,
                'raw_value': raw_val,
                'scaled_value': scaled_val,
                'weight': coefs[i],
                'contribution': contrib
            })
            
        # Sort contributions to see which features pushed it towards phishing vs legitimate
        contributions = sorted(contributions, key=lambda x: x['contribution'], reverse=True)
        
        print("\nTOP FEATURE CONTRIBUTIONS TO PHISHING RISK:")
        print(f"{'Feature':<25} | {'Raw Value':<10} | {'Impact (Log-Odds)':<18}")
        print("-"*60)
        for c in contributions:
            impact_sign = "+" if c['contribution'] >= 0 else ""
            print(f"{c['feature']:<25} | {str(c['raw_value']):<10} | {impact_sign}{c['contribution']:.4f}")
            
    else:
        # Approximate contribution using feature importance and feature value
        importances = model.feature_importances_ if hasattr(model, "feature_importances_") else None
        
        if importances is not None:
            for i, col in enumerate(feature_cols):
                scaled_val = scaled_feats[0, i]
                raw_val = df_raw.iloc[0, i]
                contrib = importances[i] * scaled_val
                contributions.append({
                    'feature': col,
                    'raw_value': raw_val,
                    'importance': importances[i],
                    'contribution': contrib
                })
            contributions = sorted(contributions, key=lambda x: abs(x['contribution']), reverse=True)
            
            print("\nKEY CONTRIBUTING FEATURES (Sorted by magnitude):")
            print(f"{'Feature':<25} | {'Raw Value':<10} | {'Global Importance':<18}")
            print("-"*60)
            for c in contributions:
                print(f"{c['feature']:<25} | {str(c['raw_value']):<10} | {c['importance']:.4f}")
        else:
            print("\nFeature values:")
            for col in feature_cols:
                print(f"  {col}: {raw_feats[col]}")

if __name__ == '__main__':
    main()
