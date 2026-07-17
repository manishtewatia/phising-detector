import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import tldextract
from features import extract_features, POPULAR_BRANDS

# Custom CSS for modern design and dark/glassmorphic aesthetics
st.markdown("""
    <style>
    .main {
        background-color: #0d1117;
        color: #c9d1d9;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    }
    .title-container {
        text-align: center;
        background: linear-gradient(135deg, #1f6feb 0%, #8957e5 100%);
        padding: 30px;
        border-radius: 15px;
        margin-bottom: 25px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.4);
    }
    .title-text {
        font-size: 38px;
        font-weight: 800;
        color: #ffffff;
        margin: 0;
    }
    .subtitle-text {
        font-size: 16px;
        color: #e6edf3;
        margin-top: 10px;
        opacity: 0.9;
    }
    .card {
        background-color: #161b22;
        border: 1px solid #30363d;
        border-radius: 12px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
    }
    .result-danger {
        border-left: 6px solid #f85149;
        background: linear-gradient(90deg, #1b0c0f 0%, #161b22 100%);
    }
    .result-safe {
        border-left: 6px solid #56d364;
        background: linear-gradient(90deg, #0e1e13 0%, #161b22 100%);
    }
    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: bold;
        font-size: 14px;
        display: inline-block;
    }
    .badge-danger {
        background-color: rgba(248, 81, 73, 0.2);
        color: #f85149;
        border: 1px solid #f85149;
    }
    .badge-safe {
        background-color: rgba(86, 211, 100, 0.2);
        color: #56d364;
        border: 1px solid #56d364;
    }
    .metric-value {
        font-size: 54px;
        font-weight: 900;
        margin: 10px 0;
    }
    .metric-danger {
        color: #f85149;
    }
    .metric-safe {
        color: #56d364;
    }
    .factor-item {
        margin-bottom: 8px;
        font-size: 14px;
        display: flex;
        align-items: center;
    }
    .factor-icon-danger {
        color: #f85149;
        margin-right: 8px;
        font-size: 16px;
    }
    .factor-icon-safe {
        color: #56d364;
        margin-right: 8px;
        font-size: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# App Title Header
st.markdown("""
    <div class="title-container">
        <div class="title-text">🛡️ Phishing URL Guard</div>
        <div class="subtitle-text">Real-time Machine Learning Phishing URL Classifier</div>
    </div>
""", unsafe_allow_html=True)

# Helper function to load model bundle
@st.cache_resource
def load_bundle():
    bundle_path = "model_bundle.joblib"
    if os.path.exists(bundle_path):
        return joblib.load(bundle_path)
    return None

bundle = load_bundle()

if not bundle:
    st.error("⚠️ Error: Model bundle (`model_bundle.joblib`) not found. Please run the model training script (`train.py`) first to generate the classifier.")
    st.stop()

model = bundle['model']
scaler = bundle['scaler']
feature_cols = bundle['feature_cols']
model_name = bundle['model_name']
threshold = bundle['threshold']

# Description layout
st.markdown(f"""
    <div class="card">
        <h3>Pipeline Information</h3>
        <p>This web dashboard queries a pre-trained <strong>{model_name}</strong> model. 
        It extracts 14 URL structures, text heuristics, typosquatting features, and makes a live SSL socket request on port 443 to evaluate certificate validity.</p>
        <p><em>Note: WHOIS port-43 checks are automatically bypassed and return -1 in networks where WHOIS traffic is blocked.</em></p>
    </div>
""", unsafe_allow_html=True)

# Quick URL test triggers
st.markdown("### Test Sample URLs")
col_t1, col_t2 = st.columns(2)
test_url = ""

with col_t1:
    if st.button("🟢 Test Safe URL (Google)", use_container_width=True):
        test_url = "https://www.google.com/search?q=security"

with col_t2:
    if st.button("🔴 Test Phishing URL (Suspicious Update)", use_container_width=True):
        test_url = "http://verify-secure-paypal.update-account.com/login"

# Main Text Input
url_input = st.text_input("Enter URL to scan:", value=test_url, placeholder="https://example.com/login")

if url_input:
    url_cleaned = url_input.strip()
    
    with st.spinner("🔍 Extracting features and validating SSL certificate..."):
        try:
            # 1. Extract live features
            raw_feats = extract_features(url_cleaned, check_network=True)
            
            # 2. Build DataFrame
            df_raw = pd.DataFrame([raw_feats])[feature_cols]
            
            # 3. Scale Features
            scaled_feats = scaler.transform(df_raw)
            df_scaled = pd.DataFrame(scaled_feats, columns=feature_cols)
            
            # 4. Predict probability
            prob = model.predict_proba(df_scaled)[0, 1]
            is_phishing = prob >= threshold
            
            # Show Results in beautiful card
            card_class = "result-danger" if is_phishing else "result-safe"
            badge_class = "badge-danger" if is_phishing else "badge-safe"
            status_text = "SUSPICIOUS / RISK DETECTED" if is_phishing else "LEGITIMATE / SAFE"
            metric_class = "metric-danger" if is_phishing else "metric-safe"
            
            st.markdown(f"""
                <div class="card {card_class}">
                    <span class="badge {badge_class}">{status_text}</span>
                    <div class="subtitle-text">Phishing Probability Score</div>
                    <div class="metric-value {metric_class}">{prob * 100:.1f}%</div>
                    <p style="margin: 0; color: #8b949e;">Classifier Decision Threshold: {threshold * 100:.0f}% (Recall optimized)</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Detailed Breakdown Section
            st.markdown("### Risk Analysis Breakdown")
            
            col_left, col_right = st.columns(2)
            
            with col_left:
                st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("#### 🚨 Key Risk Signals")
                
                # Check for various flags
                ext = tldextract.extract(url_cleaned)
                domain_part = ext.domain.lower()
                
                risk_count = 0
                
                # Flag 1: IP Address
                if raw_feats['is_ip_domain'] == 1:
                    st.markdown('<div class="factor-item"><span class="factor-icon-danger">🔴</span> Using IP address instead of domain name</div>', unsafe_allow_html=True)
                    risk_count += 1
                
                # Flag 2: HTTPS absence
                if raw_feats['has_https'] == 0:
                    st.markdown('<div class="factor-item"><span class="factor-icon-danger">🔴</span> Connection uses insecure HTTP protocol</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                # Flag 3: SSL validity
                if raw_feats['cert_valid'] == 0:
                    st.markdown('<div class="factor-item"><span class="factor-icon-danger">🔴</span> SSL certificate validation failed (invalid/untrusted)</div>', unsafe_allow_html=True)
                    risk_count += 1
                elif raw_feats['cert_valid'] == -1 and raw_feats['has_https'] == 1:
                    st.markdown('<div class="factor-item"><span class="factor-icon-danger">🔴</span> Failed to connect to secure server (port 443)</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                # Flag 4: Typosquatting
                if 0 < raw_feats['min_levenshtein_popular'] <= 2:
                    # Find closest brand
                    closest_brand = ""
                    min_dist = 99
                    for brand in POPULAR_BRANDS:
                        d = abs(len(domain_part) - len(brand)) # quick filter
                        # full levenshtein
                        from Levenshtein import distance
                        dist = distance(domain_part, brand)
                        if dist < min_dist:
                            min_dist = dist
                            closest_brand = brand
                    st.markdown(f'<div class="factor-item"><span class="factor-icon-danger">🔴</span> Potential Typosquatting (mimics brand <b>{closest_brand}</b>)</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                # Flag 5: Shorteners
                if raw_feats['is_shortened'] == 1:
                    st.markdown('<div class="factor-item"><span class="factor-icon-danger">🔴</span> URL shortener redirects active</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                # Flag 6: Suspicious Keywords
                if raw_feats['num_suspicious_keywords'] > 0:
                    st.markdown(f'<div class="factor-item"><span class="factor-icon-danger">🔴</span> Contains {raw_feats["num_suspicious_keywords"]} suspicious keywords</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                # Flag 7: Structure
                if raw_feats['subdomain_count'] >= 2:
                    st.markdown(f'<div class="factor-item"><span class="factor-icon-danger">🔴</span> Suspicious subdomain count ({raw_feats["subdomain_count"]})</div>', unsafe_allow_html=True)
                    risk_count += 1
                if raw_feats['num_slashes'] >= 4:
                    st.markdown(f'<div class="factor-item"><span class="factor-icon-danger">🔴</span> Deep directory path ({raw_feats["num_slashes"]} slashes)</div>', unsafe_allow_html=True)
                    risk_count += 1
                    
                if risk_count == 0:
                    st.markdown('<p style="color:#8b949e;">No critical heuristic risk flags triggered.</p>', unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
            with col_right:
                st.markdown('<div class="card" style="height: 100%;">', unsafe_allow_html=True)
                st.markdown("#### 🟢 Trust Factors")
                
                trust_count = 0
                
                # Trust 1: Exact popular brand
                if raw_feats['min_levenshtein_popular'] == 0:
                    st.markdown('<div class="factor-item"><span class="factor-icon-safe">🟢</span> Domain matches known popular trusted brand</div>', unsafe_allow_html=True)
                    trust_count += 1
                    
                # Trust 2: Valid HTTPS certificate
                if raw_feats['cert_valid'] == 1:
                    st.markdown('<div class="factor-item"><span class="factor-icon-safe">🟢</span> Secure HTTPS connection with a valid SSL certificate</div>', unsafe_allow_html=True)
                    trust_count += 1
                    
                # Trust 3: Simple structure
                if raw_feats['subdomain_count'] <= 1 and raw_feats['num_slashes'] <= 3:
                    st.markdown('<div class="factor-item"><span class="factor-icon-safe">🟢</span> Simple URL path & structure</div>', unsafe_allow_html=True)
                    trust_count += 1
                    
                # Trust 4: No suspicious terms
                if raw_feats['num_suspicious_keywords'] == 0:
                    st.markdown('<div class="factor-item"><span class="factor-icon-safe">🟢</span> Contains no phishing terms (login, secure, account)</div>', unsafe_allow_html=True)
                    trust_count += 1
                    
                if trust_count == 0:
                    st.markdown('<p style="color:#8b949e;">No standard trust indicators found.</p>', unsafe_allow_html=True)
                    
                st.markdown('</div>', unsafe_allow_html=True)
                
            # Extracted Feature Values Table
            st.markdown("### Raw Features Extracted")
            feat_display = {k: [v] for k, v in raw_feats.items()}
            st.dataframe(pd.DataFrame(feat_display).T.rename(columns={0: "Feature Value"}), use_container_width=True)

        except Exception as e:
            st.error(f"Failed to scan URL: {e}")
