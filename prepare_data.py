import os
import sys
import requests
import pandas as pd
import numpy as np
import zipfile
import io
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
from features import extract_features

# URLs for datasets
PHISHTANK_URL = "http://data.phishtank.com/data/online-valid.csv"
OPENPHISH_URL = "https://openphish.com/feed.txt"
TRANCO_URL = "https://tranco-list.eu/top-1m.csv.zip"

def download_file(url: str, filepath: str):
    """Downloads a file if it doesn't already exist locally, unzipping if necessary."""
    if os.path.exists(filepath):
        print(f"File {filepath} already exists. Skipping download.")
        return
    print(f"Downloading from {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    response = requests.get(url, headers=headers, timeout=15)
    response.raise_for_status()
    
    if url.endswith('.zip') or response.headers.get('Content-Type') == 'application/zip':
        print("Extracting zip archive...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            csv_filenames = [name for name in z.namelist() if name.endswith('.csv') or name.endswith('.txt')]
            if csv_filenames:
                data = z.read(csv_filenames[0]).decode('utf-8')
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(data)
                print(f"Extracted and saved to {filepath}")
            else:
                raise ValueError("No CSV/text file found in zip archive")
    else:
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Saved to {filepath}")

def load_datasets(data_dir: str):
    """Downloads and loads phishing and legitimate URLs."""
    os.makedirs(data_dir, exist_ok=True)
    
    phish_path = os.path.join(data_dir, "phishing_urls.txt")
    legit_path = os.path.join(data_dir, "tranco.csv")
    
    phishing_urls = []
    
    # 1. Attempt to download PhishTank
    try:
        phishtank_csv = os.path.join(data_dir, "phishtank.csv")
        download_file(PHISHTANK_URL, phishtank_csv)
        pt_df = pd.read_csv(phishtank_csv)
        if 'url' in pt_df.columns:
            phishing_urls.extend(pt_df['url'].dropna().tolist())
            print(f"Loaded {len(pt_df)} URLs from PhishTank.")
    except Exception as e:
        print(f"Failed to download or parse PhishTank: {e}")
        
    # 2. Attempt to download OpenPhish
    try:
        openphish_path = os.path.join(data_dir, "openphish.txt")
        download_file(OPENPHISH_URL, openphish_path)
        with open(openphish_path, 'r', encoding='utf-8') as f:
            op_urls = [line.strip() for line in f if line.strip()]
            phishing_urls.extend(op_urls)
            print(f"Loaded {len(op_urls)} URLs from OpenPhish.")
    except Exception as e:
        print(f"Failed to download OpenPhish: {e}")
        
    # Deduplicate phishing URLs
    phishing_urls = list(set(phishing_urls))
    
    if not phishing_urls:
        print("Warning: No phishing URLs downloaded! Using mock fallback.")
        phishing_urls = [
            "http://verify-paypal-login.com/secure",
            "http://signin.ebayisapi.update.support-account-validation.com/login",
            "http://192.168.10.45/login.php",
            "https://secure-banking-update.net/login"
        ]
        
    # 3. Download Tranco top 1M domains
    try:
        download_file(TRANCO_URL, legit_path)
    except Exception as e:
        print(f"Failed to download Tranco top 1M: {e}")
        with open(legit_path, 'w') as f:
            f.write("1,google.com\n2,facebook.com\n3,apple.com\n4,microsoft.com\n")
            
    # Load Legitimate domains and convert to URLs
    legit_df = pd.read_csv(legit_path, header=None, names=['rank', 'domain'])
    legitimate_urls = []
    
    # Common standard paths for legitimate URLs to simulate paths, slashes and lengths
    LEGIT_PATHS = [
        "",
        "/",
        "/about",
        "/contact",
        "/index.html",
        "/search?q=query",
        "/products",
        "/category/items/item123",
        "/privacy-policy",
        "/terms",
        "/news/today",
        "/help/faq.html",
        "/blog/post/2026/hello-world"
    ]
    
    for i, row in legit_df.head(10000).iterrows():
        domain = row['domain']
        
        # 50% chance of prepending www.
        if i % 2 == 0:
            domain = f"www.{domain}"
            
        scheme = "https://" if i % 3 == 0 else "http://"
        path = LEGIT_PATHS[i % len(LEGIT_PATHS)]
        
        legitimate_urls.append(f"{scheme}{domain}{path}")
        
    print(f"Final pool contains {len(phishing_urls)} phishing URLs and {len(legitimate_urls)} legitimate URLs.")
    return phishing_urls, legitimate_urls

def extract_features_parallel(urls, label, check_network=True, max_workers=20):
    """Extracts features for a list of URLs in parallel."""
    dataset = []
    total = len(urls)
    print(f"Extracting features for {total} URLs (label={label})...")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_url = {executor.submit(extract_features, url, check_network): url for url in urls}
        
        completed = 0
        for future in as_completed(future_to_url):
            url = future_to_url[future]
            try:
                features = future.result()
                features['url'] = url
                features['label'] = label  # 1 for phishing, 0 for legitimate
                dataset.append(features)
            except Exception as e:
                print(f"\nError extracting features for {url}: {e}")
            
            completed += 1
            if completed % 100 == 0 or completed == total:
                print(f"Progress: {completed}/{total} completed", end='\r')
    print() # Newline
    return dataset

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--sample-size', type=int, default=1000, 
                        help='Number of samples per class to extract')
    parser.add_argument('--no-network', action='store_true', 
                        help='Skip network checks (SSL & WHOIS)')
    args = parser.parse_args()

    data_dir = "data"
    cache_path = os.path.join(data_dir, "extracted_features_cache.csv")
    
    # Remove existing cache if it exists, to re-run with our new balanced paths
    if os.path.exists(cache_path):
        os.remove(cache_path)
        
    phishing_urls, legitimate_urls = load_datasets(data_dir)
    
    # Balance dataset and sample
    sample_size = min(args.sample_size, len(phishing_urls), len(legitimate_urls))
    print(f"Sampling {sample_size} URLs per class (Total: {2 * sample_size})")
    
    np.random.seed(42)
    phishing_sampled = np.random.choice(phishing_urls, sample_size, replace=False)
    legitimate_sampled = np.random.choice(legitimate_urls, sample_size, replace=False)
    
    # Parallel extraction
    check_network = not args.no_network
    phish_features = extract_features_parallel(phishing_sampled, label=1, check_network=check_network)
    legit_features = extract_features_parallel(legitimate_sampled, label=0, check_network=check_network)
    
    # Combine
    all_features = phish_features + legit_features
    df = pd.DataFrame(all_features)
    
    # Reorder columns to put url and label at the end
    cols = [col for col in df.columns if col not in ['url', 'label']] + ['url', 'label']
    df = df[cols]
    
    # Save cache
    df.to_csv(cache_path, index=False)
    print(f"Saved extracted features cache to {cache_path}")
        
    print(f"Dataset shape: {df.shape}")
    print(df['label'].value_counts())
    
    # Stratified split
    X_train, X_test, y_train, y_test = train_test_split(
        df, df['label'], test_size=0.20, random_state=42, stratify=df['label']
    )
    
    train_path = os.path.join(data_dir, "train_data.csv")
    test_path = os.path.join(data_dir, "test_data.csv")
    
    X_train.to_csv(train_path, index=False)
    X_test.to_csv(test_path, index=False)
    
    print(f"Saved train set ({len(X_train)} samples) to {train_path}")
    print(f"Saved test set ({len(X_test)} samples) to {test_path}")

if __name__ == '__main__':
    main()
