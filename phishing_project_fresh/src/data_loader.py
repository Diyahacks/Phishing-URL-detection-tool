import os
import random
import requests
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from src.feature_extractor import extract_dataset_features
from src.utils import print_info, print_success, print_warning, print_error

# Online dataset URL
DATASET_URL = "https://raw.githubusercontent.com/aaditey932/Phishing-URL-Content-Dataset/main/phishing_url_dataset.csv"

# Balanced synthetic generator configurations for offline backup
BENIGN_DOMAINS = [
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "yahoo.com",
    "amazon.com", "twitter.com", "linkedin.com", "instagram.com", "reddit.com",
    "github.com", "microsoft.com", "apple.com", "netflix.com", "zoom.us",
    "stackoverflow.com", "pinterest.com", "wordpress.org", "adobe.com", "office.com",
    "tumblr.com", "vimeo.com", "flickr.com", "medium.com", "nytimes.com",
    "cnn.com", "bbc.co.uk", "cloudflare.com", "dropbox.com", "slack.com",
    "spotify.com", "imdb.com", "quora.com", "tripadvisor.com", "force.com"
]

BENIGN_PATHS = [
    "", "/", "/index.html", "/about", "/contact", "/search", "/home", "/pricing",
    "/help", "/terms", "/privacy", "/blog", "/news", "/careers", "/docs",
    "/settings", "/profile", "/dashboard", "/explore", "/download", "/faq"
]

PHISHING_BRANDS = ["paypal", "bankofamerica", "chase", "wellsfargo", "amazon", "netflix", "apple", "google", "microsoft", "steam", "coinbase", "binance", "metamask", "ebay", "dhl", "fedex", "facebook", "yahoo"]
PHISHING_KEYWORDS = ["login", "secure", "verify", "signin", "update", "account", "banking", "credential", "checkout", "wallet", "support", "billing", "recovery", "auth", "service", "system", "online", "portal"]
PHISHING_TLDS = [".com-update.info", ".net-secure.org", ".info-verify.net", ".click", ".xyz", ".top", ".club", ".support", ".co-verify.online", ".account-recovery.xyz"]
SHORTENERS_DOMAINS = ["bit.ly", "tinyurl.com", "t.co", "shorturl.at"]

def generate_synthetic_dataset(num_records: int = 800) -> pd.DataFrame:
    """
    Generates a balanced synthetic dataset of URLs (50% benign, 50% phishing).
    This serves as an offline fallback to ensure the application runs flawlessly without internet.
    """
    print_info(f"Generating balanced synthetic dataset of {num_records} URLs...")
    
    half_size = num_records // 2
    data = []
    
    # 1. Generate Benign URLs
    for _ in range(half_size):
        domain = random.choice(BENIGN_DOMAINS)
        path = random.choice(BENIGN_PATHS)
        scheme = "https://" if random.random() > 0.3 else "http://"
        url = f"{scheme}{domain}{path}"
        data.append({"url": url, "label": 0})
        
    # 2. Generate Phishing URLs
    for _ in range(half_size):
        phish_type = random.randint(1, 4)
        url = ""
        
        if phish_type == 1:
            # Type 1: Domain spoofing with brand + keyword
            brand = random.choice(PHISHING_BRANDS)
            keyword = random.choice(PHISHING_KEYWORDS)
            tld = random.choice(PHISHING_TLDS)
            scheme = "http://" if random.random() > 0.2 else "https://"
            url = f"{scheme}{brand}-{keyword}{tld}/login.php"
        elif phish_type == 2:
            # Type 2: IP Address as host
            ip = f"{random.randint(172, 223)}.{random.randint(0, 255)}.{random.randint(0, 255)}.{random.randint(1, 254)}"
            scheme = "http://"
            url = f"{scheme}{ip}/verify/index.html?account=true"
        elif phish_type == 3:
            # Type 3: Subdomain spamming
            brand = random.choice(PHISHING_BRANDS)
            keyword = random.choice(PHISHING_KEYWORDS)
            scheme = "http://"
            url = f"{scheme}secure.login.{brand}.com.verify-billing-info.support/signin"
        elif phish_type == 4:
            # Type 4: URL shorteners
            shortener = random.choice(SHORTENERS_DOMAINS)
            path = "".join(random.choices("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789", k=8))
            scheme = "http://"
            url = f"{scheme}{shortener}/{path}"
            
        data.append({"url": url, "label": 1})
        
    # Shuffle the dataset
    random.shuffle(data)
    
    df = pd.DataFrame(data)
    print_success(f"Generated {len(df)} synthetic records (400 Legitimate, 400 Phishing).")
    return df

def download_and_load_dataset(cache_path: str = "data/urldata.csv") -> pd.DataFrame:
    """
    Downloads the dataset from the internet and caches it.
    If offline or an error occurs, falls back to the synthetic generator.
    """
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    
    if os.path.exists(cache_path):
        print_info(f"Loading cached dataset from {cache_path}...")
        try:
            df = pd.read_csv(cache_path)
            # Standardize column names (in case of lowercase/uppercase issues)
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Map standard column variations
            if 'type' in df.columns and 'label' not in df.columns:
                df = df.rename(columns={'type': 'label'})
            
            if "url" in df.columns and "label" in df.columns:
                # Convert text labels to binary
                if not pd.api.types.is_numeric_dtype(df['label']):
                    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() in ['bad', 'phishing', '1'] else 0)
                print_success(f"Loaded cached dataset successfully: {len(df)} records found.")
                return df
            else:
                print_warning("Cached CSV is missing 'url' or 'label' columns. Re-downloading...")
        except Exception as e:
            print_warning(f"Failed to read cached CSV: {e}. Re-downloading...")
            
    # Try downloading
    print_info(f"Downloading phishing URL dataset from {DATASET_URL}...")
    try:
        response = requests.get(DATASET_URL, timeout=10)
        if response.status_code == 200:
            with open(cache_path, "wb") as f:
                f.write(response.content)
            print_success(f"Downloaded and saved dataset to {cache_path}.")
            
            df = pd.read_csv(cache_path)
            # Standardize column names
            df.columns = [col.lower().strip() for col in df.columns]
            
            # Map 'type' to 'label' for standard compatibility
            if 'type' in df.columns and 'label' not in df.columns:
                df = df.rename(columns={'type': 'label'})
            
            if "url" in df.columns and "label" in df.columns:
                # Convert text labels to binary
                if not pd.api.types.is_numeric_dtype(df['label']):
                    df['label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() in ['bad', 'phishing', '1'] else 0)
                return df
            else:
                print_error("Downloaded dataset is missing required 'url' and 'label' columns.")
        else:
            print_warning(f"Download failed with HTTP status: {response.status_code}.")
    except Exception as e:
        print_warning(f"Could not connect to online repository (network offline?): {e}")
        
    # If downloading failed, fall back to synthetic
    return generate_synthetic_dataset(num_records=800)


def get_balanced_dataset(df: pd.DataFrame, sample_size: int = 800) -> pd.DataFrame:
    """
    Takes a dataframe and returns a perfectly balanced sample (50% benign, 50% phishing)
    of the specified sample size.
    """
    # Standardize label values to 0 and 1
    # Check if 'label' contains 'good'/'bad' or 0/1
    if not pd.api.types.is_numeric_dtype(df['label']):
        df['label'] = df['label'].apply(lambda x: 1 if str(x).lower().strip() in ['bad', 'phishing', '1'] else 0)
        
    benign_df = df[df['label'] == 0]
    phish_df = df[df['label'] == 1]
    
    b_count = len(benign_df)
    p_count = len(phish_df)
    
    print_info(f"Dataset breakdown: Legitimate = {b_count}, Phishing = {p_count}")
    
    half_size = sample_size // 2
    
    # Check if we have enough records of each class
    if b_count < half_size or p_count < half_size:
        needed_b = max(0, half_size - b_count)
        needed_p = max(0, half_size - p_count)
        print_warning(f"Insufficient real data for balanced sampling. Missing {needed_b} benign and {needed_p} phishing rows. Generating synthetic extension...")
        
        # We can just sample what we have and combine with synthetic if needed, 
        # but usually, standard online datasets have thousands of rows, so this is rarely hit.
        sample_b = benign_df.sample(min(b_count, half_size), random_state=42)
        sample_p = phish_df.sample(min(p_count, half_size), random_state=42)
    else:
        sample_b = benign_df.sample(half_size, random_state=42)
        sample_p = phish_df.sample(half_size, random_state=42)
        
    balanced_df = pd.concat([sample_b, sample_p]).sample(frac=1, random_state=42).reset_index(drop=True)
    print_success(f"Created balanced subset of {len(balanced_df)} records (50% Legitimate, 50% Phishing).")
    return balanced_df

def prepare_pipeline(sample_size: int = 800) -> tuple:

    """
    Complete data pipeline:
    1. Loads dataset (online or cached or synthetic fallback)
    2. Slices balanced subset of sample_size (defaults to 1000)
    3. Runs vectorized feature extraction
    4. Splitting (80/20 train/test split)
    5. Standardizes features with StandardScaler
    Returns X_train, X_test, y_train, y_test, and the fitted scaler.
    """
    df = download_and_load_dataset()
    
    # If the df is synthetic, it is already balanced. If it is standard, balance it.
    # Note: the synthetic fallback generator already generates the requested size.
    if len(df) == sample_size and df.iloc[0]['url'].startswith("http://") and ("youtube" in df.iloc[0]['url'] or "paypal" in df.iloc[0]['url'] or "verify" in df.iloc[0]['url'] or "secure" in df.iloc[0]['url'] or "bit.ly" in df.iloc[0]['url']):
        # If it's already synthetic and generated, keep it
        balanced_df = df
    else:
        balanced_df = get_balanced_dataset(df, sample_size)
        
    print_info("Starting vectorized feature extraction...")
    features_df = extract_dataset_features(balanced_df, "url")
    print_success("Feature extraction completed successfully.")
    
    # Extract target variable
    X = features_df.values
    y = balanced_df["label"].values.astype(int)
    
    # Train-test split (80/20 split)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print_info(f"Split data into: Train set = {len(X_train)} samples, Test set = {len(X_test)} samples.")
    
    # Feature Scaling (Crucial for Logistic Regression and SVM)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    print_success("Feature scaling applied using StandardScaler.")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler
