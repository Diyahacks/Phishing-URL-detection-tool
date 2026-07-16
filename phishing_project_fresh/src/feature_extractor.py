import re
from urllib.parse import urlparse
import pandas as pd
import numpy as np

# List of common URL shortening services
SHORTENERS = {
    "bit.ly", "tinyurl.com", "t.co", "goo.gl", "rebrand.ly", "tiny.cc", "is.gd", 
    "ow.ly", "buff.ly", "adf.ly", "bit.do", "lnkd.in", "db.tt", "qr.ae", 
    "shorte.st", "mstd.hn", "t.ly", "cutt.ly", "clck.ru", "shorturl.at"
}

# Whitelist of globally trusted popular domains to prevent false positives
WHITELISTED_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "wikipedia.org", "yahoo.com",
    "amazon.com", "twitter.com", "linkedin.com", "instagram.com", "reddit.com",
    "github.com", "microsoft.com", "apple.com", "netflix.com", "zoom.us",
    "stackoverflow.com", "pinterest.com", "wordpress.org", "adobe.com", "office.com",
    "tumblr.com", "vimeo.com", "flickr.com", "medium.com", "nytimes.com",
    "cnn.com", "bbc.co.uk", "cloudflare.com", "dropbox.com", "slack.com",
    "spotify.com", "imdb.com", "quora.com", "tripadvisor.com", "force.com",
    "google.co.in", "google.co.uk", "google.com.br", "apple.co", "apple.com.cn"
}

# List of common keywords used in phishing attacks
SUSPICIOUS_KEYWORDS = [
    "login", "admin", "secure", "account", "banking", "signin", "update", 
    "verify", "webscr", "ebayisapi", "wp", "client", "feedback", "paypal", 
    "service", "free", "gift", "claim", "bonus", "security", "confirm"
]

# Regex pattern to check if domain is an IP address (IPv4, Hex, or simplified IPv6)
IP_PATTERN = re.compile(
    r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$|'  # IPv4
    r'^0x[0-9a-fA-F]+\.[0-9a-fA-F]+\.[0-9a-fA-F]+\.[0-9a-fA-F]+$|'  # Hex IP
    r'^\[?[0-9a-fA-F:]+\]?$'  # IPv6
)

# Ordered feature list to ensure consistency in prediction matrix
FEATURE_NAMES = [
    "url_length",
    "hostname_length",
    "path_length",
    "qty_dot",
    "qty_hyphen",
    "qty_at",
    "qty_question",
    "qty_equal",
    "qty_slash",
    "qty_digit",
    "qty_letter",
    "has_ip",
    "has_shortening",
    "has_http_in_path",
    "suspicious_words_count"
]

def preprocess_url(url: str) -> str:
    """
    Cleans and prepends http scheme if missing to ensure reliable parsing.
    """
    url = url.strip()
    # If the URL is empty or malformed
    if not url:
        return "http://localhost"
    
    # If URL doesn't start with scheme, prepend 'http://'
    if not re.match(r"^https?://", url, re.IGNORECASE):
        url = "http://" + url
    return url

def is_whitelisted(url: str) -> bool:
    """
    Checks if the root domain of a URL is in the globally trusted whitelist.
    """
    cleaned_url = preprocess_url(url)
    try:
        parsed = urlparse(cleaned_url)
        domain = parsed.netloc.lower()
        if domain.startswith("www."):
            domain = domain[4:]
        
        # Check direct match
        if domain in WHITELISTED_DOMAINS:
            return True
            
        # Check subdomains (e.g. mail.google.com -> google.com)
        for trusted in WHITELISTED_DOMAINS:
            if domain.endswith("." + trusted):
                return True
    except Exception:
        pass
    return False

def extract_features(url: str) -> dict:
    """
    Extracts lexical features from a single raw URL.
    Returns a dictionary of features mapping FEATURE_NAMES to numerical values.
    """
    cleaned_url = preprocess_url(url)
    
    try:
        parsed = urlparse(cleaned_url)
        domain = parsed.netloc if parsed.netloc else ""
        path = parsed.path if parsed.path else ""
        path_query = path + (parsed.query if parsed.query else "") + (parsed.fragment if parsed.fragment else "")
    except Exception:
        domain = ""
        path = ""
        path_query = ""
    
    features = {}
    
    # 1. Length-based features
    features["url_length"] = len(cleaned_url)
    features["hostname_length"] = len(domain)
    features["path_length"] = len(path)
    
    # 2. Character counts
    features["qty_dot"] = cleaned_url.count('.')
    features["qty_hyphen"] = cleaned_url.count('-')
    features["qty_at"] = cleaned_url.count('@')
    features["qty_question"] = cleaned_url.count('?')
    features["qty_equal"] = cleaned_url.count('=')
    features["qty_slash"] = cleaned_url.count('/')
    
    # 3. Digits and Letters count
    features["qty_digit"] = sum(c.isdigit() for c in cleaned_url)
    features["qty_letter"] = sum(c.isalpha() for c in cleaned_url)
    
    # 4. IP address in domain
    features["has_ip"] = 1 if IP_PATTERN.match(domain) else 0
    
    # 5. Shortening service check
    features["has_shortening"] = 1 if domain.lower() in SHORTENERS or any(sh in domain.lower() for sh in SHORTENERS if len(sh) > 4) else 0
    
    # 6. Http/Https spoofing in path
    features["has_http_in_path"] = 1 if "http" in path_query.lower() or "https" in path_query.lower() else 0
    
    # 7. Suspicious keywords count (case-insensitive)
    url_lower = cleaned_url.lower()
    features["suspicious_words_count"] = sum(url_lower.count(word) for word in SUSPICIOUS_KEYWORDS)
    
    return features

def extract_dataset_features(df: pd.DataFrame, url_col: str = "url") -> pd.DataFrame:
    """
    Vectorized extraction of features across an entire pandas DataFrame.
    Highly optimized to process thousands of URLs in milliseconds.
    """
    # 1. Ensure URLs are parsed cleanly
    preprocessed_urls = df[url_col].apply(preprocess_url)
    
    # Create empty columns for features
    features_df = pd.DataFrame(index=df.index)
    
    # Apply fast vectorized string operations
    features_df["url_length"] = preprocessed_urls.str.len()
    
    # Parse domain and path components
    domains = []
    paths = []
    path_queries = []
    
    for url in preprocessed_urls:
        try:
            parsed = urlparse(url)
            dom = parsed.netloc if parsed.netloc else ""
            pth = parsed.path if parsed.path else ""
            pq = pth + (parsed.query if parsed.query else "") + (parsed.fragment if parsed.fragment else "")
            domains.append(dom)
            paths.append(pth)
            path_queries.append(pq)
        except Exception:
            domains.append("")
            paths.append("")
            path_queries.append("")
            
    domains_series = pd.Series(domains, index=df.index)
    paths_series = pd.Series(paths, index=df.index)
    path_queries_series = pd.Series(path_queries, index=df.index)
    
    features_df["hostname_length"] = domains_series.str.len()
    features_df["path_length"] = paths_series.str.len()
    
    features_df["qty_dot"] = preprocessed_urls.str.count(r'\.')
    features_df["qty_hyphen"] = preprocessed_urls.str.count(r'-')
    features_df["qty_at"] = preprocessed_urls.str.count(r'@')
    features_df["qty_question"] = preprocessed_urls.str.count(r'\?')
    features_df["qty_equal"] = preprocessed_urls.str.count(r'=')
    features_df["qty_slash"] = preprocessed_urls.str.count(r'/')
    
    features_df["qty_digit"] = preprocessed_urls.apply(lambda x: sum(c.isdigit() for c in x))
    features_df["qty_letter"] = preprocessed_urls.apply(lambda x: sum(c.isalpha() for c in x))
    
    features_df["has_ip"] = domains_series.apply(lambda x: 1 if IP_PATTERN.match(x) else 0)
    features_df["has_shortening"] = domains_series.apply(lambda x: 1 if x.lower() in SHORTENERS or any(sh in x.lower() for sh in SHORTENERS if len(sh) > 4) else 0)
    features_df["has_http_in_path"] = path_queries_series.apply(lambda x: 1 if "http" in x.lower() or "https" in x.lower() else 0)
    
    # Calculate suspicious words count
    features_df["suspicious_words_count"] = preprocessed_urls.apply(
        lambda x: sum(x.lower().count(word) for word in SUSPICIOUS_KEYWORDS)
    )
    
    return features_df
