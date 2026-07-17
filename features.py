import re
import ssl
import socket
import urllib.parse
import ipaddress
import tldextract
import Levenshtein
from datetime import datetime
import whois

# List of common URL shorteners
SHORTENERS = {
    'bit.ly', 'tinyurl.com', 't.co', 'rebrand.ly', 'is.gd', 'buff.ly',
    'adf.ly', 'ow.ly', 'goo.gl', 'bit.do', 'sof.li', 'lnkd.in', 'db.tt',
    'qr.ae', 'adfoc.us', 'sandbox.com', 'tiny.cc', 'shorturl.at'
}

# Suspicious keywords in URLs
SUSPICIOUS_KEYWORDS = {
    'login', 'verify', 'secure', 'account', 'update', 'signin', 
    'banking', 'confirm', 'paypal', 'webscr', 'admin', 'ebayisapi'
}

# Known popular brand names for typosquatting checks (Levenshtein distance)
POPULAR_BRANDS = [
    "google", "facebook", "apple", "microsoft", "amazon", 
    "netflix", "paypal", "yahoo", "live", "instagram", "github"
]

# One-time check for WHOIS connectivity to avoid hangs if port 43 is blocked
_WHOIS_AVAILABLE = None

def check_whois_availability() -> bool:
    global _WHOIS_AVAILABLE
    if _WHOIS_AVAILABLE is not None:
        return _WHOIS_AVAILABLE
    try:
        # Check if we can connect to a root WHOIS server on port 43
        with socket.create_connection(('whois.verisign-grs.com', 43), timeout=1.0):
            _WHOIS_AVAILABLE = True
    except Exception:
        _WHOIS_AVAILABLE = False
        print("WHOIS port 43 is blocked or unreachable. WHOIS features will be skipped (return -1).")
    return _WHOIS_AVAILABLE

def normalize_url(url: str) -> str:
    """Ensures the URL has a scheme (default to http:// if missing) for parsing."""
    url = url.strip()
    if not (url.startswith("http://") or url.startswith("https://")):
        return "http://" + url
    return url

def check_ip_address(domain: str) -> int:
    """Returns 1 if the domain name is an IP address, 0 otherwise."""
    if not domain:
        return 0
    # Strip port if present
    domain_clean = domain.split(':')[0]
    try:
        ipaddress.ip_address(domain_clean)
        return 1
    except ValueError:
        return 0

def check_shortener(domain: str) -> int:
    """Returns 1 if the domain is a known URL shortener, 0 otherwise."""
    if not domain:
        return 0
    domain_clean = domain.lower().split(':')[0]
    return 1 if domain_clean in SHORTENERS else 0

def get_min_levenshtein(domain_name: str) -> int:
    """Computes the minimum Levenshtein distance from the domain name to popular brands."""
    if not domain_name:
        return 999
    domain_clean = domain_name.lower()
    distances = [Levenshtein.distance(domain_clean, brand) for brand in POPULAR_BRANDS]
    return min(distances) if distances else 999

def check_ssl_validity(url: str, timeout: float = 2.0) -> int:
    """
    Checks the SSL certificate validity of the domain.
    Returns:
       1 if HTTPS and certificate is valid.
       0 if HTTPS but certificate is invalid (SSLError, Name Mismatch, Expired).
      -1 if HTTP or connection fails/timeouts.
    """
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() != 'https':
        return -1
    
    hostname = parsed.hostname
    if not hostname:
        return -1

    context = ssl.create_default_context()
    try:
        # Wrap connection with default secure context (verifies cert and host name)
        with socket.create_connection((hostname, 443), timeout=timeout) as sock:
            with context.wrap_socket(sock, server_hostname=hostname) as ssock:
                return 1
    except ssl.SSLCertVerificationError:
        # Connection established, but verification failed
        return 0
    except Exception:
        # Host name resolution, connection, or timeout errors
        return -1

def get_domain_age(domain: str) -> int:
    """
    Queries WHOIS to get the domain age in days.
    Returns:
       age in days, or -1 if the query fails or WHOIS data is unavailable.
    """
    if not domain:
        return -1
    
    if not check_whois_availability():
        return -1
    
    # Extract the registered domain to query WHOIS (e.g. google.com instead of sub.google.com)
    ext = tldextract.extract(f"http://{domain}")
    # Handle both old/new tldextract versions
    registered_domain = ext.top_domain_under_public_suffix or ext.registered_domain
    if not registered_domain:
        return -1
        
    try:
        socket.setdefaulttimeout(3.0)
        w = whois.whois(registered_domain)
        creation_date = w.creation_date
        
        if isinstance(creation_date, list):
            creation_date = creation_date[0]
            
        if not creation_date or not isinstance(creation_date, datetime):
            return -1
            
        age = (datetime.now() - creation_date).days
        return max(0, age)
    except Exception:
        return -1

def extract_features(url: str, check_network: bool = False) -> dict:
    """
    Extracts 14 features from a URL.
    Args:
        url: The input URL string.
        check_network: If True, executes SSL and WHOIS network queries.
    """
    normalized = normalize_url(url)
    parsed = urllib.parse.urlparse(normalized)
    
    # Use tldextract to split the domain securely
    ext = tldextract.extract(normalized)
    domain_part = ext.domain  # e.g., 'google'
    full_domain = parsed.netloc  # e.g., 'www.google.com'
    
    # 1. URL Length
    url_length = len(url)
    
    # 2. Number of Dots
    num_dots = url.count('.')
    
    # 3. Number of Hyphens
    num_hyphens = url.count('-')
    
    # 4. Number of Slashes
    num_slashes = url.count('/')
    
    # 5. Presence of IP Address
    is_ip_domain = check_ip_address(full_domain)
    
    # 6. Use of URL Shorteners
    is_shortened = check_shortener(full_domain)
    
    # 7. Suspicious Keywords count
    url_lower = url.lower()
    num_suspicious_keywords = sum(1 for kw in SUSPICIOUS_KEYWORDS if kw in url_lower)
    
    # 8. HTTPS Presence
    has_https = 1 if parsed.scheme.lower() == 'https' else 0
    
    # 9. SSL Validity
    cert_valid = -1
    if check_network and has_https:
        cert_valid = check_ssl_validity(normalized)
        
    # 10. Domain Age
    domain_age_days = -1
    if check_network and full_domain and not is_ip_domain:
        domain_age_days = get_domain_age(full_domain)
        
    # 11. Subdomain Count
    subdomain_count = 0
    if ext.subdomain:
        subdomain_count = len(ext.subdomain.split('.'))
        
    # 12. Presence of '@' symbol
    has_at_symbol = 1 if '@' in url else 0
    
    # 13. Double slashes in path
    has_double_slash_path = 1 if '//' in parsed.path else 0
    
    # 14. Levenshtein distance to known popular domains
    min_levenshtein_popular = get_min_levenshtein(domain_part)
    
    return {
        'url_length': url_length,
        'num_dots': num_dots,
        'num_hyphens': num_hyphens,
        'num_slashes': num_slashes,
        'is_ip_domain': is_ip_domain,
        'is_shortened': is_shortened,
        'num_suspicious_keywords': num_suspicious_keywords,
        'has_https': has_https,
        'cert_valid': cert_valid,
        'domain_age_days': domain_age_days,
        'subdomain_count': subdomain_count,
        'has_at_symbol': has_at_symbol,
        'has_double_slash_path': has_double_slash_path,
        'min_levenshtein_popular': min_levenshtein_popular
    }

if __name__ == '__main__':
    import sys
    # If run directly, run a test suite
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        print("Running feature extraction tests...")
        test_urls = [
            "https://www.google.com",
            "http://192.168.1.1/login.php",
            "https://verify-secure-paypal.update-account.com/webscr",
            "bit.ly/3xyz",
            "https://gooogle.com/update"
        ]
        
        for url in test_urls:
            print(f"\nURL: {url}")
            feats = extract_features(url, check_network=True)
            for k, v in feats.items():
                print(f"  {k}: {v}")
