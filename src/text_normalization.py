import re
import sys

# Configure encoding for Windows console output
sys.stdout.reconfigure(encoding='utf-8')

# Brands to preserve exactly (do not stem them)
BRANDS_PRESERVE = {
    "nike", "adidas", "puma", "mavi", "defacto", "koton", "polo", "colins", "hummel", "skechers",
    "reebok", "lotto", "slazenger", "lufian", "dockers", "avva", "tudors", "altınyıldız", "kiğılı",
    "damat", "beymen", "network", "derimod", "tergan", "bambi", "elle", "hotiç", "kemal tanca",
    "greyder", "lumberjack", "kinetix", "us polo", "u.s. polo", "crocs", "converse", "vans",
    "new balance", "under armour", "columbia", "the north face", "jack jones", "lc waikiki",
    "lcw", "zara", "mango", "bershka", "stradivarius"
}

# Phonetic brand corrections or direct maps
BRAND_CORRECTIONS = {
    "nayk": "nike",
    "addidas": "adidas",
    "adida": "adidas",
    "lcw": "lc waikiki",
    "lcwaikiki": "lc waikiki",
    "lcv": "lc waikiki",
    "lcvaykiki": "lc waikiki",
    "defakto": "defacto",
    "humel": "hummel",
    "skeçers": "skechers",
    "skeçer": "skechers",
}

# Turkish shopping abbreviations to normalize
ABBREVIATIONS = {
    r"\bkg\b": "kilogram",
    r"\bgr\b": "gram",
    r"\bcm\b": "santimetre",
    r"\bno\b": "numara",
    r"\byş\b": "yaş",
    r"\bkrş\b": "kuruş",
    r"\btl\b": "lira",
}

def turkish_lower(text):
    """Converts a string to lowercase using Turkish character mapping rules."""
    if not isinstance(text, str):
        return ""
    # Map Turkish capital I's correctly
    text = text.replace('İ', 'i').replace('I', 'ı')
    return text.lower()

def clean_punctuation(text):
    """Replaces punctuation with spaces and cleans up white space."""
    # Keep alphanumeric characters and replace punctuation with spaces
    text = re.sub(r'[^\w\s\d]', ' ', text)
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def levenshtein_distance(s1, s2):
    """Computes the Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
        
    return previous_row[-1]

def correct_typos_and_brands(token):
    """Corrects spelling mistakes in common brands using exact maps or Levenshtein distance 1."""
    if token in BRAND_CORRECTIONS:
        return BRAND_CORRECTIONS[token]
        
    for brand in BRANDS_PRESERVE:
        # Check Levenshtein distance 1 for brand corrections
        if len(brand) > 3 and levenshtein_distance(token, brand) == 1:
            return brand
            
    return token

def stem_turkish_word(word):
    """Applies a conservative Turkish stemmer to plurals and basic case endings."""
    if word in BRANDS_PRESERVE or len(word) <= 3:
        return word
        
    # 1. Strip basic case suffixes first
    cases = ['dan', 'den', 'tan', 'ten', 'da', 'de', 'ta', 'te', 'ya', 'ye']
    for suffix in cases:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
            
    # 2. Strip plurals
    for suffix in ['ler', 'lar']:
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[:-len(suffix)]
            break
            
    return word

def normalize_text(text, apply_stemming=True):
    """Main normalization function.
    Performs Turkish lowercasing, punctuation cleaning, abbreviations expansion,
    typo/brand correction, and optional stemming.
    """
    if not text:
        return ""
        
    # Lowercase
    text = turkish_lower(text)
    
    # Expand abbreviations
    for pattern, replacement in ABBREVIATIONS.items():
        text = re.sub(pattern, replacement, text)
        
    # Clean punctuation
    text = clean_punctuation(text)
    
    # Process tokens
    tokens = text.split()
    processed_tokens = []
    
    for token in tokens:
        # Correct brand names/typos
        token = correct_typos_and_brands(token)
        
        # Apply conservative stemming
        if apply_stemming:
            token = stem_turkish_word(token)
            
        processed_tokens.append(token)
        
    return " ".join(processed_tokens)

if __name__ == "__main__":
    # Test normalization on some examples
    examples = [
        "İSTANBUL'da LCW ve Defakto kız çocuk montu!",
        "nayk ayakkabılar ve adidas spor ayakkabısı",
        "bebek bezleri 5 kg ve 100 cm",
        "erkek ceketler koton dan"
    ]
    for ex in examples:
        print("Original:", ex)
        print("Normalized:", normalize_text(ex))
        print("-" * 40)
