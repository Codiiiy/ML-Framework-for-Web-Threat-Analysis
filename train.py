import os
import math
import logging
import pathlib
import re
import random
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
import xgboost as xgb
from urllib.parse import urlparse, unquote
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

try:
    from google.colab import drive, files
    COLAB_AVAILABLE = True
except ImportError:
    COLAB_AVAILABLE = False

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_logging(is_debug: bool):
    if is_debug:
        root_level = logging.DEBUG
    else:
        root_level = logging.WARNING
    logging.getLogger().setLevel(root_level)

@dataclass
class Config:
    model: str = "XGBoost"
    n_estimators: int = 100
    max_depth: int = 4
    learning_rate: float = 0.05
    n_splits: int = 5
    enable_feature_scaling: bool = True
    debug_mode: bool = False
    run_local: bool = True
    base_dir: str = "."
    dataset_dir: str = "dataset"
    mount_drive: bool = False
    brand_variations_count: int = 10

    def __post_init__(self):
        assert self.n_splits >= 2, "n_splits must be at least 2"
        
        if self.run_local:
            self.project_root = os.path.abspath(self.base_dir)
        else:
            if self.mount_drive and COLAB_AVAILABLE:
                try:
                    drive.mount('/content/drive')
                    logger.info("Google Drive mounted successfully")
                    self.project_root = pathlib.Path('/content/drive/MyDrive').resolve()
                except Exception as e:
                    logger.warning(f"Failed to mount Google Drive: {e}. Using local path instead.")
                    self.project_root = pathlib.Path('.').resolve()
            else:
                self.project_root = pathlib.Path('.').resolve()
        
        self.project_root = pathlib.Path(self.project_root)
        self.project_root.mkdir(parents=True, exist_ok=True)
        
        self.dataset_path = self.project_root / self.dataset_dir
        self.output_dir = str(self.project_root / "policies")
        
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.debug(f"Working directory: {self.project_root}")
        logger.debug(f"Dataset directory: {self.dataset_path}")
        logger.debug(f"Output directory: {self.output_dir}")

@staticmethod   
class BrandVariationGenerator:
    COMMON_BRANDS = [
        "paypal", "google", "facebook", "microsoft", "amazon", 
        "apple", "netflix", "instagram", "twitter", "linkedin",
        "ebay", "yahoo", "bank", "chase", "wellsfargo",
        "citibank", "bankofamerica", "americanexpress", "visa", "mastercard"
    ]
    
    CYRILLIC_LOOKALIKES = {
        'a': ['а', 'ӑ', 'ӓ'],
        'c': ['с', 'ҫ'],
        'e': ['е', 'ё', 'ӗ'],
        'h': ['һ'],
        'i': ['і', 'ӏ'],
        'o': ['о', 'ӧ'],
        'p': ['р'],
        's': ['ѕ'],
        'x': ['х'],
        'y': ['у', 'ӱ'],
        'b': ['Ь'],
        'k': ['к'],
        'm': ['м'],
        'n': ['п'],
        't': ['т'],
    }
    
    UNICODE_LOOKALIKES = {
        'a': ['ɑ', 'α', 'а', 'ａ', 'ạ', 'ả', 'ã', 'â', 'ă'],
        'b': ['Ь', 'b', 'ḃ', 'ｂ'],
        'c': ['ϲ', 'с', 'ⅽ', 'ｃ', 'ċ'],
        'd': ['ԁ', 'ḋ', 'ｄ'],
        'e': ['е', 'е', 'ė', 'ｅ', 'ẹ', 'ẻ', 'ẽ'],
        'g': ['ɡ', 'ɢ', 'ġ', 'ｇ'],
        'h': ['һ', 'ｈ', 'ḣ'],
        'i': ['і', 'ɪ', 'ı', 'ｉ', 'ị', 'ỉ', 'ĩ'],
        'j': ['ј', 'ｊ'],
        'l': ['ӏ', 'Ι', 'l', '|', 'ｌ'],
        'n': ['п', 'ո', 'ｎ', 'ṅ'],
        'o': ['о', 'ο', 'σ', 'օ', 'ｏ', 'ọ', 'ỏ', 'õ', 'ô', 'ơ'],
        'p': ['р', 'р', 'ｐ', 'ṗ'],
        'q': ['ԛ', 'ｑ'],
        's': ['ѕ', 'ꜱ', 'ｓ', 'ṡ'],
        't': ['т', 'ｔ', 'ṫ'],
        'u': ['υ', 'ս', 'ｕ', 'ụ', 'ủ', 'ũ', 'ư'],
        'v': ['ν', 'ѵ', 'ｖ'],
        'w': ['ԝ', 'ｗ', 'ẁ', 'ẃ', 'ẅ'],
        'x': ['х', 'ⅹ', 'ｘ'],
        'y': ['у', 'ү', 'ｙ', 'ỳ', 'ý', 'ỵ', 'ỷ', 'ỹ'],
        'z': ['ᴢ', 'ｚ', 'ż'],
    }
    
    def __init__(self, variations_per_brand: int = 10):
        self.variations_per_brand = variations_per_brand
        self.variations = self._generate_all_variations()
    
    def _generate_homoglyph_variations(self, brand: str) -> Set[str]:
        variations = set()
        
        for _ in range(self.variations_per_brand):
            variant = list(brand)
            num_changes = random.randint(1, min(3, len(brand)))
            positions = random.sample(range(len(brand)), num_changes)
            
            for pos in positions:
                char = brand[pos]
                if char in self.UNICODE_LOOKALIKES:
                    variant[pos] = random.choice(self.UNICODE_LOOKALIKES[char])
                elif char in self.CYRILLIC_LOOKALIKES:
                    variant[pos] = random.choice(self.CYRILLIC_LOOKALIKES[char])
            
            variations.add(''.join(variant))
        
        return variations
    
    def _generate_brand_variations(self, brand: str) -> Set[str]:
        variations = set()
        
        char_substitutions = {
            'o': ['0', 'oo', 'ou'],
            'i': ['1', 'l', '!'],
            'e': ['3', 'ee'],
            'a': ['@', '4', 'aa'],
            's': ['5', '$', 'ss'],
            'l': ['1', 'i', '|'],
            't': ['7', '+'],
            'g': ['9', 'q'],
            'b': ['8', '6'],
        }
        
        for _ in range(self.variations_per_brand):
            variant = list(brand)
            num_changes = random.randint(1, min(3, len(brand)))
            positions = random.sample(range(len(brand)), num_changes)
            
            for pos in positions:
                char = brand[pos]
                if char in char_substitutions:
                    variant[pos] = random.choice(char_substitutions[char])
            
            variations.add(''.join(variant))
        
        for i in range(len(brand)):
            if i < len(brand) - 1:
                swapped = list(brand)
                swapped[i], swapped[i+1] = swapped[i+1], swapped[i]
                variations.add(''.join(swapped))
        
        for i in range(len(brand)):
            doubled = brand[:i] + brand[i] + brand[i:]
            variations.add(doubled)
        
        for i in range(len(brand)):
            removed = brand[:i] + brand[i+1:]
            if removed:
                variations.add(removed)
        
        variations.update(self._generate_homoglyph_variations(brand))
        
        return variations
    
    def _generate_all_variations(self) -> Set[str]:
        all_variations = set()
        for brand in self.COMMON_BRANDS:
            variations = self._generate_brand_variations(brand)
            all_variations.update(variations)
        return all_variations
    
    def _normalize_unicode(self, text: str) -> str:
        normalized = []
        for char in text:
            found = False
            for latin_char, lookalikes in self.UNICODE_LOOKALIKES.items():
                if char in lookalikes or char == latin_char:
                    normalized.append(latin_char)
                    found = True
                    break
            if not found:
                for latin_char, lookalikes in self.CYRILLIC_LOOKALIKES.items():
                    if char in lookalikes:
                        normalized.append(latin_char)
                        found = True
                        break
            if not found:
                normalized.append(char)
        return ''.join(normalized)
    
    def has_misspelled_brand(self, url: str) -> bool:
        url_lower = url.lower()
        url_normalized = self._normalize_unicode(url_lower)
        
        for brand in self.COMMON_BRANDS:
            if brand in url_normalized and brand not in url_lower:
                return True
        
        return any(variation in url_lower for variation in self.variations)

class DatasetLoader:
    def __init__(self, config: Config):
        self.config = config
        self.benign_dir = config.dataset_path / "benign"
        self.malicious_dir = config.dataset_path / "malicious"
        self.skipped_ip_count = 0

    def is_ip_address(self, host: str) -> bool:
        """Check if a host string is an IP address"""
        if not host:
            return False
        host = host.split(':')[0]
        parts = host.split('.')
        if len(parts) == 4:
            try:
                return all(0 <= int(part) <= 255 for part in parts)
            except ValueError:
                return False
        return False

    def should_skip_url(self, url: str) -> bool:
        try:
            parsed = urlparse(url)
            host = (parsed.netloc or "").lower().split(':')[0]
            if self.is_ip_address(host):
                return True
            return False
        except:
            return False



    def extract_url_from_filename(self, filename: str, label: str) -> str:
        if label == "benign":
            url = re.sub(r'^genuine_', '', filename)
            url = re.sub(r'_\d+$', '', url)
        else:
            url = re.sub(r'^phishing_', '', filename)
            url = re.sub(r'_\d+$', '', url)
        
        url = url.replace('.html', '')
        
        if not url.startswith('http://') and not url.startswith('https://'):
            url = 'http://' + url
        
        url = unquote(url)
        
        return url

    def read_html_content(self, filepath: str) -> str:
        try:
            with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                return f.read()
        except Exception as e:
            logger.warning(f"Error reading {filepath}: {e}")
            return ""

    def load_data(self) -> Tuple[List[Dict], List[Dict]]:
        malicious_samples = []
        benign_samples = []

        if not self.malicious_dir.exists():
            logger.error(f"Malicious directory not found: {self.malicious_dir}")
            raise FileNotFoundError(f"Directory not found: {self.malicious_dir}")
        
        if not self.benign_dir.exists():
            logger.error(f"Benign directory not found: {self.benign_dir}")
            raise FileNotFoundError(f"Directory not found: {self.benign_dir}")

        logger.info(f"Loading malicious samples from {self.malicious_dir}")
        for filename in os.listdir(self.malicious_dir):
            filepath = self.malicious_dir / filename
            if os.path.isfile(filepath):
                url = self.extract_url_from_filename(filename, "malicious")
                
                if self.should_skip_url(url):
                    self.skipped_ip_count += 1
                    continue
                
                html_content = self.read_html_content(str(filepath))
                malicious_samples.append({
                    "url": url,
                    "html_content": html_content,
                    "label": "malicious"
                })

        logger.info(f"Loading benign samples from {self.benign_dir}")
        for filename in os.listdir(self.benign_dir):
            filepath = self.benign_dir / filename
            if os.path.isfile(filepath):
                url = self.extract_url_from_filename(filename, "benign")
                
                # Skip if URL is IP-based
                if self.should_skip_url(url):
                    self.skipped_ip_count += 1
                    continue
                
                html_content = self.read_html_content(str(filepath))
                benign_samples.append({
                    "url": url,
                    "html_content": html_content,
                    "label": "benign"
                })

        logger.info(f"Loaded {len(malicious_samples)} malicious and {len(benign_samples)} benign samples")
        logger.info(f"Skipped {self.skipped_ip_count} IP-based URLs")
        return malicious_samples, benign_samples

class FeatureExtractor:
    @staticmethod
    def url_entropy(s: str) -> float:
        if not s or len(s) == 0:
            return 0.0
        cnt = {}
        for ch in s:
            cnt[ch] = cnt.get(ch, 0) + 1
        probs = [v / len(s) for v in cnt.values()]
        return -sum(p * math.log2(p) for p in probs if p > 0)

    @staticmethod
    def count_special_chars(s: str) -> Dict[str, int]:
        return {
            "num_dots": s.count("."),
            "num_hyphens": s.count("-"),
            "num_underscores": s.count("_"),
            "num_slashes": s.count("/"),
            "num_digits": sum(c.isdigit() for c in s),
            "num_at_signs": s.count("@"),
            "num_question_marks": s.count("?"),
            "num_equals": s.count("="),
            "num_ampersands": s.count("&")
        }

    @staticmethod
    def is_ip_address(host: str) -> bool:
        if not host:
            return False
        host = host.split(':')[0]
        parts = host.split('.')
        if len(parts) != 4:
            return False
        try:
            return all(0 <= int(part) <= 255 for part in parts)
        except ValueError:
            return False

    @staticmethod
    def extract_lexical(url: str, brand_generator) -> Dict: 
        try:
            p = urlparse(url)
            host = (p.netloc or "").lower()
            path = (p.path or "")
            query = (p.query or "")
            
            if host.isdigit():
                return {
                    "url_len": len(url),
                    "host_len": 0,
                    "path_len": len(path),
                    "query_len": len(query),
                    "num_dots": 0,
                    "num_hyphens": 0,
                    "num_underscores": 0,
                    "num_slashes": url.count("/"),
                    "num_digits": sum(c.isdigit() for c in url),
                    "num_at_signs": 0,
                    "num_question_marks": url.count("?"),
                    "num_equals": url.count("="),
                    "num_ampersands": url.count("&"),
                    "has_ip_host": 0,
                    "entropy_host": 0,
                    "entropy_path": FeatureExtractor.url_entropy(path),
                    "entropy_url": FeatureExtractor.url_entropy(url),
                    "has_login_kw": int(any(kw in url.lower() for kw in ["login","signin","verify","account","secure"])),
                    "has_pay_kw": int(any(kw in url.lower() for kw in ["pay","bank","update","confirm"])),
                    "has_suspicious_tld": 0,
                    "protocol_https": int(p.scheme=="https"),
                    "subdomain_count": 0,
                    "digit_letter_ratio": (sum(c.isdigit() for c in url) / max(1, sum(c.isalpha() for c in url))),
                    "has_punycode": 0,
                    "has_misspelled_brand": int(brand_generator.has_misspelled_brand(url)),
                    "keyword_pressure": int(any(kw in url.lower() for kw in ["urgent","alert","warning","unlock"])),
                    "rare_tld": 0,
                    "path_depth": urlparse(url).path.count("/"),
                    "num_params": urlparse(url).query.count("&") + 1 if urlparse(url).query else 0
                }
            
            special_chars = FeatureExtractor.count_special_chars(url)
            return {
                "url_len": len(url),
                "host_len": len(host),
                "path_len": len(path),
                "query_len": len(query),
                **special_chars,
                "has_ip_host": int(FeatureExtractor.is_ip_address(host)),
                "entropy_host": FeatureExtractor.url_entropy(host),
                "entropy_path": FeatureExtractor.url_entropy(path),
                "entropy_url": FeatureExtractor.url_entropy(url),
                "has_login_kw": int(any(kw in url.lower() for kw in ["login", "signin", "verify", "account", "secure"])),
                "has_pay_kw": int(any(kw in url.lower() for kw in ["pay", "bank", "update", "confirm"])),
                "has_suspicious_tld": int(any(url.endswith(tld) for tld in [".tk", ".ml", ".ga", ".cf", ".gq"])),
                "protocol_https": int(p.scheme == "https"),
                "subdomain_count": host.count(".") - 1 if "." in host else 0,
                "digit_letter_ratio": (sum(c.isdigit() for c in url) / max(1, sum(c.isalpha() for c in url))),
                "has_punycode": int("xn--" in url),
                "has_misspelled_brand": int(brand_generator.has_misspelled_brand(url)),
                "keyword_pressure": int(any(kw in url.lower() for kw in ["urgent", "alert", "warning", "unlock"])),
                "rare_tld": int(any(url.endswith(t) for t in [".zip", ".kim", ".country", ".science", ".work"])),
                "path_depth": urlparse(url).path.count("/"),
                "num_params": urlparse(url).query.count("&") + 1 if urlparse(url).query else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting lexical features from {url}: {e}")
            return {}

    @staticmethod
    def extract_html_features(html_content: str) -> Dict:
        try:
            html_lower = html_content.lower()
            return {
                "html_len": len(html_content),
                "num_scripts": html_content.count("<script"),
                "num_iframes": html_content.count("<iframe"),
                "num_forms": html_content.count("<form"),
                "num_inputs": html_content.count("<input"),
                "num_links": html_content.count("<a "),
                "num_images": html_content.count("<img"),
                "has_password_field": int("type=\"password\"" in html_lower or "type='password'" in html_lower),
                "has_hidden_input": int("type=\"hidden\"" in html_lower or "type='hidden'" in html_lower),
                "num_external_links": html_content.count("http://") + html_content.count("https://"),
                "num_suspicious_keywords": sum(kw in html_lower for kw in ["verify", "suspend", "confirm", "urgent", "click here", "update", "secure"]),
                "html_entropy": FeatureExtractor.url_entropy(html_content[:10000]),
                "num_obfuscated_js": html_lower.count("eval(") + html_lower.count("atob(") + html_lower.count("unescape("),
                "has_onclick_hooks": int("onclick=" in html_lower),
                "has_onmouseover_hooks": int("onmouseover=" in html_lower),
                "external_script_loads": html_lower.count("src=\"http") + html_lower.count("src='http"),
                "form_action_external": int("action=\"http" in html_lower or "action='http" in html_lower),
                "spoofed_brand_terms": sum(t in html_lower for t in ["bank", "apple id", "office365", "paypal", "google account"]),
                "fake_security_indicators": sum(t in html_lower for t in ["secure login", "security check", "validate account"])
            }
        except Exception as e:
            logger.warning(f"Error extracting HTML features: {e}")
            return {}

    @staticmethod
    def extract_features(record: Dict) -> Dict:
        url = record.get("url", "")
        html_content = record.get("html_content", "")
        
        lexical_features = FeatureExtractor.extract_lexical(url)
        html_features = FeatureExtractor.extract_html_features(html_content)
        
        if not lexical_features:
            return {}
        
        features = {**lexical_features, **html_features}
        return features

class PhishingDetector:
    def __init__(self, config: Config):
        self.config = config
        self.loader = DatasetLoader(config)
        self.scaler = StandardScaler() if config.enable_feature_scaling else None
        self.label_encoder = LabelEncoder()
        configure_logging(config.debug_mode)
        
        self.brand_generator = BrandVariationGenerator(
            variations_per_brand=config.brand_variations_count
        )
    def process_samples(self, samples: List[Dict]) -> List[Dict]:
        rows = []
        failed_count = 0
        
        for i, sample in enumerate(samples, 1):
            try:
                features = FeatureExtractor.extract_features(sample)
                if features:
                    features["url"] = sample.get("url", "MISSING_URL")
                    features["label"] = sample.get("label")
                    rows.append(features)
                else:
                    failed_count += 1
                    logger.debug(f"Feature extraction failed for sample: {sample.get('url', 'UNKNOWN')}")
                
                if i % 500 == 0:
                    logger.info(f"Processed {i}/{len(samples)} samples ({len(rows)} successful)")
            except Exception as e:
                failed_count += 1
                logger.debug(f"Error processing sample: {e}")
        
        logger.info(f"Batch complete: {len(rows)}/{len(samples)} samples successfully processed")
        return rows

    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        malicious_samples, benign_samples = self.loader.load_data()
        
        logger.info("=" * 60)
        logger.info("PROCESSING SAMPLES")
        logger.info("=" * 60)
        
        all_samples = malicious_samples + benign_samples
        processed_rows = self.process_samples(all_samples)

        logger.info("=" * 60)
        logger.info(f"FINAL COLLECTION: {len(processed_rows)} total samples")
        logger.info("=" * 60)

        df = pd.DataFrame(processed_rows)
        logger.info(f"Initial dataset size: {len(df)} samples")

        df = df.drop_duplicates(subset=["url"])
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)
        logger.info(f"Cleaned dataset size: {len(df)} samples")
        
        label_counts = df['label'].value_counts()
        logger.info(f"Label distribution:\n{label_counts}")

        X = df.drop("label", axis=1)
        y = df["label"]
        if "url" in X.columns:
            X = X.drop("url", axis=1)
        for col in X.select_dtypes(include=['object']).columns:
            X[col] = LabelEncoder().fit_transform(X[col].astype(str))
        X = X.apply(pd.to_numeric, errors='coerce')
        X = X.fillna(0)
        X = X.clip(lower=-1e10, upper=1e10)
        y = pd.Series(self.label_encoder.fit_transform(y))

        label_mapping = dict(zip(self.label_encoder.classes_,
                                 self.label_encoder.transform(self.label_encoder.classes_)))
        logger.info(f"Label mapping: {label_mapping}")
        return X, y

    def train_with_cv(self, X: pd.DataFrame, y: pd.Series) -> xgb.XGBClassifier:
        if self.scaler:
            X_scaled = pd.DataFrame(
                self.scaler.fit_transform(X),
                columns=X.columns,
                index=X.index
            )
        else:
            X_scaled = X

        class_counts = np.bincount(y)
        total = len(y)
        scale_pos_weight = class_counts[0] / class_counts[1] if len(class_counts) > 1 else 1.0
        logger.info(f"Class imbalance ratio: {scale_pos_weight:.2f} (benign/malicious)")

        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=42
        )
        accuracies = []
        precisions = []
        recalls = []
        f1_scores = []
        fold = 1
        for train_idx, val_idx in skf.split(X_scaled, y):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                tree_method='hist',
                n_jobs=-1,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                eval_metric='logloss',
                random_state=42,
                scale_pos_weight=scale_pos_weight,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                gamma=0.1,
                reg_alpha=0.1,
                reg_lambda=1.0
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)
            acc = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred, output_dict=True, zero_division=0)
            
            accuracies.append(acc)
            precisions.append(report.get('1', {}).get('precision', 0))
            recalls.append(report.get('1', {}).get('recall', 0))
            f1_scores.append(report.get('1', {}).get('f1-score', 0))
            
            logger.info(f"Fold {fold} complete")
            fold += 1

        print("\nCross-Validation Results:")
        print(f"Accuracy  - Mean: {np.mean(accuracies):.4f} | Min: {np.min(accuracies):.4f} | Max: {np.max(accuracies):.4f}")
        print(f"Precision - Mean: {np.mean(precisions):.4f} | Min: {np.min(precisions):.4f} | Max: {np.max(precisions):.4f}")
        print(f"Recall    - Mean: {np.mean(recalls):.4f} | Min: {np.min(recalls):.4f} | Max: {np.max(recalls):.4f}")
        print(f"F1 Score  - Mean: {np.mean(f1_scores):.4f} | Min: {np.min(f1_scores):.4f} | Max: {np.min(f1_scores):.4f}")

        final_model = xgb.XGBClassifier(
            tree_method='hist',
            n_jobs=-1,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            eval_metric='logloss',
            random_state=42,
            scale_pos_weight=scale_pos_weight,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0.1,
            reg_alpha=0.1,
            reg_lambda=1.0
        )
        final_model.fit(X_scaled, y)

        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"\nTop 10 features:")
        print(feature_importance.head(10).to_string(index=False))
        
        importance_path = os.path.join(self.config.output_dir, "feature_importance.csv")
        feature_importance.to_csv(importance_path, index=False)
        
        return final_model

    def save_model(self, model: xgb.XGBClassifier):
        model_path = os.path.join(self.config.output_dir, "xgb_phishing_detector.json")
        model.save_model(model_path)

        if self.scaler:
            scaler_path = os.path.join(self.config.output_dir, "feature_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)

        le_path = os.path.join(self.config.output_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, le_path)
        
        print("\nTraining complete! Model saved successfully.")
        
        if not self.config.run_local and COLAB_AVAILABLE and not self.config.mount_drive:
            for path in [model_path, scaler_path, le_path]:
                if os.path.exists(path):
                    try:
                        files.download(path)
                    except Exception as e:
                        logger.warning(f"Could not download {path}: {e}")

    def run(self):
        try:
            X, y = self.prepare_data()
            model = self.train_with_cv(X, y)
            self.save_model(model)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise

if __name__ == "__main__":
    config = Config(
        n_estimators=100,
        max_depth=12,
        learning_rate=0.05,
        n_splits=10,
        enable_feature_scaling=True,
        debug_mode=True,
        run_local=True,
        base_dir=".",
        dataset_dir="dataset",
        brand_variations_count = 10
    )
    
    detector = PhishingDetector(config)
    print("Starting phishing detection pipeline...")
    detector.run()