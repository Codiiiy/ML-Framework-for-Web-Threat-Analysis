import os
import time
import math
import warnings
import logging
import argparse
from typing import Dict, List, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
import requests
import pandas as pd
import numpy as np
import xgboost as xgb
import wandb
from urllib.parse import urlparse
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TEMP_DIR = os.path.join(os.path.dirname(BASE_DIR), "Temp")
OUTPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "policies")

MALICIOUS_CACHE = os.path.join(TEMP_DIR, "malicious_urls.json.gz")
BENIGN_CACHE = os.path.join(TEMP_DIR, "benign_urls.json.gz")
CACHE_EXPIRY_HOURS = 24

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def configure_logging(is_debug: bool):
    if is_debug:
        root_level = logging.DEBUG
        urllib3_level = logging.INFO
    else:
        root_level = logging.INFO
        urllib3_level = logging.ERROR
    
    logger.info(f"Setting root log level to {logging.getLevelName(root_level)} (Debug: {is_debug})")
    
    logging.getLogger().setLevel(root_level)
    logging.getLogger("urllib3").setLevel(urllib3_level)

@dataclass
class Config:
    model: str = "XGBoost"
    n_estimators: int = 200
    max_depth: int = 7
    learning_rate: float = 0.05
    n_splits: int = 5
    num_malicious: int = 5000
    num_benign: int = 5000
    max_urls: Optional[int] = None
    batch_size: int = 500
    timeout: int = 8
    max_workers: int = 10
    retry_attempts: int = 3
    enable_feature_scaling: bool = True
    use_gpu: bool = False
    debug_mode: bool = False
    
    def __post_init__(self):
        assert self.num_malicious > 0, "num_malicious must be positive"
        assert self.num_benign > 0, "num_benign must be positive"
        assert self.max_workers > 0, "max_workers must be positive"
        assert self.n_splits >= 2, "n_splits must be at least 2"
        if self.max_urls is not None:
            assert self.max_urls > 0, "max_urls must be positive"

class URLDataCollector:
    OPENPHISH_FEED = "https://openphish.com/feed.txt"
    TRANCO_CSV = "https://tranco-list.eu/download/NNZPW/1000000"
    
    def __init__(self, timeout: int = 8, retry_attempts: int = 3):
        self.timeout = timeout
        self.session = self._create_session(retry_attempts)
        
    def _create_session(self, retry_attempts: int) -> requests.Session:
        session = requests.Session()
        retry_strategy = Retry(
            total=retry_attempts,
            backoff_factor=1,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"]
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
    
    def load_cache(self, path: str) -> Optional[List[str]]:
        if os.path.exists(path):
            cache_time = os.path.getmtime(path)
            if (time.time() - cache_time) < CACHE_EXPIRY_HOURS * 3600:
                logger.info(f"Loading URLs from up-to-date cache: {path}")
                try:
                    df = pd.read_json(path, compression='gzip', lines=True)
                    return df[df.columns[0]].tolist() 
                except Exception as e:
                    logger.warning(f"Error loading cache {path}: {e}")
                    return None
            else:
                logger.info(f"Cache expired for {path} after {CACHE_EXPIRY_HOURS} hours. Downloading new feed.")
        return None

    def save_cache(self, path: str, urls: List[str]):
        os.makedirs(TEMP_DIR, exist_ok=True)
        df = pd.DataFrame(urls)
        df.to_json(path, compression='gzip', orient='records', lines=True)
        logger.info(f"Saved new feed to cache: {path}")

    def download_openphish(self, n: Optional[int] = None) -> List[str]:
        try:
            logger.info("Downloading OpenPhish feed...")
            r = self.session.get(self.OPENPHISH_FEED, timeout=20)
            r.raise_for_status()
            lines = [l.strip() for l in r.text.splitlines() if l.strip()]
            urls = lines if n is None else lines[:n]
            logger.info(f"Downloaded {len(urls)} malicious URLs")
            return urls
        except requests.RequestException as e:
            logger.error(f"Failed to download OpenPhish feed: {e}")
            raise
    
    def download_tranco(self, n: Optional[int] = None) -> List[str]:
        try:
            logger.info("Downloading Tranco list...")
            r = self.session.get(self.TRANCO_CSV, timeout=20)
            r.raise_for_status()
            lines = [l.strip().split(",")[-1] for l in r.text.splitlines() if l.strip()]
            urls = ["https://" + d for d in lines]
            urls = urls if n is None else urls[:n]
            logger.info(f"Downloaded {len(urls)} benign URLs")
            return urls
        except requests.RequestException as e:
            logger.error(f"Failed to download Tranco list: {e}")
            raise
    
    def safe_fetch(self, url: str) -> Dict:
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8"
            }
            r = self.session.get(
                url, 
                timeout=self.timeout, 
                allow_redirects=True, 
                headers=headers,
                verify=True
            )
            return {
                "url": url,
                "final_url": r.url,
                "status_code": int(r.status_code),
                "content_len": len(r.content),
                "text_len": len(r.text) if r.text else 0,
                "headers": {k.lower(): v for k, v in r.headers.items()},
                "redirect_count": len(r.history),
                "error": None
            }
        except requests.Timeout:
            return {"url": url, "error": "timeout"}
        except requests.TooManyRedirects:
            return {"url": url, "error": "too_many_redirects"}
        except requests.exceptions.SSLError:
            return {"url": url, "error": "ssl_error"}
        except requests.exceptions.ConnectionError:
            return {"url": url, "error": "connection_error"}
        except Exception as e:
            return {"url": url, "error": f"unknown: {type(e).__name__}"}

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
    def extract_lexical(url: str) -> Dict:
        try:
            p = urlparse(url)
            host = (p.netloc or "").lower()
            path = (p.path or "")
            query = (p.query or "")
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
                "has_port": int(bool(p.port)),
                "subdomain_count": host.count(".") - 1 if "." in host else 0
            }
        except Exception as e:
            logger.warning(f"Error extracting lexical features from {url}: {e}")
            return {}
    
    @staticmethod
    def extract_features(record: Dict) -> Dict:
        url = record.get("url", "")
        error = record.get("error")
        features = FeatureExtractor.extract_lexical(url)
        if error:
            features.update({
                "status_code": -1,
                "content_len": 0,
                "text_len": 0,
                "num_headers": 0,
                "redirect": 0,
                "redirect_count": 0,
                "server_present": 0,
                "html_content_type": 0,
                "has_error": 1,
                "error_timeout": int(error == "timeout"),
                "error_ssl": int(error == "ssl_error"),
                "error_connection": int(error == "connection_error")
            })
        else:
            headers = record.get("headers", {})
            content_type = headers.get("content-type", "")
            features.update({
                "status_code": record.get("status_code", 0),
                "content_len": record.get("content_len", 0),
                "text_len": record.get("text_len", 0),
                "num_headers": len(headers),
                "redirect": int(record.get("final_url") != url),
                "redirect_count": record.get("redirect_count", 0),
                "server_present": int(bool(headers.get("server"))),
                "html_content_type": int("text/html" in content_type.lower()),
                "has_error": 0,
                "error_timeout": 0,
                "error_ssl": 0,
                "error_connection": 0,
                "has_x_frame_options": int("x-frame-options" in headers),
                "has_csp": int("content-security-policy" in headers),
                "has_strict_transport": int("strict-transport-security" in headers)
            })
        return features

class PhishingDetector:
    def __init__(self, config: Config):
        self.config = config
        self.collector = URLDataCollector(
            timeout=config.timeout,
            retry_attempts=config.retry_attempts
        )
        self.scaler = StandardScaler() if config.enable_feature_scaling else None
        self.label_encoder = LabelEncoder()
        wandb.init(
            project="poisonweb",
            name=f"phishing_detector_{int(time.time())}",
            config=asdict(config)
        )
        configure_logging(config.debug_mode)
    
    def process_urls_parallel(self, urls: List[str], label: str) -> List[Dict]:
        rows = []
        total = len(urls)
        with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
            future_to_url = {
                executor.submit(self.collector.safe_fetch, url): url 
                for url in urls
            }
            for i, future in enumerate(as_completed(future_to_url), 1):
                try:
                    record = future.result()
                    features = FeatureExtractor.extract_features(record)
                    
                    if features:
                        features["url"] = record.get("url", "MISSING_URL") 
                        features["label"] = label
                        rows.append(features)
                    else:
                        logger.warning(f"Feature extraction failed completely for URL: {record.get('url', 'UNKNOWN')}")

                    if i % 100 == 0:
                        logger.info(f"Processed {i}/{total} {label} URLs")
                except Exception as e:
                    logger.error(f"Error processing URL: {e}")
        return rows
    
    def prepare_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        num_mal = self.config.num_malicious
        num_ben = self.config.num_benign
        if self.config.max_urls is not None:
            max_urls_per_class = self.config.max_urls // 2
            num_mal = max_urls_per_class
            num_ben = max_urls_per_class
            logger.info(f"Using --max-urls={self.config.max_urls}. Setting mal/ben count to {max_urls_per_class} each.")
        
        malicious_urls = self.collector.load_cache(MALICIOUS_CACHE)
        if not malicious_urls:
            malicious_urls = self.collector.download_openphish(num_mal)
            self.collector.save_cache(MALICIOUS_CACHE, malicious_urls)
        
        benign_urls = self.collector.load_cache(BENIGN_CACHE)
        if not benign_urls:
            benign_urls = self.collector.download_tranco(num_ben)
            self.collector.save_cache(BENIGN_CACHE, benign_urls)
        
        malicious_urls = malicious_urls[:num_mal]
        benign_urls = benign_urls[:num_ben]
        
        logger.info("Processing malicious URLs...")
        mal_samples = self.process_urls_parallel(malicious_urls, "malicious")
        logger.info("Processing benign URLs...")
        ben_samples = self.process_urls_parallel(benign_urls, "benign")
        
        df = pd.DataFrame(mal_samples + ben_samples)
        
        logger.info(f"Initial dataset size: {len(df)} samples")
        
        df = df.drop_duplicates(subset=["url"])
        
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna(axis=0)
        df = df.reset_index(drop=True)
        logger.info(f"Cleaned dataset size: {len(df)} samples")
        logger.info(f"Label distribution:\n{df['label'].value_counts()}")
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
        wandb.log({"label_mapping": label_mapping})
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
            
        params = {}
        if self.config.use_gpu:
            params['device'] = 'cuda'
            logger.info("Using device='cuda' for GPU acceleration.")
        else:
            params['tree_method'] = 'hist'
            logger.info("Using tree_method='hist' (CPU).")
        
        skf = StratifiedKFold(
            n_splits=self.config.n_splits,
            shuffle=True,
            random_state=42
        )
        metrics = []
        fold = 1
        for train_idx, val_idx in skf.split(X_scaled, y):
            X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            model = xgb.XGBClassifier(
                **params,
                n_jobs=-1,
                n_estimators=self.config.n_estimators,
                max_depth=self.config.max_depth,
                learning_rate=self.config.learning_rate,
                eval_metric='logloss',
                random_state=42
            )
            model.fit(X_train, y_train)

            y_pred = model.predict(X_val)

            acc = accuracy_score(y_val, y_pred)
            report = classification_report(y_val, y_pred, output_dict=True)
            cm = confusion_matrix(y_val, y_pred)
            wandb.log({
                f"fold_{fold}/accuracy": acc,
                f"fold_{fold}/precision": report.get('1', {}).get('precision', 0),
                f"fold_{fold}/recall": report.get('1', {}).get('recall', 0),
                f"fold_{fold}/f1": report.get('1', {}).get('f1-score', 0),
                f"fold_{fold}/confusion_matrix": wandb.plot.confusion_matrix(
                    probs=None,
                    y_true=y_val.values,
                    preds=y_pred,
                    class_names=self.label_encoder.classes_.tolist()
                )
            })
            metrics.append(acc)
            logger.info(f"Fold {fold} - Accuracy: {acc:.4f}, "
                        f"Precision: {report.get('1', {}).get('precision', 0):.4f}, "
                        f"Recall: {report.get('1', {}).get('recall', 0):.4f}")
            fold += 1
        wandb.log({
            "cv_mean_accuracy": float(np.mean(metrics)),
            "cv_std_accuracy": float(np.std(metrics)),
            "cv_min_accuracy": float(np.min(metrics)),
            "cv_max_accuracy": float(np.max(metrics))
        })
        logger.info(f"\nCross-validation results:")
        logger.info(f"Mean Accuracy: {np.mean(metrics):.4f} (+/- {np.std(metrics):.4f})")
        logger.info("Training final model on full dataset...")
        
        final_model = xgb.XGBClassifier(
            **params,
            n_jobs=-1,
            n_estimators=self.config.n_estimators,
            max_depth=self.config.max_depth,
            learning_rate=self.config.learning_rate,
            eval_metric='logloss',
            random_state=42
        )
        final_model.fit(X_scaled, y)
            
        feature_importance = pd.DataFrame({
            'feature': X.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        logger.info(f"\nTop 10 most important features:")
        logger.info(feature_importance.head(10).to_string())
        wandb.log({
            "feature_importance": wandb.Table(dataframe=feature_importance)
        })
        return final_model
    
    def save_model(self, model: xgb.XGBClassifier):
        output_dir = OUTPUT_DIR 
        os.makedirs(output_dir, exist_ok=True)
        model_path = os.path.join(output_dir, "xgb_phishing_detector.json")
        model.save_model(model_path)
        logger.info(f"Model saved to {model_path}")
        if self.scaler:
            scaler_path = os.path.join(output_dir, "feature_scaler.pkl")
            joblib.dump(self.scaler, scaler_path)
            logger.info(f"Scaler saved to {scaler_path}")
        le_path = os.path.join(output_dir, "label_encoder.pkl")
        joblib.dump(self.label_encoder, le_path)
        logger.info(f"Label encoder saved to {le_path}")
        artifact = wandb.Artifact("phishing_detector", type="model")
        artifact.add_file(model_path)
        if self.scaler:
            artifact.add_file(scaler_path)
        artifact.add_file(le_path)
        wandb.log_artifact(artifact)
        logger.info("✅ Training complete and artifacts saved")
    
    def run(self):
        try:
            X, y = self.prepare_data()
            model = self.train_with_cv(X, y)
            self.save_model(model)
        except Exception as e:
            logger.error(f"Pipeline failed: {e}", exc_info=True)
            raise
        finally:
            wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train phishing detection model with XGBoost",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--num-malicious", type=int, default=5000,
                        help="Number of malicious URLs to collect")
    parser.add_argument("--num-benign", type=int, default=5000,
                        help="Number of benign URLs to collect")
    parser.add_argument("--max-urls", type=int, default=None,
                        help="Maximum total number of URLs (malicious + benign) to collect. Overrides --num-malicious and --num-benign if set.")
    parser.add_argument("--n-estimators", type=int, default=200,
                        help="Number of boosting rounds")
    parser.add_argument("--max-depth", type=int, default=7,
                        help="Maximum tree depth")
    parser.add_argument("--learning-rate", type=float, default=0.05,
                        help="Learning rate (eta)")
    parser.add_argument("--n-splits", type=int, default=5,
                        help="Number of cross-validation folds")
    parser.add_argument("--gpu", action="store_true",
                        help="Use GPU acceleration (gpu_hist) instead of CPU (hist)")
    parser.add_argument("--max-workers", type=int, default=10,
                        help="Number of parallel workers for URL fetching")
    parser.add_argument("--timeout", type=int, default=8,
                        help="HTTP request timeout in seconds")
    parser.add_argument("--no-scaling", action="store_true",
                        help="Disable feature scaling")
    parser.add_argument("--debug", action="store_true",
                        help="Enable DEBUG logging level for detailed output.")
    args = parser.parse_args()
    config = Config(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        learning_rate=args.learning_rate,
        n_splits=args.n_splits,
        num_malicious=args.num_malicious,
        num_benign=args.num_benign,
        max_urls=args.max_urls,
        max_workers=args.max_workers,
        timeout=args.timeout,
        enable_feature_scaling=not args.no_scaling,
        use_gpu=args.gpu,
        debug_mode=args.debug
    )
    logger.info(f"Configuration: {asdict(config)}")
    detector = PhishingDetector(config)
    detector.run()