import json
import os
import logging
import time
from functools import lru_cache
import joblib

logger = logging.getLogger(__name__)

class DatasetLoadError(Exception):
    """Raised when critical datasets fail to load or validate."""
    pass

class DatasetLoader:
    """
    Handles loading, caching, and validation of external chatbot datasets.
    Implements a hot-reload mechanism.
    """
    def __init__(self, dataset_dir: str = 'dataset'):
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_dir))
        
        self.intents = {}
        self.programme_profiles = {}
        self.language_config = {
            "romanised_tamil": [],
            "romanised_sinhala": [],
            "spelling_corrections": {},
            "out_of_domain_keywords": []
        }
        self.fallback_responses = {
            "en": [], "ta": [], "si": []
        }
        self.models = {
            "en": None, "ta": None, "si": None
        }
        
        self.reload()

    def reload(self):
        """Forces a full reload and validation of all datasets."""
        logger.info(f"Reloading datasets from {self.dataset_dir}...")
        
        # We clear the LRU cache on the internal load method
        self._load_json_file.cache_clear()
        
        try:
            self._load_and_validate_intents()
            self._load_and_validate_programmes()
            self._load_language_config()
            self._load_fallback_responses()
            self._load_ml_models()
            logger.info("All datasets loaded and validated successfully.")
        except Exception as e:
            logger.error(f"Failed to reload datasets: {e}")
            # In a production system we might want to raise, but for resilience we 
            # keep whatever state we have currently (which might be empty on first load).
            # We raise only if it's completely empty on initialization.
            if not self.intents:
                raise DatasetLoadError(f"Critical data missing. {e}")

    @lru_cache(maxsize=10)
    def _load_json_file(self, filename: str, mtime: float) -> dict:
        """
        Loads a JSON file. Cached by filename and its modification time.
        """
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"Dataset file missing: {filepath}")
            return {}
            
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                return json.load(f)
        except json.JSONDecodeError as e:
            logger.error(f"Malformed JSON in {filename}: {e}")
            return {}
        except Exception as e:
            logger.error(f"Error loading {filename}: {e}")
            return {}

    def _get_file_data(self, filename: str) -> dict:
        filepath = os.path.join(self.dataset_dir, filename)
        mtime = os.path.getmtime(filepath) if os.path.exists(filepath) else 0.0
        return self._load_json_file(filename, mtime)

    def _load_and_validate_intents(self):
        data = self._get_file_data('intents.json')
        if not data:
            logger.error("intents.json is empty or missing.")
            return

        valid_intents = {}
        for key, intent in data.items():
            # Validate structure
            if not isinstance(intent, dict) or 'patterns' not in intent or 'answer' not in intent:
                logger.warning(f"Intent '{key}' is missing 'patterns' or 'answer' keys. Skipping.")
                continue
                
            # Ensure patterns/answers have the required language keys
            # Even if empty, we ensure they exist to avoid KeyError later
            patterns = intent.get('patterns', {})
            answers = intent.get('answer', {})
            
            # Deduplicate patterns just in case
            for lang in ('en', 'ta', 'si'):
                patterns.setdefault(lang, [])
                if isinstance(patterns[lang], list):
                    patterns[lang] = list(dict.fromkeys(patterns[lang])) # remove dupes
                
                # Check for empty answers
                if lang not in answers or not answers[lang]:
                    logger.warning(f"Intent '{key}' is missing answer for language '{lang}'.")
                    
            valid_intents[key] = {
                'patterns': patterns,
                'answer': answers
            }
            
        self.intents = valid_intents
        logger.info(f"Loaded {len(self.intents)} valid intents.")

    def _load_and_validate_programmes(self):
        data = self._get_file_data('programme_profiles.json')
        valid_programmes = {}
        
        for key, profile in data.items():
            if not isinstance(profile, dict) or 'keywords' not in profile or 'answer' not in profile:
                logger.warning(f"Programme '{key}' is missing 'keywords' or 'answer'. Skipping.")
                continue
                
            keywords = profile.get('keywords', {})
            for lang in ('en', 'ta', 'si'):
                keywords.setdefault(lang, [])
                if isinstance(keywords[lang], list):
                    keywords[lang] = list(dict.fromkeys(keywords[lang]))
                    
            valid_programmes[key] = {
                'keywords': keywords,
                'answer': profile.get('answer', {})
            }
            
        self.programme_profiles = valid_programmes
        logger.info(f"Loaded {len(self.programme_profiles)} programme profiles.")

    def _load_language_config(self):
        data = self._get_file_data('language_config.json')
        if not data:
            return
            
        self.language_config = {
            "romanised_tamil": frozenset(data.get('romanised_tamil', [])),
            "romanised_sinhala": frozenset(data.get('romanised_sinhala', [])),
            "spelling_corrections": data.get('spelling_corrections', {}),
            "out_of_domain_keywords": frozenset(data.get('out_of_domain_keywords', []))
        }

    def _load_fallback_responses(self):
        data = self._get_file_data('fallback_responses.json')
        if not data:
            return
            
        for lang in ('en', 'ta', 'si'):
            if lang in data and isinstance(data[lang], list):
                self.fallback_responses[lang] = data[lang]

    def _load_ml_models(self):
        for lang in ('en', 'ta', 'si'):
            model_path = os.path.join(self.dataset_dir, f'model_{lang}.pkl')
            if os.path.exists(model_path):
                try:
                    self.models[lang] = joblib.load(model_path)
                    logger.info(f"Loaded ML model for '{lang}'.")
                except Exception as e:
                    logger.error(f"Failed to load ML model for '{lang}': {e}")
                    self.models[lang] = None
            else:
                self.models[lang] = None

    def predict_intent(self, text: str, language: str) -> tuple[str | None, float]:
        """
        Uses the trained ML model to predict the intent of the given text.
        Returns a tuple of (predicted_tag, confidence_score).
        """
        model = self.models.get(language)
        if not model:
            return None, 0.0
            
        try:
            probabilities = model.predict_proba([text])[0]
            max_prob_index = probabilities.argmax()
            confidence = probabilities[max_prob_index]
            predicted_tag = model.classes_[max_prob_index]
            return predicted_tag, confidence
        except Exception as e:
            logger.error(f"ML prediction error: {e}")
            return None, 0.0

# Singleton instance
_dataset_loader = None

def get_dataset_loader() -> DatasetLoader:
    global _dataset_loader
    if _dataset_loader is None:
        _dataset_loader = DatasetLoader()
    return _dataset_loader
