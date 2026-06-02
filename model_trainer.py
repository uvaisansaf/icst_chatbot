import os
import json
import logging
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
import joblib

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, dataset_dir: str = 'dataset'):
        self.dataset_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), dataset_dir))
        
    def _load_json(self, filename: str):
        filepath = os.path.join(self.dataset_dir, filename)
        if not os.path.exists(filepath):
            logger.warning(f"File not found: {filepath}")
            return {}
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)

    def train_and_save(self):
        logger.info("Starting ML model training...")
        
        intents_data = self._load_json('intents.json')
        programmes_data = self._load_json('programme_profiles.json')
        
        if not intents_data:
            logger.error("No training data found.")
            return

        # Prepare training data per language
        training_data = {
            'en': {'X': [], 'y': []},
            'ta': {'X': [], 'y': []},
            'si': {'X': [], 'y': []}
        }
        
        # 1. Add intents
        for tag, intent in intents_data.items():
            patterns = intent.get('patterns', {})
            for lang in ('en', 'ta', 'si'):
                for pattern in patterns.get(lang, []):
                    training_data[lang]['X'].append(pattern)
                    training_data[lang]['y'].append(tag)
                    
        # 2. Add programme profiles
        for tag, profile in programmes_data.items():
            full_tag = f"__prog__{tag}"
            keywords = profile.get('keywords', {})
            for lang in ('en', 'ta', 'si'):
                for pattern in keywords.get(lang, []):
                    training_data[lang]['X'].append(pattern)
                    training_data[lang]['y'].append(full_tag)

        # Train a model for each language
        for lang in ('en', 'ta', 'si'):
            X = training_data[lang]['X']
            y = training_data[lang]['y']
            
            if len(X) == 0:
                logger.warning(f"No training data for language '{lang}'. Skipping.")
                continue
                
            logger.info(f"Training {lang.upper()} model with {len(X)} samples across {len(set(y))} intents.")
            
            # Create a pipeline: TF-IDF (character n-grams) -> LogisticRegression
            # char_wb (3-5) is extremely robust to typos and morphology.
            pipeline = make_pipeline(
                TfidfVectorizer(
                    analyzer='char_wb',
                    ngram_range=(3, 5),
                    min_df=1,
                    sublinear_tf=True
                ),
                LogisticRegression(
                    C=10.0, 
                    class_weight='balanced',
                    solver='lbfgs',
                    max_iter=1000
                )
            )
            
            pipeline.fit(X, y)
            
            # Evaluate on training data (just for sanity check)
            score = pipeline.score(X, y)
            logger.info(f"{lang.upper()} Model Training Accuracy: {score:.2%}")
            
            # Save the model
            model_path = os.path.join(self.dataset_dir, f'model_{lang}.pkl')
            joblib.dump(pipeline, model_path)
            logger.info(f"Saved {lang.upper()} model to {model_path}")
            
        logger.info("All ML models trained and saved successfully.")

if __name__ == "__main__":
    trainer = ModelTrainer()
    trainer.train_and_save()
