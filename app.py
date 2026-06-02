## ═══════════════════════════════════════════════════════════════════════════
## ICST University Chatbot
## ═══════════════════════════════════════════════════════════════════════════
from __future__ import annotations
from dataset_loader import get_dataset_loader


import csv
import hashlib as _hashlib
import io
import json
import logging
import os
import random
import re
import threading
import time
import unicodedata
import uuid
from gtts import gTTS
import base64
from collections import defaultdict
from datetime import datetime, timezone, timedelta
from difflib import get_close_matches, SequenceMatcher
from functools import lru_cache
from typing import Optional, Tuple

from dotenv import load_dotenv
from flask import (Flask, render_template, request, jsonify, session,
                   redirect, url_for, flash, Response as FlaskResponse)
from flask_login import (LoginManager, UserMixin, login_user,
                         login_required, logout_user, current_user)
from flask_sqlalchemy import SQLAlchemy
from flask_wtf import FlaskForm
from flask_wtf.csrf import CSRFProtect
from werkzeug.security import generate_password_hash, check_password_hash
from wtforms import StringField, TextAreaField, BooleanField
from wtforms.validators import DataRequired
from sqlalchemy import func, String

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger('icst_chatbot')

# ── NLTK / spaCy (NLP utilities — English analysis only) ─────────────────────
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer

_NLTK_PACKAGES = [
    ('tokenizers/punkt',             'punkt'),
    ('tokenizers/punkt_tab',         'punkt_tab'),
    ('corpora/wordnet',              'wordnet'),
    ('corpora/omw-1.4',             'omw-1.4'),
    ('sentiment/vader_lexicon',      'vader_lexicon'),
    ('taggers/averaged_perceptron_tagger_eng', 'averaged_perceptron_tagger_eng'),
]

def _ensure_nltk():
    for path, pkg in _NLTK_PACKAGES:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception as e:
                logger.warning(f"NLTK ({pkg}): {e}")

_ensure_nltk()

try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None
    logger.info("spaCy model not found — NER disabled.")

# ── OPTIONAL: RapidFuzz for fast fuzzy matching ───────────────────────────────
try:
    from rapidfuzz import fuzz, process as rfprocess
    RAPIDFUZZ = True
except ImportError:
    RAPIDFUZZ = False
    logger.info("RapidFuzz not installed — using difflib (pip install rapidfuzz for ~5× speedup)")

load_dotenv()


# ═════════════════════════════════════════════════════════════════════════════
# FLASK APP CONFIG
# ═════════════════════════════════════════════════════════════════════════════

app = Flask(__name__, instance_relative_config=True)
os.makedirs(app.instance_path, exist_ok=True)

_secret = os.getenv('SECRET_KEY')
if not _secret:
    import secrets as _sec
    _secret = _sec.token_hex(32)
    logger.warning('SECRET_KEY not set — ephemeral key (sessions reset on restart)')

app.config.update(
    SECRET_KEY                     = _secret,
    SQLALCHEMY_DATABASE_URI        = os.getenv('DATABASE_URL', 'sqlite:///icst_chatbot.db'),
    SQLALCHEMY_TRACK_MODIFICATIONS = False,
    SQLALCHEMY_ENGINE_OPTIONS      = {'pool_pre_ping': True, 'pool_recycle': 3600},
    WTF_CSRF_TIME_LIMIT            = 3600,
    SESSION_COOKIE_HTTPONLY        = True,
    SESSION_COOKIE_SAMESITE        = 'Lax',
    SESSION_COOKIE_SECURE          = os.getenv('SESSION_COOKIE_SECURE', 'false').lower() in ('true', '1'),
    MAX_CONTENT_LENGTH             = 16 * 1024 * 1024,
)

db            = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view             = 'admin_login'
login_manager.login_message          = 'Please log in to access the admin panel.'
login_manager.login_message_category = 'warning'
csrf = CSRFProtect(app)


# ── Maintenance mode ──────────────────────────────────────────────────────────
_MAINTENANCE_EXEMPT = {'/health', '/admin/login', '/admin/logout',
                       '/static', '/favicon.ico'}

@app.before_request
def check_maintenance_mode():
    if not os.getenv('MAINTENANCE_MODE', '').lower() in ('true', '1'):
        return None
    path = request.path
    if any(path.startswith(p) for p in _MAINTENANCE_EXEMPT):
        return None
    if current_user and current_user.is_authenticated:
        return None
    return render_template('maintenance.html'), 503


@app.after_request
def set_security_headers(response):
    response.headers.setdefault('X-Content-Type-Options', 'nosniff')
    response.headers.setdefault('X-Frame-Options', 'SAMEORIGIN')
    response.headers.setdefault('Referrer-Policy', 'strict-origin-when-cross-origin')
    response.headers.setdefault('X-XSS-Protection', '1; mode=block')
    response.headers.setdefault('Permissions-Policy', 'geolocation=(), microphone=(self), camera=()')
    return response


# ── Rate limiter (in-memory) ──────────────────────────────────────────────────
_rate_store: dict = defaultdict(list)
_rate_lock        = threading.Lock()

def _check_rate_limit(uid: str, max_req: int = 40, window: int = 60) -> bool:
    now = time.time()
    with _rate_lock:
        _rate_store[uid] = [t for t in _rate_store[uid] if now - t < window]
        if len(_rate_store[uid]) >= max_req:
            return False
        _rate_store[uid].append(now)
        return True


def utc_now():
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE MODELS
# ═════════════════════════════════════════════════════════════════════════════

class User(UserMixin, db.Model):
    __tablename__ = 'users'
    id            = db.Column(db.Integer, primary_key=True)
    username      = db.Column(db.String(80), unique=True, nullable=False)
    full_name     = db.Column(db.String(200), default='')
    password_hash = db.Column(db.String(200))
    email         = db.Column(db.String(200), default='')
    role          = db.Column(db.String(20), default='admin')
    created_at    = db.Column(db.DateTime, default=utc_now)
    is_active     = db.Column(db.Boolean, default=True)
    last_login    = db.Column(db.DateTime, nullable=True)


class Intent(db.Model):
    __tablename__ = 'intents'
    id          = db.Column(db.Integer, primary_key=True)
    tag         = db.Column(db.String(100), unique=True, nullable=False)
    name        = db.Column(db.String(200))
    description = db.Column(db.Text)
    is_active   = db.Column(db.Boolean, default=True)
    created_at  = db.Column(db.DateTime, default=utc_now)
    updated_at  = db.Column(db.DateTime, default=utc_now, onupdate=utc_now)
    patterns    = db.relationship('Pattern',  backref='intent', lazy='select',
                                  cascade='all, delete-orphan')
    responses   = db.relationship('Response', backref='intent', lazy='select',
                                  cascade='all, delete-orphan')


class Pattern(db.Model):
    __tablename__ = 'patterns'
    id           = db.Column(db.Integer, primary_key=True)
    intent_id    = db.Column(db.Integer, db.ForeignKey('intents.id'), nullable=False)
    pattern_text = db.Column(db.String(500), nullable=False)
    language     = db.Column(db.String(10), default='en')
    usage_count  = db.Column(db.Integer, default=0)
    created_at   = db.Column(db.DateTime, default=utc_now)


class Response(db.Model):
    __tablename__  = 'responses'
    id             = db.Column(db.Integer, primary_key=True)
    intent_id      = db.Column(db.Integer, db.ForeignKey('intents.id'), nullable=False)
    response_text  = db.Column(db.Text, nullable=False)
    language       = db.Column(db.String(10), default='en')
    usage_count    = db.Column(db.Integer, default=0)
    created_at     = db.Column(db.DateTime, default=utc_now)


class Conversation(db.Model):
    __tablename__   = 'conversations'
    id              = db.Column(db.Integer, primary_key=True)
    user_id         = db.Column(db.String(100))
    user_name       = db.Column(db.String(120), default='')
    phone_number    = db.Column(db.String(30), default='')
    message         = db.Column(db.Text)
    response        = db.Column(db.Text)
    intent          = db.Column(db.String(50))
    confidence      = db.Column(db.Float)
    sentiment       = db.Column(db.String(20))
    sentiment_score = db.Column(db.Float)
    entities        = db.Column(db.Text)
    language        = db.Column(db.String(10), default='en')
    feedback        = db.Column(db.Integer)
    timestamp       = db.Column(db.DateTime, default=utc_now)


class Unanswered(db.Model):
    __tablename__ = 'unanswered'
    id          = db.Column(db.Integer, primary_key=True)
    question    = db.Column(db.String(500), unique=True, nullable=False)
    user_id     = db.Column(db.String(100))
    asked_count = db.Column(db.Integer, default=1)
    language    = db.Column(db.String(10), default='en')
    last_asked  = db.Column(db.DateTime, default=utc_now)
    resolved    = db.Column(db.Boolean, default=False)


class AppSettings(db.Model):
    __tablename__ = 'app_settings'
    key   = db.Column(db.String(100), primary_key=True)
    value = db.Column(db.Text, nullable=False, default='{}')

    @classmethod
    def get(cls, key: str, default=None):
        row = db.session.get(cls, key)
        if row is None:
            return default
        try:
            return json.loads(row.value)
        except Exception:
            return default

    @classmethod
    def set(cls, key: str, value) -> None:
        row = db.session.get(cls, key)
        if row is None:
            row = cls(key=key)
            db.session.add(row)
        row.value = json.dumps(value)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()


@login_manager.user_loader
def load_user(user_id):
    return db.session.get(User, int(user_id))


# ═════════════════════════════════════════════════════════════════════════════
# LANGUAGE DETECTION
# ═════════════════════════════════════════════════════════════════════════════

_RE_TAMIL   = re.compile(r'[\u0B80-\u0BFF]')
_RE_SINHALA = re.compile(r'[\u0D80-\u0DFF]')



def detect_language(text: str) -> str:
    """
    Detects language based on character blocks and Romanised cues.
    Returns: 'si', 'ta', or 'en' (default).
    """
    if not text:
        return 'en'

    text_lower = text.lower()

    # 1. Check for native Unicode characters
    has_sinhala = bool(re.search(r'[඀-෿]', text_lower))
    has_tamil = bool(re.search(r'[஀-௿]', text_lower))

    if has_sinhala and not has_tamil:
        return 'si'
    if has_tamil and not has_sinhala:
        return 'ta'

    # 2. Check for Romanised cues using dataset_loader
    dl = get_dataset_loader()
    words = set(re.findall(r'\b\w+\b', text_lower))

    for cue in dl.language_config.get('romanised_tamil', []):
        if re.search(r'\b' + re.escape(cue) + r'\b', text_lower):
            return 'ta'
    for cue in dl.language_config.get('romanised_sinhala', []):
        if re.search(r'\b' + re.escape(cue) + r'\b', text_lower):
            return 'si'

    if words.intersection(dl.language_config.get('romanised_sinhala', [])):
        return 'si'
    if words.intersection(dl.language_config.get('romanised_tamil', [])):
        return 'ta'

    ta_count = len(_RE_TAMIL.findall(text))
    si_count = len(_RE_SINHALA.findall(text))
    if ta_count > 0 and ta_count >= si_count:
        return 'ta'
    if si_count > 0:
        return 'si'

    return 'en'


# ═════════════════════════════════════════════════════════════════════════════
# MULTILINGUAL CHATBOT ENGINE
# ═════════════════════════════════════════════════════════════════════════════



# =============================================================================
#  PROGRAMME DETAIL PROFILES — Individual programme lookup responses
# =============================================================================


class MultilingualChatbotEngine:

    def __init__(self):
        self.lemmatizer         = WordNetLemmatizer()
        self.sentiment_analyzer = SentimentIntensityAnalyzer()

        self.spelling_corrections = get_dataset_loader().language_config["spelling_corrections"]

        self.faq_en = self._build_faq_en()
        self.faq_ta = self._build_faq_ta()
        self.faq_si = self._build_faq_si()

        self._flat_en: list[tuple[str, str]] = []
        self._flat_ta: list[tuple[str, str]] = []
        self._flat_si: list[tuple[str, str]] = []

        self._db_intents_en: list[tuple[str, list[str], list[str]]] = []
        self._db_intents_ta: list[tuple[str, list[str], list[str]]] = []
        self._db_intents_si: list[tuple[str, list[str], list[str]]] = []

        self._rebuild_flat_patterns()

    # ─────────────────────────────────────────────────────────────────────
    # FAQ BUILDERS
    # ─────────────────────────────────────────────────────────────────────

    def _build_faq_en(self):
        return {k: {'patterns': v['patterns']['en'], 'answer': v['answer']['en']}
                for k, v in get_dataset_loader().intents.items()}

    def _build_faq_ta(self):
        return {k: {'patterns': v['patterns']['ta'], 'answer': v['answer']['ta']}
                for k, v in get_dataset_loader().intents.items()}

    def _build_faq_si(self):
        return {k: {'patterns': v['patterns']['si'], 'answer': v['answer']['si']}
                for k, v in get_dataset_loader().intents.items()}


    # ─────────────────────────────────────────────────────────────────────
    # FLAT PATTERN BUILDERS
    # ─────────────────────────────────────────────────────────────────────

    def _rebuild_flat_patterns(self):
        self._flat_en = [(p.lower(), k) for k, v in self.faq_en.items()
                         for p in v['patterns']]
        # For Tamil/Sinhala: lowercase ASCII portions to match clean_text_ta/si output
        self._flat_ta = [(''.join(c.lower() if ord(c) < 128 and c.isalpha() else c for c in p), k)
                         for k, v in self.faq_ta.items() for p in v['patterns']]
        self._flat_si = [(''.join(c.lower() if ord(c) < 128 and c.isalpha() else c for c in p), k)
                         for k, v in self.faq_si.items() for p in v['patterns']]
        # Also include PROGRAMME_PROFILES keywords (use 'keywords' key, not 'patterns')
        for k, v in get_dataset_loader().programme_profiles.items():
            kw = v.get('keywords', {})
            self._flat_en += [(kw_en.lower(), f'__prog__{k}') for kw_en in kw.get('en', [])]
            self._flat_ta += [(kw_ta, f'__prog__{k}') for kw_ta in kw.get('ta', [])]
            self._flat_si += [(kw_si, f'__prog__{k}') for kw_si in kw.get('si', [])]

    # ─────────────────────────────────────────────────────────────────────
    # TEXT CLEANING
    # ─────────────────────────────────────────────────────────────────────

    def clean_text_en(self, text: str) -> str:
        text = text.lower().strip()
        text = unicodedata.normalize('NFC', text)
        text = re.sub(r'[^\w\s]', ' ', text)
        text = re.sub(r'\s+', ' ', text)
        words = text.split()
        corrected = [self.spelling_corrections.get(w, w) for w in words]
        text = ' '.join(corrected)
        # FIX: Skip lemmatization for short queries
        if len(words) <= 3:
            return text.strip()
        lemmatized = [self.lemmatizer.lemmatize(w) for w in text.split()]
        return ' '.join(lemmatized).strip()

    def clean_text_ta(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        # Lowercase ASCII portions (e.g. "HDIT", "BSc") so they match lowercased patterns
        text = ''.join(c.lower() if ord(c) < 128 and c.isalpha() else c for c in text)
        text = re.sub(r'[^\u0B80-\u0BFF\s\w]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    def clean_text_si(self, text: str) -> str:
        text = unicodedata.normalize('NFC', text)
        text = text.strip()
        # Lowercase ASCII portions so English loanwords/programme names match
        text = ''.join(c.lower() if ord(c) < 128 and c.isalpha() else c for c in text)
        text = re.sub(r'[^\u0D80-\u0DFF\s\w]', ' ', text)
        return re.sub(r'\s+', ' ', text).strip()

    # ─────────────────────────────────────────────────────────────────────
    # FUZZY MATCHING ENGINE (RapidFuzz with difflib fallback)
    # ─────────────────────────────────────────────────────────────────────

    def _fuzzy_score(self, query: str, candidate: str) -> float:
        if RAPIDFUZZ:
            tset = fuzz.token_set_ratio(query, candidate)
            pr   = fuzz.partial_ratio(query, candidate)
            return (0.6 * tset + 0.4 * pr) / 100.0
        m = SequenceMatcher(None, query, candidate)
        return m.ratio()

    def _match_patterns(
        self,
        query: str,
        flat_patterns: list[tuple[str, str]],
        threshold: float = 0.72
    ) -> tuple[str | None, str | None, float]:
        """Match *query* against *flat_patterns*.

        Design principles (v2 — stricter to prevent false positives):
        • Exact equality          → score 1.0  (always accepted)
        • Substring/superset      → only accepted when the shared token-overlap
          is high enough that the match is semantically plausible.
        • Fuzzy scoring           → raised threshold (0.72) to avoid accepting
          unrelated questions (e.g. "who is the president" should NOT match
          "transfer to icst" just because partial_ratio gives 0.56).
        • Single-word queries     → accepted only on exact equality with a
          pattern word; no fuzzy boost (prevents "president" → admission hit).
        """
        if not flat_patterns:
            return None, None, 0.0

        best_key   = None
        best_pat   = None
        best_score = 0.0

        q_words  = set(query.split())
        q_len    = len(query)

        # ── Phase 1: Exact / token-level substring match ─────────────────────
        exact_candidates: list[tuple[float, str, str]] = []
        for pat, key in flat_patterns:
            if query == pat:
                exact_candidates.append((1.0, pat, key))
                continue

            p_words = set(pat.split())

            # Pattern is a substring of the query (e.g. "how to apply" in
            # "i want to know how to apply to icst").
            # Only accept if the pattern is at least 40 % of the query length
            # (prevents very short patterns from matching long unrelated queries).
            if pat in query:
                coverage = len(pat) / max(q_len, 1)
                if coverage >= 0.40:
                    specificity = len(pat) / max(q_len, 1)
                    exact_candidates.append((0.90 + 0.05 * specificity, pat, key))
                continue

            # Query is a substring of the pattern.
            if query in pat:
                coverage = len(query) / max(len(pat), 1)
                if coverage >= 0.40:
                    specificity = len(query) / max(len(pat), 1)
                    exact_candidates.append((0.85 + 0.05 * specificity, pat, key))
                continue

            # All query words appear in the pattern (multi-word only).
            if len(q_words) >= 2 and q_words and q_words.issubset(p_words):
                exact_candidates.append((0.88, pat, key))

        if exact_candidates:
            exact_candidates.sort(key=lambda x: -x[0])
            best_score_ex, best_pat_ex, best_key_ex = exact_candidates[0]
            return best_key_ex, best_pat_ex, best_score_ex

        # ── Phase 2: Fuzzy scoring ────────────────────────────────────────────
        # For single-word queries we do NOT use fuzzy matching at all.
        # A query like "president" should only match if "president" literally
        # appears as a word in a pattern — never via character similarity.
        if len(q_words) == 1:
            word = next(iter(q_words))
            if len(word) >= 4:
                for pat, key in flat_patterns:
                    if word in set(pat.split()):
                        score = 0.80
                        if score > best_score:
                            best_score = score
                            best_key   = key
                            best_pat   = pat
            if best_score >= threshold:
                return best_key, best_pat, best_score
            return None, None, 0.0

        if RAPIDFUZZ:
            for pat, key in flat_patterns:
                p_words = set(pat.split())
                # token_sort_ratio is more conservative than token_set_ratio
                # for out-of-domain queries; use a blend that favours precision.
                tsort = fuzz.token_sort_ratio(query, pat)
                tset  = fuzz.token_set_ratio(query, pat)
                pr    = fuzz.partial_ratio(query, pat)
                # Weighted blend — token_sort dominates so character-level
                # partial matches don't inflate the score for unrelated inputs.
                score = (0.50 * tsort + 0.30 * tset + 0.20 * pr) / 100.0
                # Boost only for genuine multi-word subset
                if len(q_words) >= 2 and q_words.issubset(p_words):
                    score = max(score, 0.85)
                if score > best_score:
                    best_score = score
                    best_key   = key
                    best_pat   = pat
        else:
            for pat, key in flat_patterns:
                p_words = set(pat.split())
                score = self._fuzzy_score(query, pat)
                if len(q_words) >= 2 and q_words.issubset(p_words):
                    score = max(score, 0.80)
                if score > best_score:
                    best_score = score
                    best_key   = key
                    best_pat   = pat

        if best_score >= threshold:
            return best_key, best_pat, best_score
        return None, None, 0.0

    # ─────────────────────────────────────────────────────────────────────
    # SENTIMENT ANALYSIS
    # ─────────────────────────────────────────────────────────────────────

    def analyze_sentiment(self, text: str) -> tuple[str, float]:
        try:
            scores = self.sentiment_analyzer.polarity_scores(text)
            c = scores['compound']
            if c >= 0.05:  return 'positive', c
            if c <= -0.05: return 'negative', c
            return 'neutral', c
        except Exception:
            return 'neutral', 0.0

    # ─────────────────────────────────────────────────────────────────────
    # ENTITY EXTRACTION
    # ─────────────────────────────────────────────────────────────────────

    def extract_entities(self, text: str) -> str:
        if not nlp or not text:
            return '[]'
        try:
            doc = nlp(text[:200])
            ents = [{'text': e.text, 'label': e.label_} for e in doc.ents]
            return json.dumps(ents)
        except Exception:
            return '[]'

    # ─────────────────────────────────────────────────────────────────────
    # DB INTENT MATCHING
    # ─────────────────────────────────────────────────────────────────────

    def load_intents_from_db(self):
        self._db_intents_en = []
        self._db_intents_ta = []
        self._db_intents_si = []
        try:
            for intent in Intent.query.filter_by(is_active=True).all():
                for lang, store in [('en', self._db_intents_en),
                                    ('ta', self._db_intents_ta),
                                    ('si', self._db_intents_si)]:
                    ps = [p.pattern_text for p in intent.patterns if p.language == lang]
                    rs = [r.response_text for r in intent.responses if r.language == lang]
                    # FIX: English fallback for Tamil/Sinhala intents
                    if not rs:
                        rs = [r.response_text for r in intent.responses if r.language == 'en']
                    if ps and rs:
                        store.append((intent.tag, ps, rs))
            logger.info(f"DB intents loaded: EN={len(self._db_intents_en)} "
                        f"TA={len(self._db_intents_ta)} SI={len(self._db_intents_si)}")
        except Exception as e:
            logger.error(f"DB intents error: {e}")

    def _db_intent_match(
        self,
        query: str,
        db_intents: list[tuple[str, list[str], list[str]]],
        threshold: float = 0.72
    ) -> tuple[str | None, str | None, float]:
        best_tag   = None
        best_resp  = None
        best_score = 0.0

        q_words = set(query.lower().split())

        for tag, patterns, responses in db_intents:
            for pat in patterns:
                q = query.lower()
                p = pat.lower()
                p_words = set(p.split())

                # Exact match
                if q == p:
                    return tag, random.choice(responses), 1.0

                # Substring match — require meaningful coverage to avoid
                # "president" matching "transfer to icst" via a short token.
                if p in q:
                    coverage = len(p) / max(len(q), 1)
                    if coverage >= 0.40:
                        specificity = len(p) / max(len(q), 1)
                        score = 0.85 + 0.05 * specificity
                        if score > best_score:
                            best_score = score
                            best_tag   = tag
                            best_resp  = random.choice(responses)
                    continue

                if q in p:
                    coverage = len(q) / max(len(p), 1)
                    if coverage >= 0.40:
                        specificity = len(q) / max(len(p), 1)
                        score = 0.85 + 0.05 * specificity
                        if score > best_score:
                            best_score = score
                            best_tag   = tag
                            best_resp  = random.choice(responses)
                    continue

                score = self._fuzzy_score(q, p)
                # Only boost multi-word subset (never single-word)
                if len(q_words) >= 2 and q_words.issubset(p_words):
                    score = max(score, 0.82)
                if score > best_score:
                    best_score = score
                    best_tag   = tag
                    best_resp  = random.choice(responses)

        if best_score >= threshold:
            return best_tag, best_resp, best_score
        return None, None, 0.0

    # ─────────────────────────────────────────────────────────────────────
    # FALLBACK RESPONSES (multilingual)
    # ─────────────────────────────────────────────────────────────────────

    def _get_fallback(self, lang: str) -> str:
        dl = get_dataset_loader()
        if lang == 'ta' and dl.fallback_responses['ta']:
            return random.choice(dl.fallback_responses['ta'])
        if lang == 'si' and dl.fallback_responses['si']:
            return random.choice(dl.fallback_responses['si'])
        if dl.fallback_responses['en']:
            return random.choice(dl.fallback_responses['en'])
        return "Sorry, I couldn't find information related to your question. Please contact ICST University at 📞 0743 444 444 for further assistance." 

    def _is_out_of_domain(self, cleaned_query: str) -> bool:
        """Return True when the query obviously falls outside ICST's knowledge base."""
        lower = cleaned_query.lower()
        dl = get_dataset_loader()
        for kw in dl.language_config['out_of_domain_keywords']:
            if re.search(r'\b' + re.escape(kw) + r'\b', lower):
                return True
        return False

    # ─────────────────────────────────────────────────────────────────────
    # MAIN GET RESPONSE
    # ─────────────────────────────────────────────────────────────────────

    def get_response(
        self,
        message: str,
        user_id: str,
        visitor_name: str = '',
        visitor_phone: str = '',
    ) -> tuple[str, str, float, int | None]:
        try:
            original = message
            lang     = detect_language(message)

            # ── Minimum message length guard ────────────────────────────────────
            # Messages shorter than 2 characters cannot be meaningfully matched.
            if len(message.strip()) < 2:
                return self._get_fallback(lang), lang, 0.0, None

            # Choose correct FAQ and flat patterns
            if lang == 'ta':
                cleaned  = self.clean_text_ta(message)
                faq_dict = self.faq_ta
                flat     = self._flat_ta
                db_intents = self._db_intents_ta
            elif lang == 'si':
                cleaned  = self.clean_text_si(message)
                faq_dict = self.faq_si
                flat     = self._flat_si
                db_intents = self._db_intents_si
            else:
                cleaned  = self.clean_text_en(message)
                faq_dict = self.faq_en
                flat     = self._flat_en
                db_intents = self._db_intents_en

            # ── Out-of-domain fast-path ──────────────────────────────────────────
            # Skip all matching and return fallback immediately for questions
            # that are clearly outside the ICST knowledge base.
            if self._is_out_of_domain(cleaned):
                response = self._get_fallback(lang)
                tag = 'fallback'
                score = 0.0
                # Still log it as unanswered so admins can review it.
                try:
                    norm = original.strip()[:500]
                    existing = Unanswered.query.filter_by(question=norm).first()
                    if existing:
                        existing.asked_count += 1
                        existing.last_asked   = utc_now()
                    else:
                        db.session.add(Unanswered(
                            question=norm, user_id=user_id,
                            language=lang, last_asked=utc_now()
                        ))
                    db.session.commit()
                except Exception:
                    try: db.session.rollback()
                    except: pass
                return response, lang, score, None

            # ── ML Intent Prediction ─────────────────────────────────────────────
            dl = get_dataset_loader()
            ml_tag, ml_score = dl.predict_intent(cleaned, lang)
            
            if ml_tag and ml_score >= 0.20:
                tag = ml_tag
                score = ml_score
                matched_pat = cleaned
            else:
                # Fallback to fuzzy matching if ML confidence is low
                tag, matched_pat, score = self._match_patterns(cleaned, flat, threshold=0.72)

            # ── Romanized Tamil/Sinhala fallback: try English matching ──────────
           
            if not tag and lang in ('ta', 'si'):
                is_ascii_query = all(ord(c) < 128 for c in cleaned.replace(' ', ''))
                if is_ascii_query:
                    cleaned_en = self.clean_text_en(message)
                    # Try English ML Model for Romanized inputs
                    rom_tag, rom_score = dl.predict_intent(cleaned_en, 'en')
                    if rom_tag and rom_score >= 0.20:
                        tag = rom_tag
                        score = rom_score
                        matched_pat = cleaned_en
                    else:
                        # Fallback to English fuzzy matching
                        tag, matched_pat, score = self._match_patterns(
                            cleaned_en, self._flat_en, threshold=0.72
                        )

            # DB intent match (raised threshold to 0.72 to prevent false positives)
            _response_set = False
            response      = None
            if not tag:
                tag_db, resp_db, score_db = self._db_intent_match(cleaned, db_intents, threshold=0.72)
                # Also try English DB intents as fallback for non-English
                if not tag_db and lang != 'en':
                    cleaned_en = self.clean_text_en(message)
                    tag_db, resp_db, score_db = self._db_intent_match(
                        cleaned_en, self._db_intents_en, threshold=0.72
                    )
                if tag_db and resp_db:
                    tag, score = tag_db, score_db
                    response = resp_db
                    _response_set = True
                else:
                    tag = None

            # Only look up FAQ response if tag was set by pattern match (not DB which already set response)
            if tag and (not _response_set):
                # Check if this is a PROGRAMME_PROFILES match
                if tag.startswith('__prog__'):
                    prog_key = tag[len('__prog__'):]
                    prog = get_dataset_loader().programme_profiles.get(prog_key, {})
                    prog_answer = prog.get('answer', {})
                    response = prog_answer.get(lang) or prog_answer.get('en')
                    if not response:
                        tag = None
                # Get response from built-in FAQ
                elif tag in faq_dict:
                    response = faq_dict[tag]['answer']
                elif lang != 'en' and tag in self.faq_en:
                    response = self.faq_en[tag]['answer']
                else:
                    tag      = None
                    response = None

            if tag is None or response is None:
                response   = self._get_fallback(lang)
                tag        = 'fallback'
                score      = 0.0
                # Save to unanswered (with app context protection)
                def _save_unanswered():
                    try:
                        norm = original.strip()[:500]
                        existing = Unanswered.query.filter_by(question=norm).first()
                        if existing:
                            existing.asked_count += 1
                            existing.last_asked   = utc_now()
                        else:
                            db.session.add(Unanswered(
                                question=norm, user_id=user_id,
                                language=lang, last_asked=utc_now()
                            ))
                        db.session.commit()
                    except Exception as ue:
                        logger.warning(f"Unanswered save error: {ue}")
                        try: db.session.rollback()
                        except: pass

                try:
                    _save_unanswered()
                except Exception:
                    try:
                        with app.app_context():
                            _save_unanswered()
                    except Exception as e:
                        logger.error(f"Unanswered context error: {e}")

            sentiment, sentiment_score = self.analyze_sentiment(original)
            entities = self.extract_entities(original) if lang == 'en' else '[]'

            conv_id = None
            try:
                _vname  = visitor_name
                _vphone = visitor_phone

                def _save_conv():
                    nonlocal conv_id
                    conv = Conversation(
                        user_id=user_id, message=original,
                        user_name=_vname, phone_number=_vphone,
                        response=response, intent=tag,
                        confidence=score,
                        sentiment=sentiment, sentiment_score=sentiment_score,
                        entities=entities, language=lang
                    )
                    db.session.add(conv)
                    db.session.commit()
                    conv_id = conv.id

                try:
                    _save_conv()
                except Exception:
                    with app.app_context():
                        _save_conv()
            except Exception as e:
                logger.error(f"DB persist error: {e}")
                try: db.session.rollback()
                except: pass

            return response, lang, score, conv_id

        except Exception as e:
            logger.error(f"get_response error: {e}", exc_info=True)
            return (
                "Sorry, something went wrong. Please call 📞 0743 444 444",
                'en', 0.0, None
            )


# ─────────────────────────────────────────────────────────────────────────────
# SINGLETON
# ─────────────────────────────────────────────────────────────────────────────

_chatbot: MultilingualChatbotEngine | None = None
_chatbot_lock = threading.Lock()

def get_chatbot() -> MultilingualChatbotEngine:
    global _chatbot
    if _chatbot is None:
        with _chatbot_lock:
            if _chatbot is None:
                _chatbot = MultilingualChatbotEngine()
    return _chatbot

# ═══════════════════════════════════════════════════════════════════════════
# SINHALA TTS USING gTTS (Native Sinhala Voice)
# ═══════════════════════════════════════════════════════════════════════════


def sinhala_text_to_speech(text: str) -> tuple:
    """
    Convert Sinhala text to speech using gTTS with full Sinhala support
    """
    if not text or not text.strip():
        return None, None
    
    try:
        # Use direct Sinhala TTS (gTTS supports 'si' language)
        tts = gTTS(text=text, lang='si', slow=False)
        
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        return audio_base64, 'audio/mpeg'
        
    except Exception as e:
        logger.error(f"Sinhala TTS error: {e}")
        
        # Fallback: Try with English (poor quality but works)
        try:
            tts = gTTS(text=text, lang='en', slow=False)
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            return audio_base64, 'audio/mpeg'
        except Exception as e2:
            logger.error(f"Fallback TTS also failed: {e2}")
            return None, None

# ─────────────────────────────────────────────────────────────────────────────
# USER CONTEXT (per-session, in-memory)
# ─────────────────────────────────────────────────────────────────────────────

_user_context: dict[str, dict] = {}
_ctx_lock = threading.Lock()
_CTX_TTL  = 120


def _get_ctx(uid: str) -> dict:
    now = time.time()
    with _ctx_lock:
        ctx = _user_context.get(uid, {})
        if ctx and (now - ctx.get('ts', 0)) > _CTX_TTL:
            ctx = {}
        return ctx

def _set_ctx(uid: str, **kwargs):
    with _ctx_lock:
        _user_context[uid] = {**kwargs, 'ts': time.time()}


# ═════════════════════════════════════════════════════════════════════════════
# RESPONSE FORMATTING
# ═════════════════════════════════════════════════════════════════════════════

_EMOJI_HEADER_RE = re.compile(
    r'^([\U0001F300-\U0001FFFF\U00002600-\U000027BF\U0000FE00-\U0000FEFF'
    r'\U0001F900-\U0001F9FF\U00002702-\U000027B0]+\s*)',
    re.UNICODE,
)


def format_response(text: str) -> str:
    if not text:
        return ''
    text = text.replace('\\n', '\n')
    text = re.sub(
        r'(?<!["\'/])(\+?(?:\d[\d\s\-\(\)]{5,}\d))',
        lambda m: f'<a href="tel:{re.sub("[^+0-9]", "", m.group(1))}">{m.group(1)}</a>',
        text,
    )
    text = re.sub(r'(https?://[^\s<>"]+)',
                  r'<a href="\1" target="_blank" rel="noopener">\1</a>', text)
    text = re.sub(r'(?<!["/])(www\.[^\s<>"]+)',
                  r'<a href="https://\1" target="_blank" rel="noopener">\1</a>', text)
    text = re.sub(r'([a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,})',
                  r'<a href="mailto:\1">\1</a>', text)

    lines  = text.split('\n')
    blocks = []
    ol_items: list[tuple[str, list[str]]] = []
    ul_buf:   list[str] = []

    def _flush_ul():
        if ul_buf:
            blocks.append(('ul', list(ul_buf)))
            ul_buf.clear()

    def _flush_ol():
        if ol_items:
            blocks.append(('ol', list(ol_items)))
            ol_items.clear()

    for line in lines:
        line = line.strip()
        if not line:
            _flush_ul(); _flush_ol()
            continue
        if re.match(r'^\d+\.\s+', line):
            _flush_ul()
            ol_items.append((re.sub(r'^\d+\.\s+', '', line), []))
            continue
        if re.match(r'^[•\-\*]\s+', line) and not line.startswith('**'):
            bullet = re.sub(r'^[•\-\*]\s+', '', line)
            if ol_items:
                ol_items[-1][1].append(bullet)
            else:
                _flush_ol()
                ul_buf.append(bullet)
            continue
        _flush_ul(); _flush_ol()
        if line.endswith(':'):
            heading = _EMOJI_HEADER_RE.sub('', line).rstrip(':').strip()
            if heading:
                blocks.append(('heading', heading))
                continue
        blocks.append(('p', line))

    _flush_ul(); _flush_ol()

    parts = []
    for kind, content in blocks:
        if kind == 'heading':
            parts.append(f'<p class="chat-section-heading"><strong>{content}</strong></p>')
        elif kind == 'p':
            parts.append(f'<p>{content}</p>')
        elif kind == 'ul':
            items = ''.join(f'<li>{item}</li>' for item in content)
            parts.append(f'<ul>{items}</ul>')
        elif kind == 'ol':
            li_tags = []
            for header, subs in content:
                if subs:
                    nested = '<ul>' + ''.join(f'<li>{s}</li>' for s in subs) + '</ul>'
                    li_tags.append(f'<li>{header}{nested}</li>')
                else:
                    li_tags.append(f'<li>{header}</li>')
            parts.append(f'<ol>{"".join(li_tags)}</ol>')

    return ''.join(parts)


# ═════════════════════════════════════════════════════════════════════════════
# FORMS
# ═════════════════════════════════════════════════════════════════════════════

class IntentForm(FlaskForm):
    tag         = StringField('Tag',  validators=[DataRequired()])
    name        = StringField('Name', validators=[DataRequired()])
    description = TextAreaField('Description')
    is_active   = BooleanField('Active')


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — HEALTH
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/health')
def health_check():
    db_ok = True
    try:
        db.session.execute(db.text('SELECT 1'))
    except Exception:
        db_ok = False
    return jsonify({
        'status':         'ok',
        'timestamp':      utc_now().isoformat(),
        'database':       'ok' if db_ok else 'error',
        'chatbot_loaded': _chatbot is not None,
        'engine':         'multilingual-direct-v2.1',
        'rapidfuzz':      RAPIDFUZZ,
    })


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — MAIN CHAT
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/')
def index():
    session.setdefault('user_id', str(uuid.uuid4()))
    msg = request.args.get('message', '')
    return render_template('index.html', prefill_message=msg)


@app.route('/voice')
def voice():
    session.setdefault('user_id', str(uuid.uuid4()))
    return render_template('index.html', prefill_message='', voice_mode=True)


@app.route('/chat', methods=['POST'])
def chat():
    uid = session.setdefault('user_id', str(uuid.uuid4()))

    if not _check_rate_limit(uid):
        return jsonify({'error': 'Too many requests. Please slow down.'}), 429

    data    = request.get_json() or {}
    message = (data.get('message', '') or '').strip()

    if not message:
        return jsonify({'error': 'Empty message'}), 400
    if len(message) > 2000:
        return jsonify({'error': 'Message too long (max 2000 characters)'}), 400

    visitor_name  = data.get('visitor_name', '') or session.get('visitor_name', '')
    visitor_phone = data.get('visitor_phone', '') or session.get('visitor_phone', '')
    if visitor_name:
        session['visitor_name']  = visitor_name
    if visitor_phone:
        session['visitor_phone'] = visitor_phone

    try:
        response, lang, confidence, conv_id = get_chatbot().get_response(
            message, uid, visitor_name, visitor_phone
        )
        return jsonify({
            'response':        format_response(response),
            'raw':             response,
            'language':        lang,
            'confidence':      round(confidence, 3),
            'conversation_id': conv_id,
        })
    except Exception as e:
        logger.error(f"chat route error: {e}", exc_info=True)
        return jsonify({'error': 'Server error. Please try again.'}), 500


@app.route('/chat/feedback', methods=['POST'])
def chat_feedback():
    data   = request.get_json() or {}
    msg_id = data.get('message_id')
    rating = data.get('rating')

    if msg_id is None or rating not in (1, -1):
        return jsonify({'error': 'Invalid feedback'}), 400

    try:
        conv = db.session.get(Conversation, int(msg_id))
        if conv:
            conv.feedback = rating
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500

    return jsonify({'success': True})


@app.route('/chat/language', methods=['POST'])
def detect_language_route():
    data = request.get_json() or {}
    text = data.get('text', '')
    lang = detect_language(text)
    return jsonify({'language': lang})


# ── Visitor registration ──────────────────────────────────────────────────────
@app.route('/register', methods=['POST'])
@csrf.exempt
def register_visitor():
    data  = request.get_json() or {}
    name  = (data.get('name', '') or '').strip()[:120]
    phone = (data.get('phone', '') or '').strip()[:30]
    if not name:
        return jsonify({'status': 'error', 'message': 'Name is required.'}), 400
    session.setdefault('user_id', str(uuid.uuid4()))
    if name:  session['visitor_name']  = name
    if phone: session['visitor_phone'] = phone
    return jsonify({'status': 'ok'})


# ── Feedback alias ─────────────────────────────────────────────────────────────
@app.route('/feedback', methods=['POST'])
@csrf.exempt
def feedback_alias():
    data    = request.get_json() or {}
    conv_id = data.get('conversation_id')
    rating  = data.get('rating')
    if conv_id is None or rating not in (0, 1):
        return jsonify({'status': 'error', 'error': 'Invalid feedback'}), 400
    try:
        conv = db.session.get(Conversation, int(conv_id))
        if conv:
            conv.feedback = rating
            db.session.commit()
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'error': str(e)}), 500
    return jsonify({'status': 'success'})


# ── SSE streaming chat ─────────────────────────────────────────────────────────
@app.route('/chat/stream', methods=['GET'])
@csrf.exempt
def chat_stream():
    uid     = session.setdefault('user_id', str(uuid.uuid4()))
    message = (request.args.get('message', '') or '').strip()
    _uid    = uid
    _vname  = session.get('visitor_name',  '')
    _vph    = session.get('visitor_phone', '')

    if not message:
        def _err():
            yield 'data: ' + json.dumps({'type': 'error', 'text': 'Empty message'}) + '\n\n'
        return FlaskResponse(_err(), mimetype='text/event-stream')

    if not _check_rate_limit(uid):
        def _rl():
            yield 'data: ' + json.dumps({'type': 'error', 'text': 'Too many requests.'}) + '\n\n'
        return FlaskResponse(_rl(), mimetype='text/event-stream')

    def generate():
        try:
            response, detected_lang, confidence, conv_id = get_chatbot().get_response(
                message, _uid, _vname, _vph
            )
            html  = format_response(response)
            CHUNK = 50
            for i in range(0, len(html), CHUNK):
                yield 'data: ' + json.dumps({'type': 'token', 'text': html[i:i+CHUNK]}) + '\n\n'
                time.sleep(0.012)
            yield 'data: ' + json.dumps({
                'type':            'done',
                'language':        detected_lang,
                'confidence':      round(confidence, 3),
                'conversation_id': conv_id,
                'raw':             response,
            }) + '\n\n'
        except Exception as e:
            logger.error(f"SSE stream error: {e}", exc_info=True)
            yield 'data: ' + json.dumps({'type': 'error', 'text': 'Server error.'}) + '\n\n'

    return FlaskResponse(
        generate(),
        mimetype='text/event-stream',
        headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'}
    )

# ═══════════════════════════════════════════════════════════════════════════
# VOICE CHAT ENDPOINT - Supports BOTH JSON (for TTS) and Audio (for STT)
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/voice/chat', methods=['POST'])
@csrf.exempt
def voice_chat():
    """Handle voice requests - Supports JSON for TTS and FormData for audio"""
    uid = session.setdefault('user_id', str(uuid.uuid4()))
    
    if not _check_rate_limit(uid):
        return jsonify({'error': 'Too many requests. Please slow down.'}), 429
    
    visitor_name = session.get('visitor_name', '')
    visitor_phone = session.get('visitor_phone', '')
    
    # Check if this is a JSON request (TTS only) or file upload (STT + TTS)
    if request.is_json:
        # JSON request - just convert text to speech
        data = request.get_json() or {}
        user_text = data.get('text', '')
        requested_lang = data.get('lang', 'en')
        
        if not user_text:
            return jsonify({'error': 'No text provided'}), 400
        
        # Get bot response
        bot = get_chatbot()
        response_text, detected_lang, confidence, conv_id = bot.get_response(
            user_text, uid, visitor_name, visitor_phone
        )
        
        # Format response
        formatted_response = format_response(response_text)
        
        # Generate audio using gTTS
        try:
            # gTTS supports 'si', 'ta', and 'en' languages
            if detected_lang == 'si':
                tts_lang = 'si'
            elif detected_lang == 'ta':
                tts_lang = 'ta'
            else:
                tts_lang = 'en'
            
            tts = gTTS(text=response_text, lang=tts_lang, slow=False)
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'text': user_text,
                'response_text': formatted_response,
                'audio': audio_base64,
                'mime': 'audio/mpeg',
                'language': detected_lang,
                'conversation_id': conv_id
            })
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return jsonify({
                'success': True,
                'text': user_text,
                'response_text': formatted_response,
                'audio': None,
                'language': detected_lang,
                'conversation_id': conv_id
            })
    
    else:
        # FormData request - with audio file (STT)
        if 'audio' not in request.files:
            return jsonify({'error': 'No audio file provided'}), 400
        
        audio_file = request.files['audio']
        user_lang = request.form.get('lang', 'en')
        user_text = request.form.get('text', '')
        
        # If no text provided, you'd need STT here
        # For now, use sample or require text
        if not user_text:
            # Placeholder - you need to implement speech-to-text
            sample_messages = {
                'en': "Tell me about courses at ICST University",
                'ta': "ICST பல்கலைக்கழகத்தில் என்ன படிப்புகள் உள்ளன",
                'si': "ICST විශ්වවිද්‍යාලයේ පාඨමාලා මොනවාද"
            }
            user_text = sample_messages.get(user_lang, sample_messages['en'])
        
        # Get bot response
        bot = get_chatbot()
        response_text, detected_lang, confidence, conv_id = bot.get_response(
            user_text, uid, visitor_name, visitor_phone
        )
        
        formatted_response = format_response(response_text)
        
        # Generate audio
        try:
            # gTTS supports 'si', 'ta', and 'en' languages
            if detected_lang == 'si':
                tts_lang = 'si'
            elif detected_lang == 'ta':
                tts_lang = 'ta'
            else:
                tts_lang = 'en'
            
            tts = gTTS(text=response_text, lang=tts_lang, slow=False)
            
            audio_buffer = io.BytesIO()
            tts.write_to_fp(audio_buffer)
            audio_buffer.seek(0)
            audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
            
            return jsonify({
                'success': True,
                'text': user_text,
                'response_text': formatted_response,
                'audio': audio_base64,
                'mime': 'audio/mpeg',
                'language': detected_lang,
                'conversation_id': conv_id
            })
        except Exception as e:
            logger.error(f"TTS error: {e}")
            return jsonify({
                'success': True,
                'text': user_text,
                'response_text': formatted_response,
                'audio': None,
                'language': detected_lang,
                'conversation_id': conv_id
            })
            
# ── Support ────────────────────────────────────────────────────────────────────
@app.route('/support/ticket', methods=['POST'])
@login_required
def submit_support_ticket():
    data    = request.get_json() or {}
    subject = data.get('subject', '').strip()[:200]
    message = data.get('message', '').strip()[:2000]
    if not subject or not message:
        return jsonify({'error': 'Subject and message are required'}), 400
    logger.info(f"[Support Ticket] {current_user.username}: {subject}")
    return jsonify({'success': True})


@app.route('/support')
@login_required
def support_page():
    return render_template('support.html')


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — ADMIN LOGIN / LOGOUT
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/login', methods=['GET', 'POST'])
def admin_login():
    if current_user.is_authenticated:
        return redirect(url_for('admin_dashboard'))
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '').strip()
        try:
            user = User.query.filter_by(username=username).first()
            if user and check_password_hash(user.password_hash, password):
                login_user(user, remember=request.form.get('remember') == 'on')
                user.last_login = utc_now()
                db.session.commit()
                return redirect(url_for('admin_dashboard'))
            flash('Invalid credentials. Please try again.', 'error')
        except Exception as e:
            logger.error(f"Login error: {e}")
            flash('Login error. Please try again.', 'error')
    return render_template('admin_login.html')


@app.route('/admin/logout', methods=['GET', 'POST'])
@login_required
def admin_logout():
    logout_user()
    flash('You have been logged out.', 'info')
    return redirect(url_for('admin_login'))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — ADMIN DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/dashboard')
@login_required
def admin_dashboard():
    try:
        total_intents       = Intent.query.count()
        total_patterns      = Pattern.query.count()
        total_responses     = Response.query.count()
        total_conversations = Conversation.query.count()
        unanswered_count    = Unanswered.query.filter_by(resolved=False).count()

        recent = Conversation.query.order_by(Conversation.timestamp.desc()).limit(10).all()

        # FIX: Ensure unanswered is always a list
        unanswered = Unanswered.query.filter_by(resolved=False).order_by(
            Unanswered.asked_count.desc()
        ).limit(20).all()
        if unanswered is None:
            unanswered = []

        thirty_days_ago = utc_now() - timedelta(days=30)

        # FIX: Use func.date() instead of cast(..., Date) for SQLite compatibility
        daily_raw = db.session.query(
            func.date(Conversation.timestamp).label('date'),
            func.count(Conversation.id).label('count')
        ).filter(Conversation.timestamp >= thirty_days_ago)\
         .group_by(func.date(Conversation.timestamp))\
         .order_by(func.date(Conversation.timestamp)).all()

        daily = [{'date': str(r.date), 'count': r.count} for r in daily_raw]

        top_intents_raw = db.session.query(
            Conversation.intent, func.count(Conversation.id).label('count')
        ).filter(Conversation.intent.isnot(None))\
         .group_by(Conversation.intent)\
         .order_by(func.count(Conversation.id).desc()).limit(10).all()

        top_intents = [{'intent': r.intent or 'unknown', 'count': r.count}
                       for r in top_intents_raw]

        sentiments_raw = db.session.query(
            Conversation.sentiment, func.count(Conversation.id).label('count')
        ).filter(Conversation.sentiment.isnot(None))\
         .group_by(Conversation.sentiment).all()

        sentiments = {r.sentiment: r.count for r in sentiments_raw}

        lang_raw = db.session.query(
            Conversation.language, func.count(Conversation.id).label('count')
        ).group_by(Conversation.language).all()

        lang_distribution = {(r.language or 'en'): r.count for r in lang_raw}

        return render_template('admin_dashboard.html',
                               total_intents=total_intents,
                               total_patterns=total_patterns,
                               total_responses=total_responses,
                               total_conversations=total_conversations,
                               unanswered_count=unanswered_count,
                               recent=recent,
                               unanswered=unanswered,
                               daily=daily,
                               top_intents=top_intents,
                               sentiments=sentiments,
                               lang_distribution=lang_distribution)
    except Exception as e:
        logger.error(f"admin_dashboard error: {e}", exc_info=True)
        db.session.rollback()
        flash('Dashboard error. Data may be incomplete.', 'warning')
        return render_template('admin_dashboard.html',
                               total_intents=0, total_patterns=0,
                               total_responses=0, total_conversations=0,
                               unanswered_count=0, recent=[], unanswered=[],
                               daily=[], top_intents=[], sentiments={},
                               lang_distribution={})


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — ADMIN INTENTS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/intents')
@login_required
def admin_intents():
    try:
        intents = Intent.query.order_by(Intent.created_at.desc()).all()
        return render_template('admin_intents.html', intents=intents)
    except Exception as e:
        logger.error(f"admin_intents error: {e}")
        flash('Error loading intents.', 'error')
        return render_template('admin_intents.html', intents=[])


@app.route('/admin/intents/add', methods=['GET', 'POST'])
@login_required
def admin_intent_add():
    form = IntentForm()
    if form.validate_on_submit():
        try:
            intent = Intent(
                tag=form.tag.data.lower().replace(' ', '_'),
                name=form.name.data,
                description=form.description.data,
                is_active=form.is_active.data
            )
            db.session.add(intent)
            db.session.commit()
            get_chatbot().load_intents_from_db()
            flash('Intent added successfully.', 'success')
            return redirect(url_for('admin_intents'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error adding intent: {e}', 'error')
    return render_template('admin_intent_add.html', form=form)


@app.route('/admin/intent/<int:intent_id>/edit', methods=['GET', 'POST'])
@login_required
def admin_intent_edit(intent_id):
    intent = db.session.get(Intent, intent_id)
    if not intent:
        flash('Intent not found', 'error')
        return redirect(url_for('admin_intents'))
    form = IntentForm(obj=intent)
    if form.validate_on_submit():
        try:
            intent.tag         = form.tag.data.lower().replace(' ', '_')
            intent.name        = form.name.data
            intent.description = form.description.data
            intent.is_active   = form.is_active.data
            intent.updated_at  = utc_now()
            db.session.commit()
            get_chatbot().load_intents_from_db()
            flash('Intent updated.', 'success')
            return redirect(url_for('admin_intents'))
        except Exception as e:
            db.session.rollback()
            flash(f'Error updating intent: {e}', 'error')

    patterns  = Pattern.query.filter_by(intent_id=intent_id).all()
    responses = Response.query.filter_by(intent_id=intent_id).all()
    return render_template('admin_intent_edit.html', form=form, intent=intent,
                           patterns=patterns, responses=responses)


# Redirect /admin/intent/<id> → edit
@app.route('/admin/intent/<int:intent_id>', methods=['GET'])
@login_required
def admin_intent_view(intent_id):
    return redirect(url_for('admin_intent_edit', intent_id=intent_id))


@app.route('/admin/intent/<int:intent_id>/delete', methods=['POST'])
@login_required
def admin_intent_delete(intent_id):
    try:
        intent = db.session.get(Intent, intent_id)
        if intent:
            db.session.delete(intent)
            db.session.commit()
            get_chatbot().load_intents_from_db()
            if request.is_json:
                return jsonify({'status': 'success', 'message': 'Intent deleted'})
            flash('Intent deleted.', 'success')
        else:
            if request.is_json:
                return jsonify({'status': 'error', 'message': 'Intent not found'}), 404
            flash('Intent not found.', 'error')
    except Exception as e:
        db.session.rollback()
        if request.is_json:
            return jsonify({'status': 'error', 'message': str(e)}), 500
        flash(f'Delete failed: {e}', 'error')
    return redirect(url_for('admin_intents'))


@app.route('/admin/intents/<int:intent_id>/patterns', methods=['GET', 'POST', 'DELETE'])
@login_required
def admin_intent_patterns(intent_id):
    intent = db.session.get(Intent, intent_id)
    if not intent:
        return jsonify({'error': 'Not found'}), 404

    if request.method == 'GET':
        patterns = [{'id': p.id, 'text': p.pattern_text, 'language': p.language}
                    for p in intent.patterns]
        return jsonify({'patterns': patterns})

    if request.method == 'POST':
        data     = request.get_json() or {}
        text     = (data.get('text', '') or '').strip()
        language = data.get('language', 'en')
        if not text:
            return jsonify({'error': 'Pattern text required'}), 400
        try:
            p = Pattern(intent_id=intent_id, pattern_text=text, language=language)
            db.session.add(p)
            db.session.commit()
            get_chatbot().load_intents_from_db()
            return jsonify({'success': True, 'id': p.id})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    if request.method == 'DELETE':
        data   = request.get_json() or {}
        pat_id = data.get('id')
        try:
            p = db.session.get(Pattern, pat_id)
            if p and p.intent_id == intent_id:
                db.session.delete(p)
                db.session.commit()
                get_chatbot().load_intents_from_db()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
        return jsonify({'success': True})


@app.route('/admin/intents/<int:intent_id>/responses', methods=['GET', 'POST', 'DELETE'])
@login_required
def admin_intent_responses(intent_id):
    intent = db.session.get(Intent, intent_id)
    if not intent:
        return jsonify({'error': 'Not found'}), 404

    if request.method == 'GET':
        responses = [{'id': r.id, 'text': r.response_text, 'language': r.language}
                     for r in intent.responses]
        return jsonify({'responses': responses})

    if request.method == 'POST':
        data     = request.get_json() or {}
        text     = (data.get('text', '') or '').strip()
        language = data.get('language', 'en')
        if not text:
            return jsonify({'error': 'Response text required'}), 400
        try:
            r = Response(intent_id=intent_id, response_text=text, language=language)
            db.session.add(r)
            db.session.commit()
            return jsonify({'success': True, 'id': r.id})
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

    if request.method == 'DELETE':
        data   = request.get_json() or {}
        res_id = data.get('id')
        try:
            r = db.session.get(Response, res_id)
            if r and r.intent_id == intent_id:
                db.session.delete(r)
                db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500
        return jsonify({'success': True})


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — ANALYTICS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/analytics')
@login_required
def admin_analytics():
    try:
        days = int(request.args.get('days', 30))
        cutoff = utc_now() - timedelta(days=days) if days < 365 else None

        q = Conversation.query
        if cutoff:
            q = q.filter(Conversation.timestamp >= cutoff)

        total     = q.count()
        pos       = q.filter_by(sentiment='positive').count()
        neg       = q.filter_by(sentiment='negative').count()
        neutral   = q.filter_by(sentiment='neutral').count()
        unanswered_count = Unanswered.query.filter_by(resolved=False).count()

        lang_stats = {}
        for lang, name in [('en', 'English'), ('ta', 'Tamil'), ('si', 'Sinhala')]:
            cnt = q.filter_by(language=lang).count()
            lang_stats[name] = cnt

        # FIX: Use func.date() instead of cast()
        daily_raw = db.session.query(
            func.date(Conversation.timestamp).label('date'),
            func.count(Conversation.id).label('count')
        )
        if cutoff:
            daily_raw = daily_raw.filter(Conversation.timestamp >= cutoff)
        daily_raw = daily_raw.group_by(func.date(Conversation.timestamp))\
                              .order_by(func.date(Conversation.timestamp)).all()
        daily = [{'date': str(r.date), 'count': r.count} for r in daily_raw]

        top_intents_raw = db.session.query(
            Conversation.intent, func.count(Conversation.id).label('count')
        ).filter(Conversation.intent.isnot(None))
        if cutoff:
            top_intents_raw = top_intents_raw.filter(Conversation.timestamp >= cutoff)
        top_intents_raw = top_intents_raw.group_by(Conversation.intent)\
                                          .order_by(func.count(Conversation.id).desc()).limit(10).all()
        top_intents = [{'intent': r.intent or 'unknown', 'count': r.count}
                       for r in top_intents_raw]

        avg_confidence = db.session.query(
            func.avg(Conversation.confidence)
        ).scalar() or 0.0

        satisfaction_rate = 0.0
        total_feedback = Conversation.query.filter(Conversation.feedback.isnot(None)).count()
        if total_feedback > 0:
            positive_feedback = Conversation.query.filter_by(feedback=1).count()
            satisfaction_rate = round((positive_feedback / total_feedback) * 100, 1)

        return render_template('admin_analytics.html',
                               total=total, pos=pos, neg=neg, neutral=neutral,
                               unanswered=unanswered_count,
                               lang_stats=lang_stats,
                               daily=daily,
                               top_intents=top_intents,
                               days=days,
                               avg_confidence=round(avg_confidence * 100, 1),
                               satisfaction_rate=satisfaction_rate)
    except Exception as e:
        logger.error(f"admin_analytics error: {e}", exc_info=True)
        db.session.rollback()
        return render_template('admin_analytics.html',
                               total=0, pos=0, neg=0, neutral=0,
                               unanswered=0, lang_stats={}, daily=[],
                               top_intents=[], days=30, avg_confidence=0,
                               satisfaction_rate=0)


@app.route('/admin/analytics/export', methods=['POST'])
@login_required
def admin_analytics_export():
    try:
        data = request.get_json() or {}
        days = int(data.get('days', 30))
        cutoff = utc_now() - timedelta(days=days) if days < 365 else None

        q = Conversation.query
        if cutoff:
            q = q.filter(Conversation.timestamp >= cutoff)
        convs = q.order_by(Conversation.timestamp.desc()).all()

        out = io.StringIO()
        w   = csv.writer(out)
        w.writerow(['date', 'user_id', 'user_name', 'message', 'response',
                    'intent', 'language', 'confidence', 'sentiment', 'feedback'])
        for c in convs:
            w.writerow([
                c.timestamp.strftime('%Y-%m-%d') if c.timestamp else '',
                c.user_id or '', c.user_name or '',
                c.message or '', c.response or '',
                c.intent or '', c.language or 'en',
                round(c.confidence or 0, 4), c.sentiment or '',
                c.feedback if c.feedback is not None else ''
            ])
        out.seek(0)
        return FlaskResponse(out.getvalue(), mimetype='text/csv',
                             headers={'Content-Disposition':
                                      f'attachment;filename=analytics_{days}days.csv'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — USER CONVERSATIONS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/user-conversations')
@login_required
def admin_user_conversations():
    try:
        users_data = db.session.query(
            Conversation.user_id,
            Conversation.user_name,
            Conversation.phone_number,
            func.count(Conversation.id).label('message_count'),
            func.max(Conversation.timestamp).label('last_seen')
        ).group_by(Conversation.user_id, Conversation.user_name, Conversation.phone_number)\
         .order_by(func.max(Conversation.timestamp).desc()).all()

        total_messages = Conversation.query.count()
        now_ts = time.time()

        users = []
        for u in users_data:
            uid       = u.user_id or str(uuid.uuid4())
            last_ts   = u.last_seen.timestamp() if u.last_seen else 0
            is_online = (now_ts - last_ts) < 300  # online if active in last 5 min

            # Generate consistent avatar color from user_id
            h = int(_hashlib.md5((uid or 'anon').encode()).hexdigest()[:4], 16) % 360

            users.append({
                'id':            uid,
                'name':          u.user_name or 'Anonymous',
                'phone':         u.phone_number or '-',
                'message_count': u.message_count or 0,
                'last_seen_ts':  last_ts,
                'last_seen':     u.last_seen.strftime('%d %b %H:%M') if u.last_seen else 'Never',
                'is_online':     is_online,
                'avatar_color':  f"hsl({h}, 65%, 52%)",
            })

        return render_template('admin_user_conversations.html',
                               users=users,
                               total_messages=total_messages)
    except Exception as e:
        logger.error(f"admin_user_conversations error: {e}", exc_info=True)
        db.session.rollback()
        return render_template('admin_user_conversations.html',
                               users=[], total_messages=0)


@app.route('/admin/users/<path:user_id>/messages')
@login_required
def admin_user_messages(user_id):
    try:
        convs = (Conversation.query
                 .filter_by(user_id=user_id)
                 .order_by(Conversation.timestamp.asc())
                 .all())
        messages = []
        for c in convs:
            ts = c.timestamp.timestamp() if c.timestamp else None
            if c.message:
                messages.append({'role': 'user', 'content': c.message,
                                 'timestamp': ts, 'language': c.language or 'en'})
            if c.response:
                messages.append({'role': 'bot', 'content': c.response,
                                 'timestamp': ts, 'confidence': c.confidence or 0,
                                 'language': c.language or 'en'})
        return jsonify({'messages': messages})
    except Exception as e:
        logger.error(f"admin_user_messages error: {e}")
        return jsonify({'messages': [], 'error': str(e)}), 500


@app.route('/admin/users/<path:user_id>/delete_conversation', methods=['POST'])
@login_required
def admin_delete_conversation(user_id):
    try:
        count = Conversation.query.filter_by(user_id=user_id).delete()
        Unanswered.query.filter_by(user_id=user_id).delete()
        db.session.commit()
        flash(f'Deleted {count} message(s) for user.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Delete failed: {e}', 'error')
    return redirect(url_for('admin_user_conversations'))


@app.route('/admin/users/<path:user_id>/download')
@login_required
def admin_download_conversation(user_id):
    try:
        convs = (Conversation.query
                 .filter_by(user_id=user_id)
                 .order_by(Conversation.timestamp.asc())
                 .all())
        out = io.StringIO()
        w   = csv.writer(out)
        w.writerow(['timestamp', 'role', 'content', 'language', 'confidence', 'intent'])
        for c in convs:
            ts = c.timestamp.isoformat() if c.timestamp else ''
            if c.message:
                w.writerow([ts, 'user', c.message, c.language or 'en', '', ''])
            if c.response:
                w.writerow([ts, 'bot', c.response, c.language or 'en',
                            round(c.confidence or 0, 4), c.intent or ''])
        out.seek(0)
        safe_uid = re.sub(r'[^a-zA-Z0-9_-]', '_', user_id)[:40]
        return FlaskResponse(out.getvalue(), mimetype='text/csv',
                             headers={'Content-Disposition':
                                      f'attachment;filename=conversation_{safe_uid}.csv'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — UNANSWERED
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/unanswered')
@login_required
def admin_unanswered():
    try:
        items = (Unanswered.query
                 .filter_by(resolved=False)
                 .order_by(Unanswered.asked_count.desc())
                 .limit(200).all())
        return jsonify([{
            'id':         u.id,
            'question':   u.question,
            'count':      u.asked_count,
            'language':   u.language or 'en',
            'last_asked': u.last_asked.isoformat() if u.last_asked else None
        } for u in (items or [])])
    except Exception as e:
        logger.error(f"admin_unanswered error: {e}")
        return jsonify([])


@app.route('/admin/unanswered/<int:qid>/resolve', methods=['POST'])
@login_required
def resolve_unanswered(qid):
    try:
        q = db.session.get(Unanswered, qid)
        if q:
            q.resolved = True
            db.session.commit()
            if request.is_json:
                return jsonify({'success': True})
            flash('Question marked as resolved.', 'success')
        else:
            if request.is_json:
                return jsonify({'error': 'Not found'}), 404
            flash('Question not found.', 'error')
    except Exception as e:
        db.session.rollback()
        if request.is_json:
            return jsonify({'error': str(e)}), 500
        flash(f'Resolve failed: {e}', 'error')
    return redirect(url_for('admin_dashboard'))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — CONVERSATIONS EXPORT / CLEAR
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/conversations/export', methods=['POST'])
@login_required
def export_conversations():
    try:
        convs = Conversation.query.order_by(Conversation.timestamp.desc()).all()
        out   = io.StringIO()
        w     = csv.writer(out)
        w.writerow(['timestamp', 'user_id', 'user_name', 'phone', 'message', 'response',
                    'intent', 'confidence', 'language', 'sentiment', 'feedback'])
        for c in convs:
            w.writerow([
                c.timestamp.isoformat() if c.timestamp else '',
                c.user_id or '', c.user_name or '', c.phone_number or '',
                c.message or '', c.response or '',
                c.intent or '', round(c.confidence or 0, 4),
                c.language or 'en', c.sentiment or '',
                c.feedback if c.feedback is not None else ''
            ])
        out.seek(0)
        return FlaskResponse(out.getvalue(), mimetype='text/csv',
                             headers={'Content-Disposition':
                                      'attachment;filename=conversations.csv'})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/admin/clear-history', methods=['POST'])
@login_required
def admin_clear_history():
    try:
        deleted = Conversation.query.count()
        Conversation.query.delete()
        Unanswered.query.delete()
        db.session.commit()
        return jsonify({'status': 'success', 'deleted': deleted})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500


@app.route('/admin/export-data')
@login_required
def admin_export_data():
    try:
        convs = [{
            'id':              c.id,
            'user_id':         c.user_id,
            'user_name':       c.user_name,
            'phone_number':    c.phone_number,
            'message':         c.message,
            'response':        c.response,
            'intent':          c.intent,
            'confidence':      c.confidence,
            'language':        c.language,
            'sentiment':       c.sentiment,
            'sentiment_score': c.sentiment_score,
            'feedback':        c.feedback,
            'timestamp':       c.timestamp.isoformat() if c.timestamp else None,
        } for c in Conversation.query.order_by(Conversation.timestamp.desc()).all()]

        intents_data = [{
            'tag':       i.tag,
            'name':      i.name,
            'is_active': i.is_active,
            'patterns':  [{'text': p.pattern_text, 'lang': p.language} for p in i.patterns],
            'responses': [{'text': r.response_text, 'lang': r.language} for r in i.responses],
        } for i in Intent.query.all()]

        unanswered_data = [{
            'question':    u.question,
            'language':    u.language,
            'asked_count': u.asked_count,
            'resolved':    u.resolved,
            'last_asked':  u.last_asked.isoformat() if u.last_asked else None,
        } for u in Unanswered.query.all()]

        payload = {
            'exported_at':   utc_now().isoformat(),
            'conversations': convs,
            'intents':       intents_data,
            'unanswered':    unanswered_data,
            'settings':      AppSettings.get('chatbot_config', {}),
        }
        return FlaskResponse(
            json.dumps(payload, ensure_ascii=False, indent=2),
            mimetype='application/json',
            headers={'Content-Disposition': 'attachment;filename=icst_export.json'}
        )
    except Exception as e:
        return jsonify({'error': str(e)}), 500


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — SETTINGS
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/admin/settings', methods=['GET', 'POST'])
@login_required
def admin_settings():
    if request.method == 'POST':
        data = request.get_json() or request.form.to_dict()
        try:
            if 'full_name' in data or 'email' in data:
                if 'full_name' in data:
                    current_user.full_name = str(data['full_name'])[:200]
                if 'email' in data:
                    current_user.email = str(data['email'])[:200]
                db.session.commit()
                if request.is_json:
                    return jsonify({'success': True, 'message': 'Profile updated.'})

            if 'current_password' in data or 'new_password' in data:
                current_pw = data.get('current_password', '')
                new_pw     = data.get('new_password', '')
                if not check_password_hash(current_user.password_hash, current_pw):
                    return jsonify({'error': 'Current password is incorrect.'}), 400
                if len(new_pw) < 8:
                    return jsonify({'error': 'New password must be at least 8 characters.'}), 400
                current_user.password_hash = generate_password_hash(new_pw)
                db.session.commit()
                if request.is_json:
                    return jsonify({'success': True, 'message': 'Password changed.'})

            existing = AppSettings.get('chatbot_config', {}) or {}
            existing.update(data)
            AppSettings.set('chatbot_config', existing)

            if request.is_json:
                return jsonify({'success': True})
            flash('Settings saved.', 'success')
        except Exception as e:
            db.session.rollback()
            logger.error(f"admin_settings POST error: {e}")
            if request.is_json:
                return jsonify({'error': str(e)}), 500
            flash(f'Settings error: {e}', 'error')
        return redirect(url_for('admin_settings'))

    config     = AppSettings.get('chatbot_config', {}) or {}
    ns         = AppSettings.get('notification_settings', {}) or {}
    _full_name = getattr(current_user, 'full_name', None) or current_user.username
    _email     = getattr(current_user, 'email', None) or ''
    _role      = getattr(current_user, 'role', None) or 'admin'

    return render_template(
        'settings.html',
        config=config,
        chatbot_settings=config,
        notification_settings=ns,
        admin_username=current_user.username,
        admin_full_name=_full_name,
        admin_email=_email,
        admin_role=_role,
        now=utc_now(),
    )


@app.route('/admin/reload-intents', methods=['POST'])
@login_required
def admin_reload_intents():
    get_dataset_loader().reload()
    
    # Rebuild in-memory pattern matching for the chatbot singleton
    bot = get_chatbot()
    bot.faq_en = bot._build_faq_en()
    bot.faq_ta = bot._build_faq_ta()
    bot.faq_si = bot._build_faq_si()
    bot._rebuild_flat_patterns()
    bot.load_intents_from_db()
    return jsonify({'success': True, 'message': 'Intents reloaded from database.'})


@app.route('/admin/reload', methods=['POST'])
@login_required
def admin_reload():
    return admin_reload_intents()


@app.route('/admin/import-faq', methods=['POST'])
@login_required
def admin_import_faq():
    count = 0
    bot   = get_chatbot()
    try:
        for faq_dict, lang in [(bot.faq_en, 'en'), (bot.faq_ta, 'ta'), (bot.faq_si, 'si')]:
            for key, faq in faq_dict.items():
                tag = f"{key}_{lang}" if lang != 'en' else key
                if Intent.query.filter_by(tag=tag).first():
                    continue
                intent = Intent(
                    tag=tag,
                    name=f"{key.replace('_', ' ').title()} ({lang.upper()})",
                    description=f"FAQ {lang}: {faq['patterns'][0][:50]}",
                    is_active=True
                )
                db.session.add(intent)
                db.session.flush()
                for p in faq['patterns']:
                    db.session.add(Pattern(intent_id=intent.id,
                                           pattern_text=p, language=lang))
                db.session.add(Response(intent_id=intent.id,
                                        response_text=faq['answer'], language=lang))
                count += 1
        db.session.commit()
        bot.load_intents_from_db()
        flash(f'Imported {count} FAQ items to database.', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Import error: {e}', 'error')
    return redirect(url_for('admin_intents'))


# ═════════════════════════════════════════════════════════════════════════════
# ROUTES — LANGUAGE DETECTION API
# ═════════════════════════════════════════════════════════════════════════════

@app.route('/api/detect-language', methods=['POST'])
@csrf.exempt
def api_detect_language():
    data = request.get_json() or {}
    text = data.get('text', '')
    return jsonify({'language': detect_language(text)})


# ═════════════════════════════════════════════════════════════════════════════
# ERROR HANDLERS
# ═════════════════════════════════════════════════════════════════════════════

@app.errorhandler(404)
def not_found(e):
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Not found'}), 404
    return render_template('404.html'), 404


@app.errorhandler(500)
def internal_error(e):
    db.session.rollback()
    if request.is_json or request.path.startswith('/api/'):
        return jsonify({'error': 'Internal server error'}), 500
    return render_template('500.html'), 500


@app.errorhandler(403)
def forbidden(e):
    if request.is_json:
        return jsonify({'error': 'Forbidden'}), 403
    return redirect(url_for('admin_login'))


@app.errorhandler(413)
def too_large(e):
    return jsonify({'error': 'File too large — max 16 MB'}), 413


@app.errorhandler(429)
def rate_limited(e):
    return jsonify({'error': 'Too many requests — please slow down.'}), 429

@app.route('/test/sinhala-direct', methods=['GET'])
def test_sinhala_direct():
    """Test direct gTTS with si language"""
    test_text = "ආයුබෝවන්"
    try:
        tts = gTTS(text=test_text, lang='si', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        audio_base64 = base64.b64encode(audio_buffer.read()).decode('utf-8')
        
        return f'''
        <html>
        <body>
            <h2>✅ Sinhala TTS Working!</h2>
            <p>Text: {test_text}</p>
            <audio controls autoplay>
                <source src="data:audio/mpeg;base64,{audio_base64}" type="audio/mpeg">
            </audio>
            <br><br>
            <a href="/">Back to Chat</a>
        </body>
        </html>
        '''
    except Exception as e:
        return f"Error: {e}"


# ═════════════════════════════════════════════════════════════════════════════
# DATABASE INIT
# ═════════════════════════════════════════════════════════════════════════════

def init_database():
    with app.app_context():
        db.create_all()

        if not User.query.filter_by(role='admin').first():
            _pw = os.getenv('ADMIN_DEFAULT_PASSWORD', 'ChangeMe123!')
            db.session.add(User(
                username=os.getenv('ADMIN_DEFAULT_USERNAME', 'ICST'),
                password_hash=generate_password_hash(_pw),
                email='admin@icst.edu.lk',
                role='admin'
            ))
            db.session.commit()
            logger.info("Default admin created — change password immediately!")

        if Intent.query.count() == 0:
            for item in [
                {'tag': 'greeting', 'name': 'Greetings',
                 'patterns_en': ['hi', 'hello', 'hey', 'good morning'],
                 'responses_en': ['Hello! Welcome to ICST University. How can I help? 😊']},
                {'tag': 'admission', 'name': 'Admission',
                 'patterns_en': ['how to apply', 'admission process', 'enroll'],
                 'responses_en': ['Apply online at www.icst.edu.lk or call 0743 444 444']},
                {'tag': 'courses', 'name': 'Courses',
                 'patterns_en': ['what courses', 'programs offered'],
                 'responses_en': ['We offer BSc CS, Cybersecurity, HDIT, and more!']},
                {'tag': 'fees', 'name': 'Fees',
                 'patterns_en': ['fees', 'how much', 'cost'],
                 'responses_en': ['Flexible instalment plans available. Call 0743 444 444 for details.']},
            ]:
                intent = Intent(tag=item['tag'], name=item['name'])
                db.session.add(intent)
                db.session.flush()
                for p in item['patterns_en']:
                    db.session.add(Pattern(intent_id=intent.id, pattern_text=p, language='en'))
                for r in item['responses_en']:
                    db.session.add(Response(intent_id=intent.id, response_text=r, language='en'))
            db.session.commit()
            logger.info("Default intents seeded")

        get_chatbot().load_intents_from_db()
        logger.info("Multilingual chatbot ready (no-translation engine)")


_db_initialised = False

@app.before_request
def _lazy_init():
    global _db_initialised
    if not _db_initialised:
        _db_initialised = True
        try:
            init_database()
        except Exception as e:
            logger.error(f"DB init error: {e}")


# ═════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    print("\n" + "═" * 62)
    print("  ICST UNIVERSITY CHATBOT")
    print("  Engine : Direct multilingual matching")
    print("  Langs  : English | Tamil | Sinhala (direct, native)")
    print("═" * 62)
    print(f"  RapidFuzz : {'✓ (fast matching)' if RAPIDFUZZ else '✗  pip install rapidfuzz'}")
    print(f"  spaCy NER : {'✓' if nlp else '✗ (optional)'}")
    print("═" * 62)

    init_database()

    print("  🌐  Chat  : http://127.0.0.1:5000")
    print("  👤  Admin : http://127.0.0.1:5000/admin/login")
    print("  ❤️   Health: http://127.0.0.1:5000/health")
    print("─" * 62 + "\n")

    app.run(
        debug=os.getenv('FLASK_DEBUG', 'false').lower() in ('true', '1'),
        host='0.0.0.0',
        port=int(os.getenv('PORT', 5000))
    )