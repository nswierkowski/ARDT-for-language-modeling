import numpy as np
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import train_test_split, HalvingRandomSearchCV
from sklearn.metrics import accuracy_score, classification_report
from src.models.ARDTClassifier import ARDTCoTClassifier  
import xgboost as xgb
import lightgbm as lgb

texts = [
    "the quick brown fox jumps",
    "lazy dog sleeps all day",
    "fast fox outruns slow dog",
    "happy cat plays with yarn",
]
labels = [1, 0, 1, 0]

def train_model():
    vocab = set(" ".join(texts).split())
    word2idx = {w: i for i, w in enumerate(vocab)}
    embeddings = np.random.rand(len(word2idx), 50).astype(np.float32)

    X_train, X_test, y_train, y_test = train_test_split(texts, labels, test_size=0.3, random_state=42, stratify=labels)

    model = ARDTCoTClassifier(
        word2idx=word2idx,
        embeddings=embeddings,
        context_size=3,
        alpha=0.5,
        model_type='decision_tree',
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Report:\n", classification_report(y_test, y_pred))

if __name__=='__main__':
    train_model()