import joblib
from sklearn.ensemble import RandomForestClassifier


def train_model(X, y, save_path: str = 'models/model_final.pkl'):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, save_path)
    return model
