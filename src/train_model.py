from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
try:
    from xgboost import XGBClassifier
    has_xgb = True
except Exception:
    has_xgb = False
import joblib
from .data_preprocessing import load_and_preprocess

def train_and_save(data_path='data/raw/credit_data.csv'):
    X_train, X_test, y_train, y_test, pre = load_and_preprocess(data_path)
    models = {
        'Logistic_Regression': LogisticRegression(max_iter=1000),
        'Random_Forest': RandomForestClassifier(n_estimators=300, random_state=42)
    }
    if has_xgb:
        models['XGBoost'] = XGBClassifier(random_state=42, eval_metric='mlogloss')

    results = {}
    for name, clf in models.items():
        pipe = Pipeline([('pre', pre), ('clf', clf)])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = report
        joblib.dump(pipe, f'models/{name.lower()}.pkl')
    return results

if __name__ == '__main__':
    print(train_and_save())
