import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

def load_and_preprocess(data_path):
    df = pd.read_csv(data_path)
    df.fillna({
        'Monthly_Inhand_Salary': df['Monthly_Inhand_Salary'].median() if 'Monthly_Inhand_Salary' in df else 0,
        'Monthly_Balance': df['Monthly_Balance'].median() if 'Monthly_Balance' in df else 0,
        'Occupation': 'Unknown',
        'Credit_Score': 'Unknown'
    }, inplace=True)

    if 'Customer_ID' in df.columns:
        df = df.drop(columns=['Customer_ID'])

    y = df['Credit_Score']
    X = df.drop(columns=['Credit_Score'])

    numeric_features = [c for c in X.columns if str(X[c].dtype) != 'object']
    categorical_features = [c for c in X.columns if str(X[c].dtype) == 'object']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y if y.nunique() > 1 else None
    )
    return X_train, X_test, y_train, y_test, preprocessor
