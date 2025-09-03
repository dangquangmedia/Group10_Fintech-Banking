import pandas as pd

BASIC_FEATURES = [
    'Age','Annual_Income','Monthly_Inhand_Salary',
    'Num_Bank_Accounts','Num_Credit_Card','Interest_Rate','Num_of_Loan',
    'Occupation','Type_of_Loan','Delay_from_due_date'
]

def ensure_all_features(df: pd.DataFrame) -> pd.DataFrame:
    for col in BASIC_FEATURES:
        if col not in df.columns:
            df[col] = 0 if df.select_dtypes(include='number').shape[1] else ''
    return df[BASIC_FEATURES]
