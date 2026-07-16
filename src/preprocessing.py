import pandas as pd
from sklearn.preprocessing import OneHotEncoder,OrdinalEncoder,StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
import joblib
import torch
import config

education_order = ["High School","Associate","Bachelor","Master","Doctorate"]

def prepare_data(Data_path,device):
    df = pd.read_csv(Data_path)

    df.drop(df[df['person_age'] > 100].index,inplace=True)
    df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map({"No": 0,"Yes": 1})

    X = df.drop(columns=["loan_status"])
    y = df["loan_status"]

    numerical_columns = X.select_dtypes(include=["float64","int64"]).columns.tolist()
    categorical_columns = X.select_dtypes(include=["object","category"]).columns.to_list()
    columns_to_remove = ["person_education","previous_loan_defaults_on_file"]
    categorical_columns = [col for col in categorical_columns if col not in columns_to_remove]

    X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)

    num_negative = (y_train == 0).sum()
    num_positive = (y_train == 1).sum()

    pos_weight = torch.tensor([num_negative / num_positive],dtype=torch.float32,device=device)

    preprocessor = ColumnTransformer([
        ("ohe",OneHotEncoder(handle_unknown='ignore',sparse_output=False),categorical_columns),
        ("OE",OrdinalEncoder(categories=[education_order]),["person_education"]),
        ("Scaler",StandardScaler(),numerical_columns),
    ])

    X_train = preprocessor.fit_transform(X_train)
    X_test = preprocessor.transform(X_test)
    joblib.dump(preprocessor,config.PREPROCESSOR_PATH)

    return (
        X_train,
        X_test,
        y_train.to_numpy(),
        y_test.to_numpy(),
        X_train.shape[1],
        pos_weight
    )