import joblib
import config
import torch
from src.model import LoanApprovalModel

def predict(customer_dataframe):

    preprocessor = joblib.load(config.PREPROCESSOR_PATH)
    X = preprocessor.transform(customer_dataframe)
    X = torch.tensor(X,dtype=torch.float32,)

    model = LoanApprovalModel(X.shape[1])
    model.load_state_dict(torch.load(config.MODEL_PATH,map_location='cpu'))

    model.eval()
    with torch.no_grad():
        logits = model(X)
        probability = torch.sigmoid(logits).item()

        prediction = (probability > config.THRESHOLD)
    return prediction, probability