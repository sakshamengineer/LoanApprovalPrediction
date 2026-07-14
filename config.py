import torch

DATA_PATH = "Dataset\loan_data.csv"

MODEL_PATH = "Models/Loan_Approval_Model.pth"
PREPROCESSOR_PATH = "Models/preprocessor.pkl"

LOSS_CURVE_PATH = "Results/loss_curve.png"
ACCURACY_CURVE_PATH = "Results/accuracy_curve.png"
CONFUSION_MATRIX_PATH = "Results/confusion_matrix.png"
ROC_CURVE_PATH = "Results/roc_curve.png"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

RANDOM_SEED = 42

BATCH_SIZE = 512
LEARNING_RATE = 0.001
EPOCHS = 25

THRESHOLD = 0.5
PATIENCE = 5