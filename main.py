from src.preprocessing import prepare_data
import torch
import numpy as np
import config
from src.dataset import CustomDataset
from torch.utils.data import DataLoader
from src.model import LoanApprovalModel
from src.train import train_model
import torch.nn as nn
torch.optim.lr_scheduler.ReduceLROnPlateau

torch.manual_seed(config.RANDOM_SEED)
np.random.seed(config.RANDOM_SEED)

device = config.DEVICE
print(f"Using device: {device}")

X_train, X_test, y_train, y_test, input_features,pos_weight = prepare_data(config.DATA_PATH,device)

train_dataset = CustomDataset(X_train,y_train)
test_dataset = CustomDataset(X_test,y_test)

train_loader = DataLoader(train_dataset,batch_size=config.BATCH_SIZE,shuffle=True)
test_loader = DataLoader(test_dataset,batch_size=config.BATCH_SIZE,shuffle=False)

model = LoanApprovalModel(X_train.shape[1])
model.to(device)

loss_function = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
optimiser = torch.optim.Adam(model.parameters(),lr=config.LEARNING_RATE)

train_model(model,train_loader,test_loader,loss_function,optimiser,config.EPOCHS,device)