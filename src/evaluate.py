import torch
import config
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,roc_auc_score

def evaluate_model(model,test_loader,loss_function,device):

    model.eval()

    running_loss = 0

    all_probabilities = []
    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for batch_features,batch_labels in test_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            loss = loss_function(logits,batch_labels)

            running_loss += loss.item()

            probabilities = torch.sigmoid(logits)
            predictions = (probabilities > config.THRESHOLD).float()

            all_probabilities.extend(probabilities.cpu().numpy().flatten())
            all_predictions.extend(predictions.cpu().numpy().flatten())
            all_labels.extend(batch_labels.cpu().numpy().flatten())
        
        val_loss = running_loss / len(test_loader)

    accuracy = accuracy_score(all_labels,all_predictions)
    precision = precision_score(all_labels,all_predictions)
    recall = recall_score(all_labels,all_predictions)
    f1 = f1_score(all_labels,all_predictions)
    roc_auc = roc_auc_score(all_labels,all_probabilities)

    return (
        val_loss,
        accuracy,
        precision,
        recall,
        f1,
        roc_auc,
        all_labels,
        all_predictions,
        all_probabilities
    )
