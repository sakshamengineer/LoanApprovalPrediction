from src.evaluate import evaluate_model
import config
import torch
from src.utils import *

def train_model(model,train_loader,test_loader,loss_function,optimiser,epochs,device):

    train_loss_history = []
    val_loss_history = []
    accuracy_history = []

    best_val_loss = float("inf")
    patience = config.PATIENCE
    counter = 0

    for epoch in range(epochs):

        model.train()

        running_train_loss = 0

        for batch_features,batch_labels in train_loader:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

            logits = model(batch_features)
            train_loss = loss_function(logits,batch_labels)
            optimiser.zero_grad()
            train_loss.backward()
            optimiser.step()
            
            running_train_loss += train_loss.item()
        
        epoch_train_loss = running_train_loss / len(train_loader)
        train_loss_history.append(epoch_train_loss)

        val_loss, accuracy, precision, recall, f1, roc_auc, all_labels, all_predictions, all_probablities = evaluate_model(model, test_loader,loss_function,device)

        val_loss_history.append(val_loss)
        accuracy_history.append(accuracy)

        print(
            f"Epoch [{epoch+1}/{epochs}] "
            f"| Train Loss: {epoch_train_loss:.4f} "
            f"| Val Loss: {val_loss:.4f} "
            f"| Accuracy: {accuracy:.4f} "
            f"| Precision: {precision:.4f} "
            f"| Recall: {recall:.4f} "
            f"| F1: {f1:.4f} "
            f"| ROC-AUC: {roc_auc:.4f}"
        )

        if val_loss < best_val_loss :
            best_val_loss = val_loss
            torch.save(model.state_dict(),config.MODEL_PATH)
            counter = 0
        else:
            counter += 1
        
        if counter >= patience:
            print("\nEarly Stopping Triggered.")
            break

    plot_loss(train_loss_history,val_loss_history)
    plot_accuracy(accuracy_history)
    plot_confusion_matrix(all_labels,all_predictions)
    plot_roc_curve(all_labels,all_probablities)    
        
    return (
        train_loss_history,
        val_loss_history,
        accuracy_history,
    )