import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay,roc_curve,auc
import config

def plot_loss(train_loss, val_loss):
    plt.figure(figsize=(8, 5))
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(config.LOSS_CURVE_PATH)
    plt.show()

def plot_accuracy(accuracy):
    plt.figure(figsize=(8, 5))
    plt.plot(accuracy)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Validation Accuracy")
    plt.grid(True)
    plt.savefig(config.ACCURACY_CURVE_PATH)
    plt.show()

def plot_confusion_matrix(labels, predictions):
    ConfusionMatrixDisplay.from_predictions(labels,predictions)
    plt.title("Confusion Matrix")
    plt.savefig(config.CONFUSION_MATRIX_PATH)
    plt.show()


def plot_roc_curve(labels, probabilities):
    fpr, tpr, _ = roc_curve(labels,probabilities)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr,tpr,label=f"AUC = {roc_auc:.4f}",linewidth=2)
    plt.plot([0, 1],[0, 1],linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(config.ROC_CURVE_PATH)
    plt.show()