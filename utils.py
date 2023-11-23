import time

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm


def train(
    model,
    epochs,
    device,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    notification: bool = False,
):
    best_accuracy = 0.0
    best_model_weights = model.state_dict()
    model = model.to(device)
    train_loss_list = []
    val_loss_list = []
    for epoch in range(epochs):
        model.train()
        print("*" * 50)
        print(f"Epoch {epoch+1}:")
        start_ep = time.time()
        train_losses = []
        idx = 0
        for image, label in tqdm(train_loader):
            start = time.time()
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_losses.append(loss.item())
            end = time.time()
            if notification:
                print(
                    f"\t iteration {idx}/{len(train_loader)}:\t Loss: {loss:.7f}\t Time: {(end-start):.7f}"
                )
            idx = idx + 1
        train_losses = sum(train_losses) / len(train_losses)
        train_loss_list.append(train_losses)
        print("**Validation**")
        val_losses = []
        acc_list = []
        recall_list = []
        precision_list = []
        f1_score_list = []
        model.eval()
        idx = 0
        for image, label in tqdm(val_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)
            loss = criterion(output, label)
            val_losses.append(loss.item())

            _, predict = torch.max(output, 1)
            predict_cpu = predict.cpu()
            label_cpu = label.cpu()

            acc = accuracy_score(label_cpu, predict_cpu)
            precision = precision_score(
                label_cpu, predict_cpu, average="weighted"
            )
            recall = recall_score(label_cpu, predict_cpu, average="weighted")
            f1 = f1_score(label_cpu, predict_cpu, average="weighted")
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1)

            if notification:
                print(
                    f"\t Iteration {idx+1}/{len(val_loader)}: \t accuracy: {acc:.7f}\t precision: {precision:.7f}\t recall: {recall:.7f}\t f1: {f1:.7f}"
                )
            idx = idx + 1
        val_losses = sum(val_losses) / len(val_losses)
        val_loss_list.append(val_losses)
        accuracy = sum(acc_list) / len(acc_list)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_score_list) / len(f1_score_list)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict()
        end_ep = time.time()
        if notification:
            print(
                f"Espoch {epoch + 1}/{epochs}: \ttrain_loss: {train_losses:.7f} \t time: {(end_ep-start_ep):.7f}\n accuracy: {accuracy:.7f}\t precision {precision:.7f}\t recall: {recall:.7f}\t f1 score: {f1:.7f}"
            )
    return (
        model.state_dict(),
        best_model_weights,
        train_loss_list,
        val_loss_list,
    )


def test(model, device, test_loader, notification: bool = False):
    model.eval()
    with torch.no_grad():
        print("*" * 50)
        acc_list = []
        recall_list = []
        precision_list = []
        f1_score_list = []

        idx = 0
        for image, label in tqdm(test_loader):
            image, label = image.to(device), label.to(device)
            output = model(image)

            _, predict = torch.max(output, 1)
            predict_cpu = predict.cpu()
            label_cpu = label.cpu()

            acc = accuracy_score(label_cpu, predict_cpu)
            precision = precision_score(
                label_cpu, predict_cpu, average="weighted"
            )
            recall = recall_score(label_cpu, predict_cpu, average="weighted")
            f1 = f1_score(label_cpu, predict_cpu, average="weighted")
            acc_list.append(acc)
            precision_list.append(precision)
            recall_list.append(recall)
            f1_score_list.append(f1)
            if notification:
                print(
                    f"Iteration {idx+1}/{len(test_loader)}: \t accuracy: {acc:.7f}\t precision: {precision:.7f}\t recall: {recall:.7f}\t f1: {f1:.7f}"
                )
            idx = idx + 1
        print("*" * 25)
        accuracy = sum(acc_list) / len(acc_list)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_score_list) / len(f1_score_list)
        print(
            f"Test phase: \t accuracy: {accuracy:.7f}\t precision {precision:.7f}\t recall: {recall:.7f}\t f1 score: {f1:.7f}"
        )
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
    }


def plot_loss(train_loss, valid_loss, save_path):
    # Tạo một danh sách thể hiện số epoch (vòng lặp huấn luyện)
    epochs = range(1, len(train_loss) + 1)

    # Vẽ đồ thị train loss và validation loss
    plt.plot(epochs, train_loss, "b", label="Train Loss")
    plt.plot(epochs, valid_loss, "r", label="Validation Loss")
    plt.title("Train and Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.xticks(epochs, rotation=45)
    plt.legend()
    plt.savefig(save_path, format="svg", bbox_inches="tight")
