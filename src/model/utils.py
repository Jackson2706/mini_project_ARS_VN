import time

import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)


def train(
    model, epochs, device, train_loader, val_loader, criterion, optimizer
):
    best_accuracy = 0.0
    best_model_weights = model.state_dict()
    model = model.to(device)
    for epoch in range(epochs):
        model.train()
        print("*" * 50)
        print(f"Epoch {epoch+1}:")
        start_ep = time.time()
        losses = []
        for idx, (image, label) in enumerate(train_loader):
            start = time.time()
            image, label = image.to(device), label.to(device)
            optimizer.zero_grad()
            output = model(image)
            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            end = time.time()
            print(
                f"\t iteration {idx}/{len(train_loader)}:\t Loss: {loss:.7f}\t Time: {(end-start):.7f}"
            )
        losses = sum(losses) / len(losses)

        print("**Validation**")
        val_losses = []
        acc_list = []
        recall_list = []
        precision_list = []
        f1_score_list = []
        model.eval()
        for idx, (image, label) in enumerate(val_loader):
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
            print(
                f"\t Iteration {idx+1}/{len(val_loader)}: \t accuracy: {acc:.7f}\t precision: {precision:.7f}\t recall: {recall:.7f}\t f1: {f1:.7f}"
            )
        accuracy = sum(acc_list) / len(acc_list)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_score_list) / len(f1_score_list)

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model_weights = model.state_dict()
        end_ep = time.time()
        print(
            f"Espoch {epoch + 1}/{epochs}: \ttrain_loss: {losses:.7f} \t time: {(end_ep-start_ep):.7f}\n accuracy: {accuracy:.7f}\t precision {precision:.7f}\t recall: {recall:.7f}\t f1 score: {f1:.7f}"
        )
    return model.state_dict(), best_model_weights


def test(model, device, test_loader):
    with torch.no_grad():
        print("*" * 50)
        acc_list = []
        recall_list = []
        precision_list = []
        f1_score_list = []

        for idx, (image, label) in enumerate(test_loader):
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
            print(
                f"Iteration {idx+1}/{len(test_loader)}: \t accuracy: {acc:.7f}\t precision: {precision:.7f}\t recall: {recall:.7f}\t f1: {f1:.7f}"
            )
        print("*" * 25)
        accuracy = sum(acc_list) / len(acc_list)
        precision = sum(precision_list) / len(precision_list)
        recall = sum(recall_list) / len(recall_list)
        f1 = sum(f1_score_list) / len(f1_score_list)
        print(
            f"Test phase: \t accuracy: {accuracy:.7f}\t precision {precision:.7f}\t recall: {recall:.7f}\t f1 score: {f1:.7f}"
        )
