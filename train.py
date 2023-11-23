import torch
import torch.nn as nn
import torch.optim as opt
from torch.utils.data import DataLoader

from dataset import MyDataset
from model import CNN
from utils import plot_loss, test, train

epochs = 5
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Downloads/conveyor_video_for_train-20231122T125805Z-001/dataset",
    phase="train",
)
val_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Downloads/conveyor_video_for_train-20231122T125805Z-001/dataset",
    phase="valid",
)
test_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Downloads/conveyor_video_for_train-20231122T125805Z-001/dataset",
    phase="test",
)
train_loader_2 = DataLoader(
    train_dataset_2,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)
val_loader_2 = DataLoader(val_dataset_2, batch_size=4, shuffle=False)
test_loader_2 = DataLoader(test_dataset_2, batch_size=4, shuffle=False)

CNN_clf = CNN()

criterion = nn.CrossEntropyLoss()
optimizer = opt.Adam(CNN_clf.model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
cnn_last_weight, cnn_best_weight, train_loss_list, val_loss_list = train(
    model=CNN_clf,
    epochs=epochs,
    device=device,
    train_loader=train_loader_2,
    val_loader=val_loader_2,
    criterion=criterion,
    optimizer=optimizer,
    notification=True,
)
torch.save(
    cnn_best_weight,
    "./weights/best_cnn_res18.pt",
)
torch.save(
    cnn_last_weight,
    "./weights/last_cnn_res18.pt",
)

CNN_clf.load_state_dict(torch.load("./weights/best_cnn_res18.pt"))

print("*" * 25 + "Test phase" + "*" * 25)
test(model=CNN_clf, device=device, test_loader=test_loader_2, notification=True)
plot_loss(train_loss_list, val_loss_list, "./plt_loss.svg")
