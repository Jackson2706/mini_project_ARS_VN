import torch
import torch.nn as nn
import torch.optim as opt
from lazypredict.Supervised import LazyClassifier
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader

from dataset import MyDataset
from model.CNN import CNN
from model.MLP import MLP
from model.utils import plot_loss, test, train

# """
# Defining the dataset
# """
# train_dataset = MyDataset(
#     dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
#     phase="train",
# )
# val_dataset = MyDataset(
#     dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
#     phase="valid",
# )
# test_dataset = MyDataset(
#     dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
#     phase="test",
# )
# X_train, y_train = train_dataset()
# X_val, y_val = val_dataset()
# X_test, y_test = test_dataset()
# """
#     (Machine learning model)
# """
# """
#     Preprocessing data
# """
# # Normalization
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# # Normalize the features
# X_test = scaler.transform(X_test)
# """
#     Select model phase
# """
# # Create a list to store all of the models which are chosen to test, using lazypredict
# clf = LazyClassifier(verbose=0, ignore_warnings=True, custom_metric=None)
# models, predictions = clf.fit(X_train, X_test, y_train, y_test)
# print(models)


# """
#     (Deep learning model - MLP
# """
# print("*" * 100)
# epochs = 100
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# mlp_clf = MLP()
# train_loader = DataLoader(
#     train_dataset,
#     batch_size=8,
#     shuffle=True,
#     drop_last=True,
# )
# val_loader = DataLoader(val_dataset, batch_size=8, shuffle=False)
# test_loader = DataLoader(test_dataset, batch_size=8, shuffle=False)

# criterion = nn.CrossEntropyLoss()
# optimizer = opt.Adam(mlp_clf.parameters(), lr=1e-3)


# print("*" * 25 + "Train phase" + "*" * 25)
# last_model, best_model, train_loss_list, val_loss_list = train(
#     model=mlp_clf,
#     epochs=epochs,
#     device=device,
#     train_loader=train_loader,
#     val_loader=val_loader,
#     criterion=criterion,
#     optimizer=optimizer,
# )
# print("*" * 25 + "Test phase" + "*" * 25)

"""
    (Deep learning model - CNN
"""
epochs = 50
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-4",
    phase="train",
    hog=False,
)
val_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-4",
    phase="valid",
    hog=False,
)
test_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-4",
    phase="test",
    hog=False,
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
optimizer = opt.Adam(CNN_clf.parameters(), lr=1e-3, weight_decay=1e-4)
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
    "/home/jackson/Desktop/ARS_VN/mini_project/weights/best_cnn.pt",
)
# torch.save(
#     best_model, "/home/jackson/Desktop/ARS_VN/mini_project/weights/best_mlp.pt"
# )
CNN_clf.load_state_dict(
    torch.load("/home/jackson/Desktop/ARS_VN/mini_project/weights/best_cnn.pt")
)
# mlp_clf.load_state_dict(
#     # torch.load("/home/jackson/Desktop/ARS_VN/mini_project/weights/best_mlp.pt")
# )
print("*" * 25 + "Test phase" + "*" * 25)
test(model=CNN_clf, device=device, test_loader=test_loader_2, notification=True)
# test(model=mlp_clf, device=device, test_loader=test_loader)
plot_loss(train_loss_list, val_loss_list, "./plt_loss.svg")
