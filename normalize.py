from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MyDataset

train_dataset_2 = MyDataset(
    dataset_dir="/home/jackson/Downloads/conveyor_video_for_train-20231122T125805Z-001/dataset",
    phase="train",
)

train_loader_2 = DataLoader(
    train_dataset_2,
    batch_size=4,
    shuffle=True,
    drop_last=True,
)

mean = 0.0
std = 0.0
total_images = 0

for inputs, _ in tqdm(train_loader_2):
    batch_size = inputs.size(0)
    inputs = inputs.view(batch_size, inputs.size(1), -1)
    mean += inputs.mean(2).sum(0)
    std += inputs.std(2).sum(0)
    total_images += batch_size

mean /= total_images
std /= total_images

print("Mean: ", mean)
print("Std: ", std)
