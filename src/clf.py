from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from dataset import MyDataset

train_dataset = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
    phase="train",
)
val_dataset = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
    phase="valid",
)
test_dataset = MyDataset(
    dataset_dir="/home/jackson/Desktop/ARS_VN/mini_project/ARS-2",
    phase="test",
)
X_train, y_train = train_dataset()
X_val, y_val = val_dataset()
X_test, y_test = test_dataset()

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
# Normalize the features
X_test = scaler.transform(X_test)


svc = SVC()
# Define the parameter grid to search
param_grid = {
    "C": [0.1, 1],
    "kernel": ["linear", "rbf", "poly"],
    "gamma": ["scale", "auto"],
    "probability": [True, False],
    "class_weight": [None, "balanced"],
}

# Create a GridSearchCV object
grid_search = GridSearchCV(svc, param_grid, cv=5, scoring="accuracy", verbose=1)

# Fit the model to the data
grid_search.fit(X_train, y_train)

# Print the best parameters and corresponding accuracy
print("Best Parameters: ", grid_search.best_params_)
print("Best Accuracy: ", grid_search.best_score_)

# Evaluate the model on the test set
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print("Test Accuracy: ", test_accuracy)
