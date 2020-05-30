import torch.nn as nn
import torch
import torch.optim as optim
from torchvision import datasets, transforms, models

import numpy as np
import os
from sklearn import metrics


def train_loop(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs,
    device,
    save_model=False,
):
    """Main generic training loop. Includes test loss recording
       returns: train losses and test losses"""

    train_losses, test_losses = [], []

    best_loss = np.inf

    for ep in range(epochs):
        train_batch_loss = []
        model.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, targets)

            loss.backward()
            optimizer.step()

            train_batch_loss.append(loss.item())
        train_mean_batch_loss = np.mean(train_batch_loss)

        test_batch_loss = []
        model.eval()
        for inputs, targets in test_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            test_batch_loss.append(loss.item())
        test_mean_batch_loss = np.mean(test_batch_loss)

        train_losses.append(train_mean_batch_loss)
        test_losses.append(test_mean_batch_loss)

        print(
            f"Epoch {ep+1}/{epochs}, Train Loss:{train_mean_batch_loss:.4f}, \
                Test Loss: {test_mean_batch_loss:.4f}"
        )
        if save_model:
            if test_mean_batch_loss < best_loss:
                best_loss = test_mean_batch_loss
                torch.save(model, f"/content/models/{model.__class__.__name__}.pt")
                print(
                    f"saving new best model with Test Loss: {test_mean_batch_loss:.4f}"
                )

    return np.array(train_losses), np.array(test_losses)


def get_transforms():
    """Transformations for train and test images"""

    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(),
            transforms.CenterCrop(size=224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.CenterCrop(size=224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )
    return train_transform, test_transform


def get_dataset(path):
    """Image dataset from path"""

    train_transform, test_transform = get_transforms()
    train_dataset = datasets.ImageFolder(
        os.path.join(path, "train"), transform=train_transform
    )
    test_dataset = datasets.ImageFolder(
        os.path.join(path, "test"), transform=test_transform
    )
    return train_dataset, test_dataset


def get_dataloaders(path, batch_size=64):
    """Image dataset loaders"""

    train_dataset, test_dataset = get_dataset(path)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False
    )
    return train_loader, test_loader


def accuracy(data_loader, model, device):
    """Calculate accuracy"""

    model.eval()
    n_correct = 0
    n_total = 0
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            _, predictions = torch.max(outputs, 1)

            n_correct += (predictions == targets).sum().item()
            n_total += targets.size(0)

    return n_correct / n_total


def get_classification_report(data_loader, model, device):
    """Get a sklearn multivariable classification report"""

    model.eval()
    y_true = []
    y_pred = []
    with torch.no_grad():
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)

            _, predictions = torch.max(outputs, 1)

            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(targets.cpu().numpy())
    return metrics.classification_report(y_true, y_pred)


def define_model(trial):
    """Model for Optuna"""

    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False

    n_features = model.classifier[0].in_features
    n_layers = trial.suggest_int("n_layers", 1, 5)
    layers = []
    in_features = model.classifier[0].in_features
    for i in range(n_layers):
        out_features = trial.suggest_int("n_units_l{}".format(i), 64, 12544)
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        p = trial.suggest_uniform("dropout_l{}".format(i), 0.2, 0.5)
        layers.append(nn.Dropout(p))

        in_features = out_features
    layers.append(nn.Linear(in_features, CLASSES))

    model.classifier = nn.Sequential(*layers)

    return model


def objective(trial):
    """Objective for Optuna"""

    # Generate the model.
    model = define_model(trial).to(DEVICE)

    # Generate the optimizers.
    optimizer_name = trial.suggest_categorical("optimizer", ["Adam", "RMSprop", "SGD"])
    lr = trial.suggest_loguniform("lr", 1e-5, 1e-1)
    optimizer = getattr(optim, optimizer_name)(model.parameters(), lr=lr)

    train_loader, test_loader = get_dataloaders(
        os.path.join("/content", "brand_data"), batch_size=BATCHSIZE
    )
    criterion = nn.CrossEntropyLoss()

    # Training of the model.
    model.train()
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break

            data, target = data.to(DEVICE), target.to(DEVICE)

            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)

        trial.report(accuracy, epoch)

        # Handle pruning based on the intermediate value.
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy


def optimized_model(params):
    """The optimized model
       Input: params from Optuna"""

    model = models.vgg16(pretrained=True)
    for param in model.parameters():
        param.requires_grad = False
    n_layers = params["n_layers"]
    layers = []
    in_features = model.classifier[0].in_features
    for i in range(n_layers):
        out_features = params[f"n_units_l{i}"]
        layers.append(nn.Linear(in_features, out_features))
        layers.append(nn.BatchNorm1d(out_features))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(params[f"dropout_l{i}"]))
        in_features = out_features

    layers.append(nn.Linear(in_features, CLASSES))
    model.classifier = nn.Sequential(*layers)

    return model


def optimized_loop(model, optimizer):
    """Training loop for optimized model"""

    train_loader, test_loader = get_dataloaders(
        os.path.join("/content", "brand_data"), batch_size=BATCHSIZE
    )
    criterion = nn.CrossEntropyLoss()
    test_accuracies = []
    # Training of the model.
    model.train()

    best_test = 0
    for epoch in range(EPOCHS):
        for batch_idx, (data, target) in enumerate(train_loader):
            # Limiting training data for faster epochs.
            if batch_idx * BATCHSIZE >= N_TRAIN_EXAMPLES:
                break
            data, target = data.to(DEVICE), target.to(DEVICE)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # Validation of the model.
        model.eval()
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(DEVICE), target.to(DEVICE)
                output = model(data)
                pred = output.argmax(
                    dim=1, keepdim=True
                )  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(test_loader.dataset)
        test_accuracies.append(accuracy)
        if accuracy > best_test:
            best_test = accuracy
            torch.save(
                model, f"/content/models/{model.__class__.__name__}_optimized.pt"
            )
        print(f"Epoch {epoch}/{EPOCHS}, Testing Accuracy is: {accuracy}")
    return test_accuracies
