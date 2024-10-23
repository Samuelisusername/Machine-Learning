import requests
import numpy as np
import ssl
import certifi

from torchvision import transforms
from torch.utils.data import DataLoader, TensorDataset
import os
import torch
from torchvision import transforms
import torchvision.datasets as datasets
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import ResNet152_Weights, resnet152

requests.packages.urllib3.disable_warnings()

# Set up SSL
ssl_context = ssl.create_default_context(cafile=certifi.where())
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE

# Set device to GPU if available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def generate_embeddings():
    """
    Transform, resize and normalize the images and then use a pretrained model to extract
    the embeddings.
    """
    # Loading the model with weights
    weights = ResNet152_Weights.DEFAULT
    model = resnet152(weights=weights)
    preprocess = weights.transforms()
    train_transform = transforms.Compose([transforms.ToTensor()])
    train_dataset = datasets.ImageFolder(root="dataset/", transform=train_transform)

    model.to(device)
    embedding_size = list(model.children())[-1].in_features
    model = torch.nn.Sequential(*(list(model.children())[:-1]))  # Remove the last classification layer
    model.eval()

    num_images = len(train_dataset)
    embeddings = np.zeros((num_images, embedding_size))

    index = 0
    with torch.no_grad():
        for images, _ in train_dataset:
            embeddings[index, :] = torch.squeeze(model(preprocess(images).unsqueeze(0).to(device))).cpu().numpy()
            index += 1

    np.save('dataset/embeddings.npy', embeddings)

def get_data(file, train=True):
    """
    Loading the triplets from the file and generating the features and labels.
    """
    triplets = []
    with open(file) as f:
        for line in f:
            triplets.append(line.strip())

    # Loading dataset and embeddings
    train_dataset = datasets.ImageFolder(root="dataset/", transform=None)
    filenames = [s[0].split('/')[-1].replace('.jpg', '') for s in train_dataset.samples]
    embeddings = np.load('dataset/embeddings.npy')

    file_to_embedding = {filenames[i]: embeddings[i] for i in range(len(filenames))}

    X, y = [], []
    for t in triplets:
        emb = [file_to_embedding[a] for a in t.split()]
        X.append(np.hstack([emb[0], emb[1], emb[2]]))  # Image1 closer to Image0 than Image2
        y.append(1)
        if train:
            X.append(np.hstack([emb[0], emb[2], emb[1]]))  # Augmentation
            y.append(0)

    X = np.vstack(X)
    y = np.hstack(y)
    return X, y

def create_loader_from_np(X, y=None, train=True, batch_size=64, shuffle=True, num_workers=2):
    """
    Create a DataLoader object from numpy arrays.
    """
    if train:
        dataset = TensorDataset(torch.from_numpy(X).float(), torch.from_numpy(y).long())
    else:
        dataset = TensorDataset(torch.from_numpy(X).float())

    loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=num_workers)
    return loader

# Define the model
class Net(nn.Module):
    def __init__(self, input_size):
        super(Net, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_size, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 25),
            nn.ReLU(),
            nn.Linear(25, 10),
            nn.Sigmoid(),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        return self.layers(x)

def train_model(train_loader, input_size):
    """
    The training procedure of the model; it accepts the training data, defines the model 
    and then trains it.
    """
    model = Net(input_size)
    model.train()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()
    n_epochs = 10

    for epoch in range(n_epochs):
        total_loss = 0
        for X, y in train_loader:
            X, y = X.to(device), y.to(device).float()
            optimizer.zero_grad()
            output = model(X).squeeze()
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{n_epochs}, Loss: {total_loss:.4f}")

    return model

def test_model(model, loader):
    """
    Testing procedure of the model.
    """
    model.eval()
    predictions = []

    with torch.no_grad():
        for x_batch in loader:
            x_batch = x_batch[0].to(device)  # Batch as tuple (X,)
            predicted = model(x_batch).cpu().numpy()
            predicted[predicted >= 0.5] = 1
            predicted[predicted < 0.5] = 0
            predictions.append(predicted)

    predictions = np.vstack(predictions)
    np.savetxt("results.txt", predictions, fmt='%i')

if __name__ == '__main__':
    TRAIN_TRIPLETS = 'train_triplets.txt'
    TEST_TRIPLETS = 'test_triplets.txt'

    if not os.path.exists('dataset/embeddings.npy'):
        generate_embeddings()

    # Load training data
    X, y = get_data(TRAIN_TRIPLETS)
    input_size = X.shape[1]  # Update input size
    train_loader = create_loader_from_np(X, y, train=True, batch_size=64)
    del X, y

    # Load testing data
    X_test, _ = get_data(TEST_TRIPLETS, train=False)
    test_loader = create_loader_from_np(X_test, train=False, batch_size=2048, shuffle=False)
    del X_test

    # Train and test the model
    model = train_model(train_loader, input_size)
    test_model(model, test_loader)

    print("Results saved to results.txt")
