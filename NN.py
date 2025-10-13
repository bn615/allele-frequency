import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from typing import Tuple, Optional, Dict


class DataGenerator:
    def __init__(self, seed: Optional[int] = None):
        if seed is not None:
            np.random.seed(seed)
            torch.manual_seed(seed)
            
    def sample_allele_frequencies(self, n_samples: int = 1, distribution: str = "uniform") -> np.ndarray:
        if distribution == "uniform":
            alpha = [1.0, 1.0, 1.0]

        elif distribution == "realistic":
            # A allele is more frequent than B, while O allele is most frequent
            alpha = [7.0, 2.0, 20.0]

        else:
            raise ValueError("Unsupported distribution type")
        
        return np.random.dirichlet(alpha, size=n_samples)
    
    def generate_blood_types(self, p: float, q: float, r: float, n: int) -> Tuple[int, int, int, int]:
        # Genotype frequencies based on allele frequencies
        freq_A = p*p + 2*p*r
        freq_B = q*q + 2*q*r
        freq_O = r*r
        freq_AB = 2*p*q
        
        total = freq_A + freq_B + freq_O + freq_AB

        probabilities = [freq_A/total, freq_B/total, freq_AB/total, freq_O/total]
        
        return np.random.multinomial(n, probabilities)
    
    def generate_sample(self, sample_size: int, distribution: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        freqs = self.sample_allele_frequencies(distribution = distribution)[0]
        p, q, r = freqs

        n1, n2, n3, n4 = self.generate_blood_types(p, q, r, sample_size)

        X = np.array([n1, n2, n3, n4], dtype=np.float32)
        y = np.array([p, q, r], dtype=np.float32)

        X /= X.sum()  # Normalize input features

        return X, y
    
    def generate_dataset(self, n_samples: int = 5000, sample_size_range: Tuple[int, int] = (100, 1000), distribution: str = "uniform") -> Tuple[np.ndarray, np.ndarray]:
        X_list, y_list = [], []
        for _ in range(n_samples):
            sample_size = np.random.randint(sample_size_range[0], sample_size_range[1] + 1)
            X, y = self.generate_sample(sample_size, distribution)
            X_list.append(X)
            y_list.append(y)
        
        return np.array(X_list), np.array(y_list)
    
    def create_dataloader(self, n_train: int = 3000, n_val: int = 1000, n_test: int = 1000, batch_size: int = 32, sample_size_range: Tuple[int, int] = (100, 1000), distribution: str = "uniform", **kwargs) -> Tuple[DataLoader, DataLoader, DataLoader]:
        # Training samples, validation samples, test samples
        X_train, y_train = self.generate_dataset(n_train, sample_size_range, distribution) 
        X_val, y_val = self.generate_dataset(n_val, sample_size_range, distribution)
        X_test, y_test = self.generate_dataset(n_test, sample_size_range, distribution)

        train_dataset = ABODataset(X_train, y_train)
        val_dataset = ABODataset(X_val, y_val)
        test_dataset = ABODataset(X_test, y_test)

        train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = True, **kwargs)
        val_loader = DataLoader(val_dataset, batch_size = batch_size, shuffle = False, **kwargs)
        test_loader = DataLoader(test_dataset, batch_size = batch_size, shuffle = False, **kwargs)

        return train_loader, val_loader, test_loader


class ABODataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
        

class ABONN(torch.nn.Module):
    # Simple NN with 2 hidden layers
    # input(4) -> hidden(32) -> hidden(32) -> output(3)
    def __init__(self, input_size = 4, hidden_size: int = 32):
        super().__init__()
        layers = []

        for i in range(2):
            layers.append(nn.Linear(input_size, hidden_size))
            layers.append(nn.ReLU())
            input_size = hidden_size

        layers.append(nn.Linear(hidden_size, 3))
        layers.append(nn.Softmax(dim=1))

        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
    
class ABONN_Dropout(torch.nn.Module):
    def __init__(self, input_size=4, hidden_size=32, dropout_rate=0.2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),  # Add dropout
            nn.Linear(hidden_size, 3),
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        return self.net(x)
    
    def predict_with_uncertainty(self, x, n_samples=100):
        """Get mean and confidence intervals via MC Dropout"""
        self.train()  # Keep dropout active!
        predictions = []
        
        with torch.no_grad():
            for _ in range(n_samples):
                pred = self.forward(x)
                predictions.append(pred.numpy())
        
        predictions = np.array(predictions)  # Shape: (n_samples, batch_size, 3)
        
        mean = predictions.mean(axis=0)
        std = predictions.std(axis=0)
        ci_lower = np.percentile(predictions, 2.5, axis=0)
        ci_upper = np.percentile(predictions, 97.5, axis=0)
        
        return {
            'mean': mean,
            'std': std,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper
        }


class ABOTrainer:
    def __init__(self, model: torch.nn.Module, lr: float = 1e-3, loss_fn = torch.nn.MSELoss()):
        self.model = model
        self.criterion = loss_fn
        self.optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        self.history = {'train_loss': [], 'val_loss': [], 'learning_rates': []}

    def train_epoch(self, dataloader: DataLoader) -> float:
        self.model.train()
        total_loss = 0.0
        for X_batch, y_batch in dataloader:
            # forward pass
            self.optimizer.zero_grad()
            outputs = self.model(X_batch)

            loss = self.criterion(outputs, y_batch)

            # backward pass
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(dataloader.dataset)

    def evaluate(self, dataloader: DataLoader) -> float:
        self.model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for X_batch, y_batch in dataloader:
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                total_loss += loss.item() * X_batch.size(0)
        
        return total_loss / len(dataloader.dataset)
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader, n_epochs: int = 100) -> Dict:
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.evaluate(val_loader)

            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['learning_rates'].append(self.optimizer.param_groups[0]['lr'])
        
            if (epoch+1) % 10 == 0 or epoch == 0:
                print(f"Epoch [{epoch+1}/{n_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
        return self.history

    
    def predict(self, X: np.ndarray) -> np.ndarray:
        X = X / X.sum(axis=1, keepdims=True)

        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.from_numpy(X).float()
            outputs = self.model(X_tensor)
            return outputs.numpy()
        
def visualize_training(history: Dict):
    epochs = range(1, len(history['train_loss']) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history['train_loss'], label='Train Loss')
    plt.plot(epochs, history['val_loss'], label='Val Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history['learning_rates'], label='Learning Rate')
    plt.xlabel('Epochs')
    plt.ylabel('Learning Rate')
    plt.title('Learning Rate Schedule')
    plt.legend()

    plt.tight_layout()
    plt.show()

def test():
    generator = DataGenerator()
    train_loader, val_loader, test_loader = generator.create_dataloader(n_train=10000, n_val=2000, n_test=2000, batch_size=64, sample_size_range=(100, 1000), distribution="realistic")

    model = ABONN()
    trainer = ABOTrainer(model)

    history = trainer.train(train_loader, val_loader, n_epochs=100)
    visualize_training(history)

    test_loss = trainer.evaluate(test_loader)
    print(f"Test Loss: {test_loss:.4f}")

    p, q, r = trainer.predict(np.array([[186, 38, 36, 284]]))[0]
    print(f"Predicted allele frequencies: p={p:.4f}, q={q:.4f}, r={r:.4f}")

if __name__ == "__main__":
    test()