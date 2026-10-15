"""
Training Engine and Optimizer Benchmarking Module.
Handles:
- Optimizer instantiation (SGD, NAdam, Adadelta, Adagrad/Ftrl, RMSprop, Adam)
- Training & validation loops
- Loss tracking and checkpoint management
"""

import os
from pathlib import Path
from typing import Dict, Any, List, Tuple
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from configs.config import Config


def get_optimizer(model: nn.Module, optimizer_name: str, opt_params: dict) -> torch.optim.Optimizer:
    """Instantiate optimizer based on paper Table 3 settings."""
    name = optimizer_name.strip()
    if name == "SGD":
        return torch.optim.SGD(model.parameters(), lr=opt_params.get("lr", 0.01), momentum=opt_params.get("momentum", 0.9))
    elif name == "NAdam":
        return torch.optim.NAdam(model.parameters(), lr=opt_params.get("lr", 0.001))
    elif name == "Adadelta":
        return torch.optim.Adadelta(model.parameters(), lr=opt_params.get("lr", 1.0))
    elif name in ("Adagrad", "Ftrl"):
        return torch.optim.Adagrad(model.parameters(), lr=opt_params.get("lr", 0.01))
    elif name == "RMSprop":
        return torch.optim.RMSprop(model.parameters(), lr=opt_params.get("lr", 0.001))
    elif name == "Adam":
        return torch.optim.Adam(model.parameters(), lr=opt_params.get("lr", 0.001))
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")


class Trainer:
    """Trainer class for ElectricityLSTM."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: nn.Module,
        config: Config,
        optimizer_name: str = "Adam",
    ):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.config = config
        self.optimizer_name = optimizer_name
        self.device = config.device

        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float("inf")

        # Ensure checkpoints directory exists
        os.makedirs(self.config.checkpoints_dir, exist_ok=True)
        self.checkpoint_path = Path(self.config.checkpoints_dir) / f"{self.optimizer_name}_best.pt"

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Run single training epoch."""
        self.model.train()
        total_loss = 0.0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(batch_x)
            loss = self.criterion(predictions, batch_y)
            loss.backward()

            # Gradient clipping to ensure numerical stability in recurrent nets
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item() * len(batch_x)

        return total_loss / len(train_loader.dataset)

    def validate(self, val_loader: DataLoader) -> float:
        """Run validation loop."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                predictions = self.model(batch_x)
                loss = self.criterion(predictions, batch_y)
                total_loss += loss.item() * len(batch_x)

        return total_loss / len(val_loader.dataset)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """Full training loop with early stopping."""
        patience_counter = 0

        for epoch in range(1, self.config.num_epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)

            # Check if this is the best model so far
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                torch.save(self.model.state_dict(), self.checkpoint_path)
                patience_counter = 0
            else:
                patience_counter += 1

            if verbose and (epoch % 10 == 0 or epoch == 1 or epoch == self.config.num_epochs):
                print(
                    f"[{self.optimizer_name}] Epoch {epoch:03d}/{self.config.num_epochs:03d} "
                    f"- Train Loss: {train_loss:.6f} - Val Loss: {val_loss:.6f} "
                    f"{'(*Best Saved)' if patience_counter == 0 else ''}"
                )

            if patience_counter >= self.config.early_stopping_patience:
                if verbose:
                    print(f"[{self.optimizer_name}] Early stopping triggered at epoch {epoch}.")
                break

        # Load best weights before returning
        if os.path.exists(self.checkpoint_path):
            self.model.load_state_dict(torch.load(self.checkpoint_path, map_location=self.device))

        return {
            "optimizer_name": self.optimizer_name,
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "best_val_loss": self.best_val_loss,
            "checkpoint_path": str(self.checkpoint_path),
        }
