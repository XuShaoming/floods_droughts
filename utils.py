"""Shared training utilities."""

import os
import json
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt

from hydrology_metrics import calculate_hydrology_metrics


def get_scheduler(optimizer, config):
	"""
	Create learning rate scheduler based on configuration.

	Parameters:
	-----------
	optimizer : torch.optim.Optimizer
		The optimizer to schedule
	config : dict
		Scheduler configuration

	Returns:
	--------
	scheduler : torch.optim.lr_scheduler
		Learning rate scheduler
	"""
	scheduler_type = config.get("type", "ReduceLROnPlateau")

	if scheduler_type == "ReduceLROnPlateau":
		return optim.lr_scheduler.ReduceLROnPlateau(
			optimizer,
			mode="min",
			factor=float(config.get("factor", 0.5)),
			patience=int(config.get("patience", 5)),
			min_lr=float(config.get("min_lr", 1e-6)),
			verbose=True,
		)
	if scheduler_type == "StepLR":
		return optim.lr_scheduler.StepLR(
			optimizer,
			step_size=int(config.get("step_size", 20)),
			gamma=float(config.get("gamma", 0.5)),
		)
	if scheduler_type == "CosineAnnealingWarmRestarts":
		return optim.lr_scheduler.CosineAnnealingWarmRestarts(
			optimizer,
			T_0=int(config.get("T_0", 10)),
			T_mult=int(config.get("T_mult", 2)),
			eta_min=float(config.get("min_lr", 1e-6)),
		)
	if scheduler_type == "ExponentialLR":
		return optim.lr_scheduler.ExponentialLR(
			optimizer,
			gamma=float(config.get("gamma", 0.95)),
		)
	if scheduler_type == "OneCycleLR":
		raise NotImplementedError(
			"OneCycleLR requires train_loader information. Use train_yaml.py for this scheduler."
		)
	if scheduler_type == "CosineAnnealingLR":
		return optim.lr_scheduler.CosineAnnealingLR(
			optimizer,
			T_max=int(config.get("T_max", 50)),
			eta_min=float(config.get("min_lr", 1e-6)),
		)
	raise ValueError(f"Unknown scheduler type: {scheduler_type}")


class EarlyStopping:
	"""Early stopping utility to prevent overfitting."""

	def __init__(self, patience=7, min_delta=0, restore_best_weights=True):
		self.patience = patience
		self.min_delta = min_delta
		self.restore_best_weights = restore_best_weights
		self.best_loss = None
		self.counter = 0
		self.best_weights = None

	def __call__(self, val_loss, model):
		if self.best_loss is None:
			self.best_loss = val_loss
			self.save_checkpoint(model)
		elif val_loss < self.best_loss - self.min_delta:
			self.best_loss = val_loss
			self.counter = 0
			self.save_checkpoint(model)
		else:
			self.counter += 1

		if self.counter >= self.patience:
			if self.restore_best_weights:
				model.load_state_dict(self.best_weights)
			return True
		return False

	def save_checkpoint(self, model):
		"""Save model weights."""
		self.best_weights = model.state_dict().copy()


def calculate_metrics(predictions, targets, scaler=None):
	"""
	Calculate evaluation metrics and optionally denormalize.

	Parameters:
	-----------
	predictions : np.ndarray
		Predicted values
	targets : np.ndarray
		True values
	scaler : sklearn.preprocessing.StandardScaler, optional
		Scaler to inverse transform the data

	Returns:
	--------
	dict
		Dictionary containing various metrics
	"""
	if scaler is not None:
		if predictions.ndim == 3:
			orig_shape = predictions.shape
			predictions = predictions.reshape(-1, predictions.shape[-1])
			targets = targets.reshape(-1, targets.shape[-1])

			predictions = scaler.inverse_transform(predictions)
			targets = scaler.inverse_transform(targets)

			predictions = predictions.reshape(orig_shape)
			targets = targets.reshape(orig_shape)
		else:
			predictions = scaler.inverse_transform(predictions)
			targets = scaler.inverse_transform(targets)

	return calculate_hydrology_metrics(predictions, targets)


def save_model(model, optimizer, epoch, loss, model_config, save_path):
	"""Save model checkpoint."""
	checkpoint = {
		"epoch": epoch,
		"model_state_dict": model.state_dict(),
		"optimizer_state_dict": optimizer.state_dict(),
		"loss": loss,
		"model_config": model_config,
	}
	torch.save(checkpoint, save_path)
	print(f"Model saved to {save_path}")


def plot_training_history(train_losses, val_losses, save_dir, filename="learning_curves.png"):
	"""Plot and save training/validation loss curves."""
	plt.figure(figsize=(10, 6))

	plt.plot(train_losses, label="Training Loss", color="blue")
	plt.plot(val_losses, label="Validation Loss", color="red")
	plt.title("Training and Validation Loss")
	plt.xlabel("Epoch")
	plt.ylabel("Loss")
	plt.legend()
	plt.grid(True)

	plt.tight_layout()
	output_path = os.path.join(save_dir, filename)
	plt.savefig(output_path, dpi=300, bbox_inches="tight")
	plt.close()
	print(f"Saved learning curves to {output_path}")


def seed_everything(seed=42):
	"""
	Seed all random number generators for reproducible results.
	"""
	import random

	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)

	if torch.cuda.is_available():
		torch.cuda.manual_seed(seed)
		torch.cuda.manual_seed_all(seed)

	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

	os.environ["PYTHONHASHSEED"] = str(seed)
	print(f"All random seeds set to {seed} for reproducible results")
