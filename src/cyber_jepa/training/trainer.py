"""
Training and Evaluation Loop for Cyber-JEPA.

Supports AdamW, Cosine Warmup Scheduler, Automatic Mixed Precision (AMP),
EMA target encoder updates, dynamic gradient accumulation, early stopping,
and atomic checkpointing.
"""

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from cyber_jepa.models.jepa import CyberJEPA


class Trainer:
    """Cyber-JEPA Trainer managing optimization, validation, and checkpointing."""

    def __init__(
        self,
        model: CyberJEPA,
        train_loader: DataLoader,
        val_loader: DataLoader,
        optimizer: torch.optim.Optimizer,
        scheduler: Any,
        run_dir: Path,
        device: torch.device,
        max_epochs: int = 50,
        min_epochs: int = 15,
        patience: int = 10,
        grad_clip: float = 1.0,
        accum_steps: int = 2,
        monitor_metric: str = "val_pred_loss",
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.run_dir = run_dir
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.device = device

        self.max_epochs = max_epochs
        self.min_epochs = min_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.accum_steps = accum_steps
        self.monitor_metric = monitor_metric
        self.last_val_metrics: dict[str, float] = {}

        # AMP GradScaler if CUDA available
        self.scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.global_step = 0
        self.total_steps = max_epochs * len(train_loader) // max(1, accum_steps)

    def train_epoch(self, epoch: int) -> float:
        """Run one training epoch with gradient accumulation and EMA target updates."""
        self.model.train()
        total_loss = 0.0
        num_batches = len(self.train_loader)

        self.optimizer.zero_grad()

        for step, batch in enumerate(self.train_loader):
            hist = batch["history_flat"].to(self.device)
            actions = batch["action_seq"].to(self.device)
            target = batch["target_flat"].to(self.device)

            with torch.amp.autocast("cuda", enabled=(self.device.type == "cuda")):
                loss, pred_z, target_z = self.model(hist, actions, target)
                loss_scaled: torch.Tensor = loss / self.accum_steps

            self.scaler.scale(loss_scaled).backward()
            loss_val = loss.item()
            if not np.isnan(loss_val) and not np.isinf(loss_val):
                total_loss += loss_val
            else:
                print(f"[Warning] Non-finite loss encountered at step {step}: {loss_val}", flush=True)

            if (step + 1) % self.accum_steps == 0 or (step + 1) == num_batches:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                if self.scheduler is not None:
                    self.scheduler.step()

                self.global_step += 1
                # EMA update after optimizer step
                self.model.update_target_encoder(self.global_step, self.total_steps)

        avg_loss = total_loss / max(1, num_batches)
        return avg_loss

    @torch.no_grad()
    def evaluate(self, return_dict: bool = False) -> float | dict[str, float]:
        """Evaluate validation loss deterministically across all batches."""
        self.model.eval()
        total_loss = 0.0
        total_pred_loss = 0.0
        total_var_loss = 0.0
        total_cov_loss = 0.0
        num_batches = len(self.val_loader)

        for batch in self.val_loader:
            hist = batch["history_flat"].to(self.device)
            actions = batch["action_seq"].to(self.device)
            target = batch["target_flat"].to(self.device)

            loss, _, _ = self.model(hist, actions, target)
            total_loss += loss.item()
            if hasattr(self.model, "last_loss_breakdown") and self.model.last_loss_breakdown:
                total_pred_loss += self.model.last_loss_breakdown.get("pred_loss", loss.item())
                total_var_loss += self.model.last_loss_breakdown.get("var_loss", 0.0)
                total_cov_loss += self.model.last_loss_breakdown.get("cov_loss", 0.0)
            else:
                total_pred_loss += loss.item()

        n = max(1, num_batches)
        self.last_val_metrics = {
            "val_loss": total_loss / n,
            "val_pred_loss": total_pred_loss / n,
            "val_var_loss": total_var_loss / n,
            "val_cov_loss": total_cov_loss / n,
        }
        if return_dict:
            return self.last_val_metrics
        return total_loss / n

    def fit(self) -> dict[str, Any]:
        """Run full training loop with early stopping and checkpointing."""
        history: dict[str, list[float]] = {
            "train_loss": [],
            "val_loss": [],
            "val_pred_loss": [],
            "val_var_loss": [],
            "val_cov_loss": [],
        }

        for epoch in range(1, self.max_epochs + 1):
            t_loss = self.train_epoch(epoch)
            v_loss = self.evaluate()
            v_pred_loss = self.last_val_metrics.get("val_pred_loss", v_loss)

            history["train_loss"].append(t_loss)
            history["val_loss"].append(v_loss)
            history["val_pred_loss"].append(v_pred_loss)
            history["val_var_loss"].append(self.last_val_metrics.get("val_var_loss", 0.0))
            history["val_cov_loss"].append(self.last_val_metrics.get("val_cov_loss", 0.0))

            monitor_val = self.last_val_metrics.get(self.monitor_metric, v_loss)

            print(
                f"Epoch {epoch:02d}/{self.max_epochs:02d} - Train Loss: {t_loss:.6f} - "
                f"Val Loss: {v_loss:.6f} - Val Pred Loss: {v_pred_loss:.6f}",
                flush=True,
            )

            # Save last checkpoint atomically
            self.save_checkpoint(self.run_dir / "last.pt", epoch, v_loss, val_pred_loss=v_pred_loss)

            # Best model selection based on configured monitor_metric
            if monitor_val < self.best_val_loss:
                self.best_val_loss = monitor_val
                self.patience_counter = 0
                self.save_checkpoint(self.run_dir / "best.pt", epoch, v_loss, val_pred_loss=v_pred_loss)
            else:
                self.patience_counter += 1

            if epoch >= self.min_epochs and self.patience_counter >= self.patience:
                print(f"Early stopping triggered at epoch {epoch} (monitored {self.monitor_metric}={monitor_val:.6f})")
                break

        with open(self.run_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        return history

    def save_checkpoint(
        self,
        path: Path,
        epoch: int,
        val_loss: float,
        val_pred_loss: float | None = None,
    ) -> None:
        """Save complete atomic model checkpoint."""
        ckpt = {
            "epoch": epoch,
            "global_step": self.global_step,
            "val_loss": val_loss,
            "val_pred_loss": val_pred_loss if val_pred_loss is not None else val_loss,
            "online_encoder": self.model.online_encoder.state_dict(),
            "target_encoder": self.model.target_encoder.state_dict(),
            "action_encoder": self.model.action_encoder.state_dict(),
            "predictor": self.model.predictor.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        tmp_path = path.with_suffix(".tmp")
        torch.save(ckpt, tmp_path)
        tmp_path.replace(path)


JEPATrainer = Trainer
