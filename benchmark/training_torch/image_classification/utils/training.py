import time

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class WarmUpLR(_LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        total_iters: int,
        last_epoch: int = -1,
    ):
        """Scheduler for learning rate warmup.

        Parameters
        ----------
        optimizer: torch.optim.Optimizer
            Optimizer, e.g. SGD.
        total_iters: int
            Number of iterations for warmup Learning rate phase.
        last_epoch: int
            The index of last epoch. Default: -1
        """
        self.total_iters = total_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        """Return current learning rate."""
        return [
            base_lr * self.last_epoch / (self.total_iters + 1e-8)
            for base_lr in self.base_lrs
        ]


def train_one_epoch(
    model: torch.nn.Module,
    train_dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    writer: torch.utils.tensorboard.SummaryWriter = None,
    warmup_scheduler: _LRScheduler = None,
    warmup_epochs: int = 0,
    train_scheduler: _LRScheduler = None,
):
    start = time.time()
    model.train()
    for batch_index, (images, labels) in enumerate(train_dataloader):
        if torch.cuda.is_available():
            labels = labels.cuda()
            images = images.cuda()

        optimizer.zero_grad()
        outputs = model(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        n_iter = (epoch - 1) * len(train_dataloader) + batch_index + 1

        # update training loss for each iteration
        writer.add_scalar("Train/loss", loss.item(), n_iter)
        writer.add_scalar("Lr", optimizer.param_groups[0]["lr"], n_iter)

    if epoch <= warmup_epochs:
        warmup_scheduler.step()
    else:
        train_scheduler.step(epoch)

    print(
        f"Training Epoch: "
        f"{epoch} "
        f"Loss: {loss.item():0.4f}\t"
        f'LR: {optimizer.param_groups[0]["lr"]:0.6f}'
    )

    finish = time.time()

    print(f"epoch {epoch} training time consumed: {finish - start:.2f}s")


def eval_training(
    model: torch.nn.Module,
    dataloader: torch.utils.data.DataLoader,
    loss_function: torch.nn.Module,
    epoch: int = 0,
    log_to_tensorboard: bool = True,
    writer: SummaryWriter = None,
):
    start = time.time()
    model.eval()

    test_loss = 0.0  # cost function error
    correct = 0.0
    with torch.no_grad():
        for images, labels in dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
                labels = labels.cuda()

            outputs = model(images)
            loss = loss_function(outputs, labels)

            test_loss += loss.item()
            _, preds = outputs.max(1)
            correct += preds.eq(labels).sum()

    finish = time.time()
    print("Evaluating Network.....")
    dataset_size = len(dataloader.dataset)
    loss = test_loss / dataset_size
    accuracy = correct.float() / dataset_size
    print(
        f"Test set: Epoch: {epoch},"
        f" Average loss: {loss:.4f},"
        f" Accuracy: {accuracy:.4f},"
        f" Time consumed:{finish - start:.2f}s"
    )
    print()

    # add information to tensorboard
    if log_to_tensorboard and writer:
        writer.add_scalar("Test/Average loss", loss, epoch)
        writer.add_scalar("Test/Accuracy", accuracy, epoch)

    return correct.float() / dataset_size
