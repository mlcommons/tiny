import argparse

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from utils.data import get_test_dataloader
from utils.data import get_training_dataloader
from utils.model import Resnet8v1EEMBC
from utils.training import WarmUpLR
from utils.training import eval_training
from utils.training import train_one_epoch

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size. Default value is 32 according to TF training procedure.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Number of epochs. Default value is 500 according to TF training procedure.",
    )
    parser.add_argument(
        "--warmup-epochs",
        type=int,
        default=0,
        help="Number of epochs for LR linear warmup.",
    )
    parser.add_argument(
        "--lr",
        default=0.001,
        type=float,
        help="Initial learning rate. Default value is 1e-3 according to TF training procedure.",
    )
    parser.add_argument(
        "--lr-decay",
        default=0.99,
        type=float,
        help="Initial learning rate. Default value is 1e-3 according to TF training procedure.",
    )
    parser.add_argument(
        "--data-dir",
        default="cifar-10-torch",
        type=str,
        help="Path to dataset (will be downloaded).",
    )
    parser.add_argument(
        "--workers", default=4, type=int, help="Number of data loading processes."
    )
    parser.add_argument(
        "--weight-decay", default=1e-4, type=float, help="Weight decay for optimizer."
    )
    parser.add_argument("--log-dir", type=str, default="trained_models")
    args = parser.parse_args()

    train_loader = get_training_dataloader(
        cifar_10_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    val_loader = get_test_dataloader(
        cifar_10_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )

    model = Resnet8v1EEMBC()
    if torch.cuda.is_available():
        model = model.cuda()
    optimizer = Adam(
        params=model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    train_scheduler = LambdaLR(
        optimizer=optimizer, lr_lambda=lambda epoch: args.lr_decay**epoch
    )
    warmup_scheduler = None
    if args.warmup_epochs:
        warmup_scheduler = WarmUpLR(optimizer=optimizer, total_iters=args.warmup_epochs)

    writer = SummaryWriter(log_dir=args.log_dir)

    loss_function = torch.nn.CrossEntropyLoss()

    best_accuracy = 0.0
    for epoch in range(1, args.epochs + 1):
        if epoch > args.warmup_epochs:
            train_scheduler.step()

        train_one_epoch(
            model=model,
            train_dataloader=train_loader,
            loss_function=loss_function,
            optimizer=optimizer,
            epoch=epoch,
            writer=writer,
            warmup_scheduler=warmup_scheduler,
            warmup_epochs=args.warmup_epochs,
            train_scheduler=train_scheduler,
        )

        accuracy = eval_training(
            model=model,
            dataloader=val_loader,
            loss_function=loss_function,
            epoch=epoch,
            log_to_tensorboard=True,
            writer=writer,
        )

        if best_accuracy < accuracy:
            weights_path = f"{args.log_dir}/best.pth"
            print(f"saving weights file to {weights_path}")
            torch.save(model, weights_path)
            best_accuracy = accuracy
            continue

        writer.flush()

    writer.close()
