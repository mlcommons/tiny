import argparse

import torch
from utils.data import get_test_dataloader
from utils.training import eval_training

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-ckpt",
        default="trained_models/best.pth",
        type=str,
        help="Path to model checkpoint for evaluation.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size. Default value is 32 according to TF training procedure.",
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
    args = parser.parse_args()

    model = torch.load(args.model_ckpt)

    val_loader = get_test_dataloader(
        cifar_10_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
    loss_function = torch.nn.CrossEntropyLoss()

    accuracy = eval_training(
        model=model,
        dataloader=val_loader,
        loss_function=loss_function,
        epoch=0,
        log_to_tensorboard=False,
        writer=None,
    )

    print(f"Model {args.model_ckpt} has accuracy: {accuracy}")
