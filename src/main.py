import argparse
import os
import shutil
import sys
import time
import warnings
from random import sample
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn import metrics
from torch.autograd import Variable
from torch.optim.lr_scheduler import MultiStepLR

# Use the unified CIFData (with optional charges) and utilities
from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet

import wandb


# ----------------------------------------------------------------------------- #
# Constants
# ----------------------------------------------------------------------------- #

PARTIAL_CHARGE_FILE = "charges/charges_dict.json"


# ----------------------------------------------------------------------------- #
# Argument parsing
# ----------------------------------------------------------------------------- #

def str2bool(v: str) -> bool:
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "t", "yes", "y", "1"):
        return True
    if v.lower() in ("false", "f", "no", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected for --charge.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Crystal Graph Convolutional Neural Networks"
    )

    parser.add_argument(
        "data_options",
        metavar="OPTIONS",
        nargs="+",
        help=(
            "Dataset options, starting with the path to the root directory, "
            "followed by additional options (currently only root_dir is used)."
        ),
    )

    parser.add_argument(
        "--task",
        choices=["regression", "classification"],
        default="regression",
        help="Regression or classification task (default: regression).",
    )
    parser.add_argument(
        "--disable-cuda",
        action="store_true",
        help="Disable CUDA.",
    )
    parser.add_argument(
        "-j",
        "--workers",
        default=0,
        type=int,
        metavar="N",
        help="Number of data loading workers (default: 0).",
    )
    parser.add_argument(
        "--epochs",
        default=30,
        type=int,
        metavar="N",
        help="Number of total epochs to run (default: 30).",
    )
    parser.add_argument(
        "--start-epoch",
        default=0,
        type=int,
        metavar="N",
        help="Manual epoch number (for restarts).",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        default=256,
        type=int,
        metavar="N",
        help="Mini-batch size (default: 256).",
    )
    parser.add_argument(
        "--lr",
        "--learning-rate",
        dest="lr",
        default=0.01,
        type=float,
        metavar="LR",
        help="Initial learning rate (default: 0.01).",
    )
    parser.add_argument(
        "--lr-milestones",
        default=[500],
        nargs="+",
        type=int,
        metavar="N",
        help="Milestones for the LR scheduler (default: [500]).",
    )
    parser.add_argument(
        "--momentum",
        default=0.9,
        type=float,
        metavar="M",
        help="Momentum (for SGD).",
    )
    parser.add_argument(
        "--weight-decay",
        "--wd",
        dest="weight_decay",
        default=0.0,
        type=float,
        metavar="W",
        help="Weight decay (default: 0).",
    )
    parser.add_argument(
        "--print-freq",
        "-p",
        default=10,
        type=int,
        metavar="N",
        help="Print frequency (default: 10).",
    )
    parser.add_argument(
        "--resume",
        default="",
        type=str,
        metavar="PATH",
        help="Path to latest checkpoint (default: none).",
    )

    # Train/val/test splits
    train_group = parser.add_mutually_exclusive_group()
    train_group.add_argument(
        "--train-ratio",
        default=None,
        type=float,
        metavar="N",
        help="Training ratio (default: None).",
    )
    train_group.add_argument(
        "--train-size",
        default=None,
        type=int,
        metavar="N",
        help="Number of training samples (default: None).",
    )

    valid_group = parser.add_mutually_exclusive_group()
    valid_group.add_argument(
        "--val-ratio",
        default=0.1,
        type=float,
        metavar="N",
        help="Validation ratio (default: 0.1).",
    )
    valid_group.add_argument(
        "--val-size",
        default=None,
        type=int,
        metavar="N",
        help="Number of validation samples (default: None).",
    )

    test_group = parser.add_mutually_exclusive_group()
    test_group.add_argument(
        "--test-ratio",
        default=0.1,
        type=float,
        metavar="N",
        help="Test ratio (default: 0.1).",
    )
    test_group.add_argument(
        "--test-size",
        default=None,
        type=int,
        metavar="N",
        help="Number of test samples (default: None).",
    )

    # Model & optimization
    parser.add_argument(
        "--optim",
        default="SGD",
        type=str,
        metavar="SGD",
        help="Optimizer: SGD or Adam (default: SGD).",
    )
    parser.add_argument(
        "--atom-fea-len",
        default=64,
        type=int,
        metavar="N",
        help="Hidden atom feature size in conv layers.",
    )
    parser.add_argument(
        "--h-fea-len",
        default=128,
        type=int,
        metavar="N",
        help="Hidden feature size after pooling.",
    )
    parser.add_argument(
        "--n-conv",
        default=3,
        type=int,
        metavar="N",
        help="Number of convolution layers.",
    )
    parser.add_argument(
        "--n-h",
        default=1,
        type=int,
        metavar="N",
        help="Number of hidden layers after pooling.",
    )
    parser.add_argument(
        "--embedding",
        default="default",
        type=str,
        help='Embedding name for atom_init file (default: "default").',
    )

    # Whether to use partial charges
    parser.add_argument(
        "--charge",
        type=str2bool,
        default=True,
        help="Use partial charges (True/False). Default: True.",
    )

    # Random seed
    parser.add_argument(
        "--random",
        default=123,
        type=int,
        help="Random seed (default: 123).",
    )

    # WandB project
    parser.add_argument(
        "--wandb-project",
        default="my-project",
        type=str,
        help="Weights & Biases project name.",
    )

    return parser


# ----------------------------------------------------------------------------- #
# Metric helpers
# ----------------------------------------------------------------------------- #

class AverageMeter:
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val: float = 0.0
        self.avg: float = 0.0
        self.sum: float = 0.0
        self.count: int = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


class Normalizer:
    def __init__(self, tensor: torch.Tensor) -> None:
        self.mean = torch.mean(tensor)
        self.std = torch.std(tensor)

    def norm(self, tensor: torch.Tensor) -> torch.Tensor:
        return (tensor - self.mean) / self.std

    def denorm(self, normed_tensor: torch.Tensor) -> torch.Tensor:
        return normed_tensor * self.std + self.mean

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]


def mae(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(target - prediction))


def class_eval(prediction: torch.Tensor, target: torch.Tensor) -> Tuple[float, float, float, float, float]:
    prediction = np.exp(prediction.numpy())
    target_np = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target_np)

    if not target_label.shape:
        target_label = np.asarray([target_label])

    if prediction.shape[1] != 2:
        raise NotImplementedError("Only binary classification is supported.")

    precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
        target_label, pred_label, average="binary"
    )
    auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
    accuracy = metrics.accuracy_score(target_label, pred_label)

    return accuracy, precision, recall, fscore, auc_score


# ----------------------------------------------------------------------------- #
# Checkpoint helpers
# ----------------------------------------------------------------------------- #

def save_checkpoint(
    state: Dict[str, Any],
    is_best: bool,
    embedding: str,
    random_seed: int,
    use_charge: bool,
    filename: str = "checkpoint.pth.tar",
) -> None:
    torch.save(state, filename)
    if is_best:
        tag = "charge" if use_charge else "base"
        best_name = f"model_best_{tag}_{embedding}_{random_seed}.pth.tar"
        shutil.copyfile(filename, best_name)
        wandb.save(best_name)


# ----------------------------------------------------------------------------- #
# Training / evaluation loops
# ----------------------------------------------------------------------------- #

def train_one_epoch(
    train_loader,
    model: nn.Module,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    normalizer: Normalizer,
    args: argparse.Namespace,
) -> None:
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()

    if args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    model.train()
    end = time.time()

    for i, (input_batch, target, _) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.cuda:
            input_var = (
                Variable(input_batch[0].cuda(non_blocking=True)),
                Variable(input_batch[1].cuda(non_blocking=True)),
                input_batch[2].cuda(non_blocking=True),
                [idx.cuda(non_blocking=True) for idx in input_batch[3]],
            )
        else:
            input_var = (
                Variable(input_batch[0]),
                Variable(input_batch[1]),
                input_batch[2],
                input_batch[3],
            )

        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        if args.cuda:
            target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.detach().cpu()), target)
            losses.update(loss.detach().cpu().item(), target.size(0))
            mae_errors.update(mae_error.item(), target.size(0))
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.detach().cpu(), target
            )
            losses.update(loss.detach().cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
                )
            else:
                print(
                    f"Epoch: [{epoch}][{i}/{len(train_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Data {data_time.val:.3f} ({data_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                    f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                    f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                    f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                    f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
                )

    if args.task == "regression":
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses.avg,
                "train_mae": mae_errors.avg,
            }
        )
    else:
        wandb.log(
            {
                "epoch": epoch,
                "train_loss": losses.avg,
                "train_accuracy": accuracies.avg,
                "train_precision": precisions.avg,
                "train_recall": recalls.avg,
                "train_fscore": fscores.avg,
                "train_auc": auc_scores.avg,
            }
        )


def evaluate(
    val_loader,
    model: nn.Module,
    criterion: nn.Module,
    normalizer: Normalizer,
    epoch,
    args: argparse.Namespace,
    test: bool = False,
):
    batch_time = AverageMeter()
    losses = AverageMeter()

    if args.task == "regression":
        mae_errors = AverageMeter()
    else:
        accuracies = AverageMeter()
        precisions = AverageMeter()
        recalls = AverageMeter()
        fscores = AverageMeter()
        auc_scores = AverageMeter()

    if test:
        test_targets: List[float] = []
        test_preds: List[float] = []
        test_cif_ids: List[str] = []

    model.eval()
    end = time.time()

    for i, (input_batch, target, batch_cif_ids) in enumerate(val_loader):
        if args.cuda:
            with torch.no_grad():
                input_var = (
                    Variable(input_batch[0].cuda(non_blocking=True)),
                    Variable(input_batch[1].cuda(non_blocking=True)),
                    input_batch[2].cuda(non_blocking=True),
                    [idx.cuda(non_blocking=True) for idx in input_batch[3]],
                )
        else:
            with torch.no_grad():
                input_var = (
                    Variable(input_batch[0]),
                    Variable(input_batch[1]),
                    input_batch[2],
                    input_batch[3],
                )

        if args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        if args.cuda:
            with torch.no_grad():
                target_var = Variable(target_normed.cuda(non_blocking=True))
        else:
            with torch.no_grad():
                target_var = Variable(target_normed)

        output = model(*input_var)
        loss = criterion(output, target_var)

        if args.task == "regression":
            mae_error = mae(normalizer.denorm(output.detach().cpu()), target)
            losses.update(loss.detach().cpu().item(), target.size(0))
            mae_errors.update(mae_error.item(), target.size(0))

            if test:
                test_pred = normalizer.denorm(output.detach().cpu())
                test_preds += test_pred.view(-1).tolist()
                test_targets += target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.detach().cpu(), target
            )
            losses.update(loss.detach().cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))

            if test:
                test_prob = torch.exp(output.detach().cpu())
                assert test_prob.shape[1] == 2
                test_preds += test_prob[:, 1].tolist()
                test_targets += target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if args.task == "regression":
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})"
                )
            else:
                print(
                    f"Test: [{i}/{len(val_loader)}]\t"
                    f"Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    f"Loss {losses.val:.4f} ({losses.avg:.4f})\t"
                    f"Accu {accuracies.val:.3f} ({accuracies.avg:.3f})\t"
                    f"Precision {precisions.val:.3f} ({precisions.avg:.3f})\t"
                    f"Recall {recalls.val:.3f} ({recalls.avg:.3f})\t"
                    f"F1 {fscores.val:.3f} ({fscores.avg:.3f})\t"
                    f"AUC {auc_scores.val:.3f} ({auc_scores.avg:.3f})"
                )

    if test:
        if args.task == "regression":
            wandb.log({"test_loss": losses.avg, "test_mae": mae_errors.avg})
        else:
            wandb.log(
                {
                    "test_loss": losses.avg,
                    "test_accuracy": accuracies.avg,
                    "test_precision": precisions.avg,
                    "test_recall": recalls.avg,
                    "test_fscore": fscores.avg,
                    "test_auc": auc_scores.avg,
                }
            )

        # Save test predictions
        import csv

        tag = "charge" if args.charge else "base"
        out_path = f"test_results_{tag}_{args.embedding}_{args.random}.csv"
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for cif_id, target_val, pred_val in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target_val, pred_val))
    else:
        if args.task == "regression":
            wandb.log(
                {
                    "epoch": epoch,
                    "val_loss": losses.avg,
                    "val_mae": mae_errors.avg,
                }
            )
        else:
            wandb.log(
                {
                    "epoch": epoch,
                    "val_loss": losses.avg,
                    "val_accuracy": accuracies.avg,
                    "val_precision": precisions.avg,
                    "val_recall": recalls.avg,
                    "val_fscore": fscores.avg,
                    "val_auc": auc_scores.avg,
                }
            )

    star_label = "**" if test else "*"
    if args.task == "regression":
        print(f" {star_label} MAE {mae_errors.avg:.3f}")
        return mae_errors.avg
    else:
        print(f" {star_label} AUC {auc_scores.avg:.3f}")
        return auc_scores.avg


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main() -> None:
    parser = build_parser()
    args = parser.parse_args(sys.argv[1:])

    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    wandb.init(project=args.wandb_project, config=vars(args))

    # Root dir is first data_options element
    root_dir = args.data_options[0]

    # Decide whether to pass charges
    partial_charge_file = PARTIAL_CHARGE_FILE if args.charge else None

    dataset = CIFData(
        root_dir=root_dir,
        random_seed=args.random,
        embedding_name=args.embedding,
        partial_charge_file=partial_charge_file,
    )

    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=args.batch_size,
        train_ratio=args.train_ratio,
        num_workers=args.workers,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        pin_memory=args.cuda,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        return_test=True,
    )

    # Normalizer
    if args.task == "classification":
        normalizer = Normalizer(torch.zeros(2))
        normalizer.load_state_dict({"mean": 0.0, "std": 1.0})
    else:
        if len(dataset) < 500:
            warnings.warn(
                "Dataset has fewer than 500 data points. "
                "Lower accuracy is expected."
            )
            sample_data_list = [dataset[i] for i in range(len(dataset))]
        else:
            sample_data_list = [dataset[i] for i in sample(range(len(dataset)), 500)]

        _, sample_target, _ = collate_pool(sample_data_list)
        normalizer = Normalizer(sample_target)

    # Model
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=args.atom_fea_len,
        n_conv=args.n_conv,
        h_fea_len=args.h_fea_len,
        n_h=args.n_h,
        classification=(args.task == "classification"),
        use_global_context=True,
    )

    if args.cuda:
        model.cuda()

    wandb.watch(model, log="all")

    if args.task == "classification":
        criterion: nn.Module = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    if args.optim == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    elif args.optim == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            args.lr,
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError("Only SGD or Adam is allowed as --optim")

    best_mae_error = 1e10 if args.task == "regression" else 0.0

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"=> loading checkpoint '{args.resume}'")
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint["epoch"]
            best_mae_error = checkpoint["best_mae_error"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            normalizer.load_state_dict(checkpoint["normalizer"])
            print(f"=> loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")
        else:
            print(f"=> no checkpoint found at '{args.resume}'")

    scheduler = MultiStepLR(optimizer, milestones=args.lr_milestones, gamma=0.1)

    for epoch in range(args.start_epoch, args.epochs):
        train_one_epoch(train_loader, model, criterion, optimizer, epoch, normalizer, args)
        mae_or_auc = evaluate(val_loader, model, criterion, normalizer, epoch, args)

        if mae_or_auc != mae_or_auc:  # NaN
            print("Exit due to NaN.")
            sys.exit(1)

        scheduler.step()

        if args.task == "regression":
            is_best = mae_or_auc < best_mae_error
            best_mae_error = min(mae_or_auc, best_mae_error)
        else:
            is_best = mae_or_auc > best_mae_error
            best_mae_error = max(mae_or_auc, best_mae_error)

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "best_mae_error": best_mae_error,
                "optimizer": optimizer.state_dict(),
                "normalizer": normalizer.state_dict(),
                "args": vars(args),
            },
            is_best=is_best,
            embedding=args.embedding,
            random_seed=args.random,
            use_charge=args.charge,
        )

    # Test on best model
    print("--------- Evaluate Model on Test Set ---------")
    tag = "charge" if args.charge else "base"
    best_checkpoint_path = f"model_best_{tag}_{args.embedding}_{args.random}.pth.tar"
    best_checkpoint = torch.load(best_checkpoint_path)
    model.load_state_dict(best_checkpoint["state_dict"])

    evaluate(test_loader, model, criterion, normalizer, epoch="test", args=args, test=True)


if __name__ == "__main__":
    main()