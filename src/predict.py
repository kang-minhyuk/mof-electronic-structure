import argparse
import os
import sys
import time
import warnings
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from sklearn import metrics

from cgcnn.data import CIFData, collate_pool, get_train_val_test_loader
from cgcnn.model import CrystalGraphConvNet


PARTIAL_CHARGE_FILE = "charges/charges_dict.json"


parser = argparse.ArgumentParser(description="Crystal graph neural network prediction")

parser.add_argument("modelpath", help="Path to the trained model checkpoint (.pth.tar).")
parser.add_argument("root_dir", help="Path to directory containing CIF files and id_prop.csv.")

parser.add_argument(
    "--disable-cuda",
    action="store_true",
    help="Disable CUDA.",
)
parser.add_argument(
    "--print-freq",
    "-p",
    default=10,
    type=int,
    metavar="N",
    help="Print frequency (default: 10).",
)

# Optional overrides (if you really want to deviate from the training config)
parser.add_argument(
    "--batch-size",
    default=None,
    type=int,
    metavar="N",
    help="Batch size for prediction (default: use training setting).",
)
parser.add_argument(
    "--workers",
    default=None,
    type=int,
    metavar="N",
    help="Number of data loading workers (default: use training setting).",
)
parser.add_argument(
    "--train-ratio",
    type=float,
    default=None,
    help="Train ratio used during training, if you want to override.",
)
parser.add_argument(
    "--val-ratio",
    type=float,
    default=None,
    help="Validation ratio used during training, if you want to override.",
)
parser.add_argument(
    "--test-ratio",
    type=float,
    default=None,
    help="Test ratio used during training, if you want to override.",
)
parser.add_argument(
    "--train-size",
    type=int,
    default=None,
    help="Train size used during training, if you want to override.",
)
parser.add_argument(
    "--val-size",
    type=int,
    default=None,
    help="Validation size used during training, if you want to override.",
)
parser.add_argument(
    "--test-size",
    type=int,
    default=None,
    help="Test size used during training, if you want to override.",
)

args = parser.parse_args(sys.argv[1:])
args.cuda = not args.disable_cuda and torch.cuda.is_available()

# ----------------------------------------------------------------------------- #
# Load checkpoint and training args
# ----------------------------------------------------------------------------- #

if os.path.isfile(args.modelpath):
    print(f"=> loading model params from '{args.modelpath}'")
    model_checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
    model_args = argparse.Namespace(**model_checkpoint["args"])
    print(f"=> loaded model params from '{args.modelpath}'")
else:
    raise FileNotFoundError(f"=> no model params found at '{args.modelpath}'")

if model_args.task == "regression":
    best_mae_error = 1e10
else:
    best_mae_error = 0.0


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

class Normalizer(object):
    """Normalize a tensor and restore it later."""

    def __init__(self, tensor: torch.Tensor):
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


def class_eval(prediction: torch.Tensor, target: torch.Tensor):
    prediction = np.exp(prediction.numpy())
    target = target.numpy()
    pred_label = np.argmax(prediction, axis=1)
    target_label = np.squeeze(target)
    if prediction.shape[1] == 2:
        precision, recall, fscore, _ = metrics.precision_recall_fscore_support(
            target_label, pred_label, average="binary"
        )
        auc_score = metrics.roc_auc_score(target_label, prediction[:, 1])
        accuracy = metrics.accuracy_score(target_label, pred_label)
    else:
        raise NotImplementedError
    return accuracy, precision, recall, fscore, auc_score


class AverageMeter(object):
    """Computes and stores the average and current value."""

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(self, val: float, n: int = 1) -> None:
        self.val = float(val)
        self.sum += float(val) * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0.0


# ----------------------------------------------------------------------------- #
# Main
# ----------------------------------------------------------------------------- #

def main():
    global args, model_args, best_mae_error

    # Ensure device flag
    args.cuda = not args.disable_cuda and torch.cuda.is_available()

    # Use training config by default; allow CLI override if provided
    batch_size = args.batch_size if args.batch_size is not None else model_args.batch_size
    workers = args.workers if args.workers is not None else model_args.workers

    train_ratio = args.train_ratio if args.train_ratio is not None else getattr(model_args, "train_ratio", None)
    val_ratio = args.val_ratio if args.val_ratio is not None else getattr(model_args, "val_ratio", 0.1)
    test_ratio = args.test_ratio if args.test_ratio is not None else getattr(model_args, "test_ratio", 0.1)

    train_size = args.train_size if args.train_size is not None else getattr(model_args, "train_size", None)
    val_size = args.val_size if args.val_size is not None else getattr(model_args, "val_size", None)
    test_size = args.test_size if args.test_size is not None else getattr(model_args, "test_size", None)

    # Consistent random seed & embedding with training
    random_seed = getattr(model_args, "random", 123)
    embedding_name = getattr(model_args, "embedding", "default")
    use_charge = getattr(model_args, "charge", False)

    # Decide whether to use charges based on how the model was trained
    partial_charge_file = PARTIAL_CHARGE_FILE if use_charge else None
    if use_charge and not os.path.exists(PARTIAL_CHARGE_FILE):
        raise FileNotFoundError(
            f"Model was trained with charges, but {PARTIAL_CHARGE_FILE} does not exist."
        )

    print(f"Using root_dir = {args.root_dir}")
    print(f"Using embedding_name = {embedding_name}")
    print(f"Using random_seed = {random_seed}")
    print(f"Using charges = {use_charge}")

    # Load dataset (same CIFData as training)
    dataset = CIFData(
        root_dir=args.root_dir,
        random_seed=random_seed,
        embedding_name=embedding_name,
        partial_charge_file=partial_charge_file,
    )

    # We only really need the test loader, but get_train_val_test_loader expects return_test=True
    train_loader, val_loader, test_loader = get_train_val_test_loader(
        dataset=dataset,
        collate_fn=collate_pool,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        num_workers=workers,
        pin_memory=args.cuda,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        return_test=True,
    )

    # Build model using stored training arguments
    structures, _, _ = dataset[0]
    orig_atom_fea_len = structures[0].shape[-1]
    nbr_fea_len = structures[1].shape[-1]

    model = CrystalGraphConvNet(
        orig_atom_fea_len,
        nbr_fea_len,
        atom_fea_len=model_args.atom_fea_len,
        n_conv=model_args.n_conv,
        h_fea_len=model_args.h_fea_len,
        n_h=model_args.n_h,
        classification=(model_args.task == "classification"),
        use_global_context=getattr(model_args, "use_global_context", True),
    )

    if args.cuda:
        model.cuda()

    # Loss (only used to report numbers, not to train)
    if model_args.task == "classification":
        criterion = nn.NLLLoss()
    else:
        criterion = nn.MSELoss()

    # Normalizer: will be overwritten by checkpoint state_dict
    normalizer = Normalizer(torch.zeros(3))

    # Load trained weights + normalizer from checkpoint
    print(f"=> loading model '{args.modelpath}'")
    if os.path.isfile(args.modelpath):
        checkpoint = torch.load(args.modelpath, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint["state_dict"])
        normalizer.load_state_dict(checkpoint["normalizer"])
        print(
            "=> loaded model '{}' (epoch {}, best_val_metric {})".format(
                args.modelpath, checkpoint["epoch"], checkpoint["best_mae_error"]
            )
        )
    else:
        raise FileNotFoundError(f"=> no model found at '{args.modelpath}'")

    # Evaluate on test set
    validate(test_loader, model, criterion, normalizer, test=True)


def validate(val_loader, model, criterion, normalizer, test: bool = False):
    batch_time = AverageMeter()
    losses = AverageMeter()
    if model_args.task == "regression":
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
        with torch.no_grad():
            if args.cuda:
                input_var = (
                    Variable(input_batch[0].cuda(non_blocking=True)),
                    Variable(input_batch[1].cuda(non_blocking=True)),
                    input_batch[2].cuda(non_blocking=True),
                    [crys_idx.cuda(non_blocking=True) for crys_idx in input_batch[3]],
                )
            else:
                input_var = (
                    Variable(input_batch[0]),
                    Variable(input_batch[1]),
                    input_batch[2],
                    input_batch[3],
                )

        if model_args.task == "regression":
            target_normed = normalizer.norm(target)
        else:
            target_normed = target.view(-1).long()

        with torch.no_grad():
            if args.cuda:
                target_var = Variable(target_normed.cuda(non_blocking=True))
            else:
                target_var = Variable(target_normed)

        # compute output
        output = model(*input_var)
        loss = criterion(output, target_var)

        # metrics
        if model_args.task == "regression":
            mae_error = mae(normalizer.denorm(output.data.cpu()), target)
            losses.update(loss.data.cpu().item(), target.size(0))
            mae_errors.update(mae_error.item(), target.size(0))
            if test:
                test_pred = normalizer.denorm(output.data.cpu())
                test_target = target
                test_preds += test_pred.view(-1).tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids
        else:
            accuracy, precision, recall, fscore, auc_score = class_eval(
                output.data.cpu(), target
            )
            losses.update(loss.data.cpu().item(), target.size(0))
            accuracies.update(accuracy, target.size(0))
            precisions.update(precision, target.size(0))
            recalls.update(recall, target.size(0))
            fscores.update(fscore, target.size(0))
            auc_scores.update(auc_score, target.size(0))
            if test:
                test_pred = torch.exp(output.data.cpu())
                test_target = target
                assert test_pred.shape[1] == 2
                test_preds += test_pred[:, 1].tolist()
                test_targets += test_target.view(-1).tolist()
                test_cif_ids += batch_cif_ids

        # time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            if model_args.task == "regression":
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "MAE {mae_errors.val:.3f} ({mae_errors.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        mae_errors=mae_errors,
                    )
                )
            else:
                print(
                    "Test: [{0}/{1}]\t"
                    "Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t"
                    "Loss {loss.val:.4f} ({loss.avg:.4f})\t"
                    "Accu {accu.val:.3f} ({accu.avg:.3f})\t"
                    "Precision {prec.val:.3f} ({prec.avg:.3f})\t"
                    "Recall {recall.val:.3f} ({recall.avg:.3f})\t"
                    "F1 {f1.val:.3f} ({f1.avg:.3f})\t"
                    "AUC {auc.val:.3f} ({auc.avg:.3f})".format(
                        i,
                        len(val_loader),
                        batch_time=batch_time,
                        loss=losses,
                        accu=accuracies,
                        prec=precisions,
                        recall=recalls,
                        f1=fscores,
                        auc=auc_scores,
                    )
                )

    if test:
        star_label = "**"
        import csv

        out_path = "test_results_pred.csv"
        print(f"Writing prediction CSV to {out_path}")
        with open(out_path, "w", newline="") as f:
            writer = csv.writer(f)
            for cif_id, target_val, pred_val in zip(test_cif_ids, test_targets, test_preds):
                writer.writerow((cif_id, target_val, pred_val))
    else:
        star_label = "*"

    if model_args.task == "regression":
        print(f" {star_label} MAE {mae_errors.avg:.3f}")
        return mae_errors.avg
    else:
        print(f" {star_label} AUC {auc_scores.avg:.3f}")
        return auc_scores.avg


if __name__ == "__main__":
    main()