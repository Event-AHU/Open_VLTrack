import argparse
import os
import pickle
import sys

import torch


ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from lib.test.analysis.extract_results import extract_results
from lib.test.evaluation import get_dataset, trackerlist
from lib.test.evaluation.environment import env_settings


def _parse_tracker_spec(spec):
    parts = spec.split(":")
    if len(parts) < 2 or len(parts) > 4:
        raise ValueError(
            "Tracker spec must be name:param[:run_id][:display_name], got %r" % spec
        )

    name, param = parts[0], parts[1]
    run_id = None
    display_name = None

    if len(parts) >= 3 and parts[2] != "":
        run_id = int(parts[2])
    if len(parts) == 4 and parts[3] != "":
        display_name = parts[3]

    return name, param, run_id, display_name


def _load_eval_data(args, trackers, dataset):
    settings = env_settings()
    eval_path = args.eval_data
    if eval_path is None:
        eval_path = os.path.join(settings.result_plot_path, args.report_name, "eval_data.pkl")

    if args.use_cache and os.path.isfile(eval_path):
        with open(eval_path, "rb") as fh:
            return pickle.load(fh)

    return extract_results(
        trackers,
        dataset,
        args.report_name,
        skip_missing_seq=args.skip_missing,
        plot_bin_gap=args.plot_bin_gap,
        exclude_invalid_frames=args.exclude_invalid_frames,
    )


def _summarize(eval_data):
    valid = torch.tensor(eval_data["valid_sequence"], dtype=torch.bool)
    overlap = torch.tensor(eval_data["ave_success_rate_plot_overlap"])[valid]
    center = torch.tensor(eval_data["ave_success_rate_plot_center"])[valid]
    center_norm = torch.tensor(eval_data["ave_success_rate_plot_center_norm"])[valid]
    thresholds = torch.tensor(eval_data["threshold_set_overlap"])

    auc_curve = overlap.mean(0) * 100.0
    auc = auc_curve.mean(-1)
    op50 = auc_curve[:, thresholds == 0.50].squeeze(-1)
    op75 = auc_curve[:, thresholds == 0.75].squeeze(-1)
    precision = center.mean(0)[:, 20] * 100.0
    norm_precision = center_norm.mean(0)[:, 20] * 100.0

    rows = []
    for idx, tracker in enumerate(eval_data["trackers"]):
        name = tracker.get("disp_name")
        if name is None:
            if tracker.get("run_id") is None:
                name = "%s_%s" % (tracker["name"], tracker["param"])
            else:
                name = "%s_%s_%03d" % (
                    tracker["name"],
                    tracker["param"],
                    tracker["run_id"],
                )
        rows.append((name, auc[idx], op50[idx], op75[idx], precision[idx], norm_precision[idx]))

    return int(valid.sum()), len(valid), rows


def main():
    parser = argparse.ArgumentParser(
        description="Summarize tracking result metrics without importing plot_results/matplotlib."
    )
    parser.add_argument("--dataset", default="lasot")
    parser.add_argument("--report-name", default="lasot")
    parser.add_argument(
        "--tracker",
        action="append",
        default=[],
        help="Tracker spec: name:param[:run_id][:display_name]. Can be repeated.",
    )
    parser.add_argument("--eval-data", default=None, help="Optional explicit eval_data.pkl path.")
    parser.add_argument("--use-cache", action="store_true", help="Read eval_data.pkl if present.")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--exclude-invalid-frames", action="store_true")
    parser.add_argument("--plot-bin-gap", type=float, default=0.05)
    args = parser.parse_args()

    dataset = get_dataset(args.dataset)
    trackers = []
    for spec in args.tracker:
        name, param, run_id, display_name = _parse_tracker_spec(spec)
        trackers.extend(
            trackerlist(
                name=name,
                parameter_name=param,
                dataset_name=args.dataset,
                run_ids=run_id,
                display_name=display_name,
                result_only=False,
            )
        )

    eval_data = _load_eval_data(args, trackers, dataset)
    valid_count, total_count, rows = _summarize(eval_data)

    print("valid_sequences,%d/%d" % (valid_count, total_count))
    print("tracker,AUC,OP50,OP75,P,NP")
    for name, auc, op50, op75, precision, norm_precision in rows:
        print(
            "%s,%.2f,%.2f,%.2f,%.2f,%.2f"
            % (name, auc, op50, op75, precision, norm_precision)
        )


if __name__ == "__main__":
    main()
