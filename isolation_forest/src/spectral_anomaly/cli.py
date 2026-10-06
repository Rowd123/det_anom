"""Command-line entry points for independently executable processing stages."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import pandas as pd

from .config import load_config
from .workflows import (apply_spot,
                        score_isolation_forest, train_isolation_forest)


def _json(path):
    with Path(path).open(encoding="utf8") as stream: return json.load(stream)


def parser():
    root = argparse.ArgumentParser(prog="spectral-anomaly")
    root.add_argument("--log-level", default="INFO")
    commands = root.add_subparsers(dest="command", required=True)
    train = commands.add_parser("train-iforest"); train.add_argument("features"); train.add_argument("model"); train.add_argument("--config", required=True)
    score = commands.add_parser("score-iforest"); score.add_argument("features"); score.add_argument("model"); score.add_argument("output"); score.add_argument("--batch-size", type=int, default=65536); score.add_argument("--resume", action="store_true")
    spot = commands.add_parser("spot"); spot.add_argument("calibration"); spot.add_argument("scores"); spot.add_argument("output"); spot.add_argument("state"); spot.add_argument("--config", required=True); spot.add_argument("--resume", action="store_true")
    plot = commands.add_parser("plot-spot", help="Plot original CSV and saved SPOT window decisions")
    plot.add_argument("input"); plot.add_argument("decisions"); plot.add_argument("output")
    plot.add_argument("--source-id", required=True); plot.add_argument("--channel-id", required=True)
    plot.add_argument("--value-col", default="value"); plot.add_argument("--index-col", default="time")
    plot.add_argument("--sep", default=","); plot.add_argument("--datetime-format", default="ISO8601")
    prepare = commands.add_parser("prepare-windows")
    prepare.add_argument("input"); prepare.add_argument("output"); prepare.add_argument("--config", required=True)
    prepare.add_argument("--source-id", required=True); prepare.add_argument("--channel-id", required=True)
    prepare.add_argument("--value-col", default="value"); prepare.add_argument("--index-col", default="time")
    prepare.add_argument("--sep", default=",")
    prepare.add_argument("--datetime-format", default="ISO8601")
    windows = commands.add_parser("extract-windows")
    windows.add_argument("input"); windows.add_argument("output"); windows.add_argument("--config", required=True)
    split = commands.add_parser("split-windows")
    split.add_argument("input"); split.add_argument("output_dir"); split.add_argument("--config", required=True)
    analysis = commands.add_parser("analyze-scores", help="Rank saved scores and inspect raw feature correlations")
    analysis.add_argument("features"); analysis.add_argument("scores"); analysis.add_argument("output")
    analysis.add_argument("--model"); analysis.add_argument("--top-n", type=int, default=20)
    return root


def main(argv=None):
    args = parser().parse_args(argv); logging.basicConfig(level=getattr(logging, args.log_level.upper()), format="%(asctime)s %(levelname)s %(name)s %(message)s")
    if args.command == "analyze-scores":
        from .analysis import analyze_scores
        output = analyze_scores(args.features, args.scores, args.output, model_path=args.model, top_n=args.top_n)
        logging.info("Analysis reports: %s", output)
        return None
    if args.command == "plot-spot":
        from .spot_plot import export_spot_plot
        output = export_spot_plot(args.input, args.decisions, args.output,
                                 source_id=args.source_id, channel_id=args.channel_id,
                                 value_col=args.value_col, index_col=args.index_col,
                                 sep=args.sep, datetime_format=args.datetime_format)
        logging.info("Plotly report: %s", output)
        return None
    if args.command in {"prepare-windows", "extract-windows", "split-windows"}:
        from .window_features import prepare_windows, extract_window_features, split_window_features
        config = _json(args.config)
        if args.command == "prepare-windows":
            frame = pd.read_csv(args.input, sep=args.sep)
            frame.index = pd.to_datetime(frame.pop(args.index_col), utc=True, format=args.datetime_format)
            prepare_windows(frame, args.output, config, source_id=args.source_id,
                            channel_id=args.channel_id, value_col=args.value_col)
        elif args.command == "extract-windows":
            extract_window_features(args.input, args.output, config)
        else:
            split_window_features(args.input, args.output_dir, config["split"])
        return None
    if args.command == "train-iforest":
        train_isolation_forest(args.features, args.model, load_config(args.config, "models"))
        return None
    if args.command == "score-iforest":
        score_isolation_forest(args.features, args.model, args.output, batch_size=args.batch_size, resume=args.resume)
        return None
    config = _json(args.config); apply_spot(args.calibration, args.scores, args.output, args.state, config, resume=args.resume)
