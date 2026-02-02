#!/usr/bin/env python
"""
LTCNとNeural ODEの外乱（Corruption）耐性評価スクリプト

異なる外乱（ノイズ、白飛び、トンネル出口など）におけるモデルの頑健性を評価します。
"""

import argparse
from pathlib import Path
import json
import random
from typing import Any

import mlflow
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from src.core.models import LTCNController, NeuralODEController
from src.driving.data import setup_dataloaders
from src.utils import (
    add_gaussian_noise,
    add_static_bias,
    add_overexposure
)


class CorruptionRobustnessEvaluator:
    """外乱耐性評価クラス"""

    def __init__(self, device: torch.device = None):
        self.device = device or torch.device("cpu")

    def evaluate_robustness(
        self,
        models: dict[str, nn.Module],
        test_data: dict[str, torch.Tensor],
        levels: list[float],
        corruption_type: str = "noise",
    ) -> dict[str, Any]:
        """比較評価を実行"""
        results: dict[str, Any] = {
            "models": {},
            "comparison": {},
            "metadata": {"corruption_type": corruption_type},
        }

        for level in levels:
            print(f"Testing {corruption_type} level: {level}")

            clean_frames = test_data["clean_frames"]
            sensors = test_data["sensors"]

            # 外乱の適用
            corrupted_frames = self._apply_corruption(
                clean_frames, level, corruption_type
            )

            # 各モデルのテスト
            for name, model in models.items():
                if name not in results["models"]:
                    results["models"][name] = {}
                
                metrics = self._evaluate_model(
                    model, clean_frames, corrupted_frames, sensors
                )
                results["models"][name][f"level_{level}"] = metrics

        # 比較サマリー (2モデル以上ある場合のみ)
        if len(models) > 1:
            results["comparison"] = self._generate_comparison_summary(results)

        return results

    def _apply_corruption(
        self, frames: torch.Tensor, level: float, corruption_type: str
    ) -> torch.Tensor:
        """フレームに外乱を適用"""
        if corruption_type == "noise":
            # ガウシアンノイズ (level = std)
            return torch.stack([add_gaussian_noise(f, std=level) for f in frames])
        elif corruption_type == "bias":
            # Level 1: Static Bias (level = bias value)
            return torch.stack([add_static_bias(f, bias=level) for f in frames])
        elif corruption_type == "overexposure":
            # Level 2: Contrast/Overexposure (level = factor)
            return torch.stack([add_overexposure(f, factor=level) for f in frames])

    def _evaluate_model(
        self,
        model: nn.Module,
        clean_data: torch.Tensor,
        corrupted_data: torch.Tensor,
        sensors: torch.Tensor,
    ) -> dict[str, float]:
        """単一モデルの評価"""
        clean_data = clean_data.to(self.device)
        corrupted_data = corrupted_data.to(self.device)
        sensors = sensors.to(self.device)

        model.eval()
        with torch.no_grad():
            # クリーンデータでの予測
            pred_clean, _ = model(clean_data)
            # 外乱付きデータでの予測
            pred_corrupted, _ = model(corrupted_data)

            # シーケンスの最後のタイムステップで評価
            pred_clean_last = pred_clean[:, -1, :]
            pred_corrupted_last = pred_corrupted[:, -1, :]
            sensors_last = sensors[:, -1, :]

            # 外乱付きデータでの誤差
            control_mse = nn.MSELoss()(pred_corrupted_last, sensors_last).item()
            control_mae = nn.L1Loss()(pred_corrupted_last, sensors_last).item()

            # 出力の安定性: 外乱による予測の変動
            output_variance = nn.MSELoss()(pred_corrupted_last, pred_clean_last).item()

        return {
            "control_mse": control_mse,
            "control_mae": control_mae,
            "output_variance": output_variance,
        }

    def _generate_comparison_summary(self, results: dict[str, Any]) -> dict[str, Any]:
        """比較サマリーを生成"""
        summary: dict[str, Any] = {
            "winner_by_metric": {},
            "robustness_score": {},
        }

        # 比較は results["models"] の中の最初の2つで行う（現状の実装では）
        model_names = list(results["models"].keys())
        if len(model_names) < 2:
            return {}
        
        m1, m2 = model_names[0], model_names[1]

        metrics = ["control_mse", "control_mae", "output_variance"]

        for metric in metrics:
            m1_values = []
            m2_values = []

            for key in results["models"][m1]:
                if key.startswith("level_"):
                    m1_val = results["models"][m1][key][metric]
                    m2_val = results["models"][m2][key][metric]
                    m1_values.append(m1_val)
                    m2_values.append(m2_val)

            m1_avg = float(np.mean(m1_values))
            m2_avg = float(np.mean(m2_values))

            winner = m1 if m1_avg < m2_avg else m2

            summary["winner_by_metric"][metric] = {
                "winner": winner,
                f"{m1}_avg": m1_avg,
                f"{m2}_avg": m2_avg,
                "diff": float(abs(m1_avg - m2_avg)),
            }

        # ロバスト性スコア (Slope)
        for model_name in model_names:
            model_results = results["models"][model_name]
            levels = []
            mse_values = []
            for key in sorted(model_results.keys()):
                if key.startswith("level_"):
                    lvl = float(key.split("_")[1])
                    levels.append(lvl)
                    mse_values.append(model_results[key]["control_mse"])

            if len(levels) > 1:
                slope = np.polyfit(levels, mse_values, 1)[0]
                summary["robustness_score"][model_name] = {"slope": float(slope)}

        return summary

    def visualize_comparison(
        self, results: dict[str, Any], save_path: Path | None = None
    ):
        """比較結果を可視化"""
        num_cols = 3
        fig, axes = plt.subplots(1, num_cols, figsize=(18, 5))
        if num_cols == 1:
            axes = [axes]
        else:
            axes = axes.flatten()

        c_type = results["metadata"]["corruption_type"]
        
        # Determine levels from the first model found
        first_model = list(results["models"].keys())[0]
        levels = [
            float(k.split("_")[1]) for k in results["models"][first_model] if k.startswith("level_")
        ]
        levels.sort()

        metrics_to_plot = [
            ("control_mse", f"MSE vs {c_type}"),
            ("control_mae", f"MAE vs {c_type}"),
            ("output_variance", f"Variance vs {c_type}"),
        ]

        # Plot for each model
        styles = ["b-o", "r-s", "g-^", "y-d"] # Support up to 4 models distinctively

        for i, (metric, title) in enumerate(metrics_to_plot):
            for idx, (name, model_results) in enumerate(results["models"].items()):
                vals = [model_results[f"level_{lvl}"][metric] for lvl in levels]
                style = styles[idx % len(styles)]
                axes[i].plot(levels, vals, style, label=name, linewidth=2)
            
            axes[i].set_xlabel("Severity Level")
            axes[i].set_ylabel(metric)
            axes[i].set_title(title)
            axes[i].legend()
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"Plot saved to {save_path}")


def run_corruption_robustness_evaluation(args: argparse.Namespace):
    """評価実行"""
    # ... (Setup Code similar to previous script)
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")

    # Init seeds
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run() as _:
        mlflow.log_params(vars(args))

        # Data
        _, _, test_loader, _ = setup_dataloaders(
            data_dir=args.data_dir,
            sequence_length=args.sequence_length,
            batch_size=args.batch_size,
            processed_dir=args.processed_dir,
        )

        # Model Selection
        models = {}
        model_names = args.model_type
        
        # Check if single specific path is provided but multiple models requested
        if len(model_names) > 1 and args.model_path is not None:
             print("Warning: --model-path provided but multiple models are selected. Ignoring custom path and using defaults.")

        for model_name in model_names:
            # Set default path if not provided or valid
            if args.model_path is None or len(model_names) > 1:
                if model_name == "ltcn":
                    model_path = "./driving_results/LTCN_checkpoint.pth"
                elif model_name == "node":
                    model_path = "./driving_results/NODE_checkpoint.pth"
            else:
                model_path = args.model_path

            print(f"Initializing {model_name.upper()} model...")
            
            if model_name == "ltcn":
                model = LTCNController(
                    frame_height=64,
                    frame_width=64,
                    output_dim=6,
                    hidden_dim=args.hidden_dim,
                    num_layers=args.num_layers_ltcn,
                )
            elif model_name == "node":
                model = NeuralODEController(
                    frame_height=64,
                    frame_width=64,
                    output_dim=6,
                    hidden_dim=args.hidden_dim,
                    num_hidden_layers=args.num_hidden_layers_node,
                )
            
            # Load Weights
            print(f"Loading {model_name} from {model_path}")
            try:
                ckpt = torch.load(model_path, map_location=device)
                state_dict = (
                    ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
                )
                model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Error loading {model_name}: {e}")
                # Try raw load if state dict fails matching
                try:
                    model = torch.load(model_path, map_location=device)
                except Exception as e2:
                    print(f"Failed to load {model_name} from {model_path}. Skipping. Error: {e2}")
                    continue

            model.eval()
            model.to(device)
            models[model_name] = model

        if not models:
            print("No models loaded. Exiting.")
            return

        # Get Test Batch
        batch = next(iter(test_loader))
        frames, sensors = batch[0].to(device), batch[1].to(device)
        test_data = {"clean_frames": frames, "sensors": sensors}

        # Levels
        levels = [float(x) for x in args.levels.split(",")]

        # Run Eval
        evaluator = CorruptionRobustnessEvaluator(device)
        results = evaluator.evaluate_robustness(
            models, test_data, levels, args.corruption_type
        )

        # Save
        # Make filename reflect models used
        names_str = "_".join(sorted(models.keys()))
        r_path = save_dir / f"robustness_{args.corruption_type}_{names_str}.json"
        p_path = save_dir / f"robustness_{args.corruption_type}_{names_str}.png"

        with open(r_path, "w") as f:
            json.dump(results, f, indent=2)

        evaluator.visualize_comparison(results, p_path)

        mlflow.log_artifact(str(r_path))
        mlflow.log_artifact(str(p_path))

        print(f"Done. Check {save_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate corruption robustness")
    
    # Required or Core Args
    parser.add_argument("--model-type", required=True, nargs='+', choices=["ltcn", "node"], help="Model type(s) to evaluate")
    parser.add_argument("--data-dir", default="./data/raw", help="Path to raw data directory")
    
    # Optional Args
    parser.add_argument("--model-path", default=None, help="Path to model checkpoint (optional, defaults based on model-type. Ignored if multiple models)")
    parser.add_argument("--processed-dir", default=None)
    parser.add_argument("--save-dir", default="./corruption_results")

    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers-ltcn", type=int, default=4)
    parser.add_argument("--num-hidden-layers-node", type=int, default=1)
    parser.add_argument("--solver", default="dopri5")

    parser.add_argument("--sequence-length", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")

    # Corruption Params
    parser.add_argument(
        "--corruption-type",
        choices=["noise", "bias", "overexposure"],
        default="noise",
        help="Type of corruption to apply",
    )
    parser.add_argument(
        "--levels",
        default="0.0,0.1,0.2,0.3",
        help="Comma-separated levels (std, bias, factor, intensity)",
    )

    args = parser.parse_args()
    run_corruption_robustness_evaluation(args)
