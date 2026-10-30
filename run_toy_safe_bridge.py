"""2D safe-bridge toy experiment for Gaussian versus diffusion policies.

This experiment is tailored to the unsafe-bridge theorem in the paper:
two disconnected safe action regions are separated by an unsafe middle
region.  A log-concave Gaussian fitted to both modes must place mass in the
bridge, whereas a diffusion policy can model the disconnected safe support.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from pathlib import Path
from typing import Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.patches import Circle, Rectangle

from agents.bc_diffusion import Diffusion_BC


class ToyDataSampler:
    def __init__(self, state: torch.Tensor, action: torch.Tensor, reward: torch.Tensor, device: str):
        self.state = state.float()
        self.action = action.float()
        self.reward = reward.float()
        self.device = device
        self.size = self.state.shape[0]

    def sample(self, batch_size: int):
        ind = torch.randint(0, self.size, size=(batch_size,))
        state = self.state[ind].to(self.device)
        action = self.action[ind].to(self.device)
        reward = self.reward[ind].to(self.device)
        cost = torch.zeros_like(reward)
        not_done = torch.ones_like(reward)
        return state, action, state, reward, cost, not_done


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gaussian bridge-mass versus diffusion multimodality toy experiment."
    )
    parser.add_argument("--seed", default=2026, type=int)
    parser.add_argument("--device", default="auto", type=str)
    parser.add_argument("--num_data", default=10_000, type=int)
    parser.add_argument("--num_eval", default=10_000, type=int)
    parser.add_argument("--epochs", default=1_000, type=int)
    parser.add_argument("--batch_size", default=100, type=int)
    parser.add_argument("--lr", default=3e-4, type=float)
    parser.add_argument("--n_timesteps", default=50, type=int)
    parser.add_argument("--beta_schedule", default="vp", type=str)
    parser.add_argument("--mode_center", default=0.8, type=float)
    parser.add_argument("--safe_radius", default=0.22, type=float)
    parser.add_argument("--dataset_radius", default=0.16, type=float)
    parser.add_argument("--data_std", default=0.07, type=float)
    parser.add_argument("--max_action", default=1.5, type=float)
    parser.add_argument("--output_dir", default="toy_imgs/safe_bridge", type=str)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Use a tiny configuration for smoke tests only; not paper-quality.",
    )
    return parser.parse_args()


def choose_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda:0" if torch.cuda.is_available() else "cpu"
    return requested_device


def sample_truncated_gaussian_disk(
    num_samples: int,
    center: np.ndarray,
    std: float,
    radius: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a concentrated Gaussian blob while keeping all points inside a disk."""

    accepted = []
    remaining = num_samples

    while remaining > 0:
        proposal_count = max(remaining * 3, 256)
        candidates = rng.normal(loc=center, scale=std, size=(proposal_count, 2))
        inside = np.linalg.norm(candidates - center, axis=1) <= radius
        kept = candidates[inside][:remaining]
        accepted.append(kept)
        remaining -= len(kept)

    return np.concatenate(accepted, axis=0)


def build_dataset(args: argparse.Namespace, device: str) -> tuple[ToyDataSampler, np.ndarray]:
    rng = np.random.default_rng(args.seed)
    left_center = np.array([-args.mode_center, 0.0], dtype=np.float32)
    right_center = np.array([args.mode_center, 0.0], dtype=np.float32)

    left_count = args.num_data // 2
    right_count = args.num_data - left_count

    left_actions = sample_truncated_gaussian_disk(
        left_count,
        left_center,
        args.data_std,
        args.dataset_radius,
        rng,
    )
    right_actions = sample_truncated_gaussian_disk(
        right_count,
        right_center,
        args.data_std,
        args.dataset_radius,
        rng,
    )

    actions = np.concatenate([left_actions, right_actions], axis=0).astype(np.float32)
    rng.shuffle(actions)

    action_tensor = torch.from_numpy(actions)
    state_tensor = torch.zeros_like(action_tensor)
    reward_tensor = torch.zeros((len(actions), 1), dtype=torch.float32)
    return ToyDataSampler(state_tensor, action_tensor, reward_tensor, device), actions


def fit_full_gaussian(actions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the exact empirical full-covariance Gaussian MLE parameters."""

    mean = actions.mean(axis=0)
    cov = np.cov(actions.T, bias=True)
    cov = cov + 1e-6 * np.eye(actions.shape[1], dtype=np.float32)
    return mean, cov


def train_diffusion_policy(
    args: argparse.Namespace,
    data_sampler: Data_Sampler,
    device: str,
) -> Diffusion_BC:
    agent = Diffusion_BC(
        state_dim=2,
        action_dim=2,
        max_action=args.max_action,
        device=device,
        discount=0.99,
        tau=0.005,
        beta_schedule=args.beta_schedule,
        n_timesteps=args.n_timesteps,
        lr=args.lr,
    )

    iterations = math.ceil(args.num_data / args.batch_size)
    log_every = max(1, args.epochs // 10)

    for epoch in range(1, args.epochs + 1):
        agent.train(data_sampler, iterations=iterations, batch_size=args.batch_size)
        if epoch == 1 or epoch % log_every == 0 or epoch == args.epochs:
            print(f"[diffusion] epoch {epoch:4d}/{args.epochs}")

    return agent


def sample_diffusion(
    agent: DiffusionBC,
    num_samples: int,
    device: str,
    chunk_size: int = 2048,
) -> np.ndarray:
    samples = []
    with torch.no_grad():
        for start in range(0, num_samples, chunk_size):
            batch_size = min(chunk_size, num_samples - start)
            state = torch.zeros((batch_size, 2), device=device)
            action = agent.actor.sample(state)
            samples.append(action.cpu().numpy())
    return np.concatenate(samples, axis=0)


def in_disk(samples: np.ndarray, center: np.ndarray, radius: float) -> np.ndarray:
    return np.linalg.norm(samples - center, axis=1) <= radius


def compute_region_metrics(samples: np.ndarray, args: argparse.Namespace) -> Dict[str, float]:
    left_center = np.array([-args.mode_center, 0.0], dtype=np.float32)
    right_center = np.array([args.mode_center, 0.0], dtype=np.float32)
    midpoint_center = np.array([0.0, 0.0], dtype=np.float32)

    left_safe = in_disk(samples, left_center, args.safe_radius)
    right_safe = in_disk(samples, right_center, args.safe_radius)
    midpoint_set = in_disk(samples, midpoint_center, args.safe_radius)

    bridge_half_width = args.mode_center - args.safe_radius
    bridge = (
        (np.abs(samples[:, 0]) <= bridge_half_width)
        & (np.abs(samples[:, 1]) <= args.safe_radius)
    )

    safe_union = left_safe | right_safe
    left_mass = float(left_safe.mean())
    right_mass = float(right_safe.mean())

    return {
        "safe_mode_1_mass": left_mass,
        "safe_mode_2_mass": right_mass,
        "safe_union_mass": float(safe_union.mean()),
        "unsafe_bridge_mass": float(bridge.mean()),
        "midpoint_set_mass": float(midpoint_set.mean()),
        "outside_safe_union_mass": float((~safe_union).mean()),
        "prekopaleindler_half_bound": float(math.sqrt(left_mass * right_mass)),
    }


def add_geometry(ax: plt.Axes, args: argparse.Namespace) -> None:
    left_center = (-args.mode_center, 0.0)
    right_center = (args.mode_center, 0.0)
    bridge_half_width = args.mode_center - args.safe_radius

    ax.add_patch(
        Rectangle(
            (-bridge_half_width, -args.safe_radius),
            2 * bridge_half_width,
            2 * args.safe_radius,
            facecolor="#d62728",
            edgecolor="none",
            alpha=0.13,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            left_center,
            args.safe_radius,
            facecolor="#2ca02c",
            edgecolor="#1b7f1b",
            alpha=0.16,
            linewidth=1.5,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            right_center,
            args.safe_radius,
            facecolor="#2ca02c",
            edgecolor="#1b7f1b",
            alpha=0.16,
            linewidth=1.5,
            zorder=0,
        )
    )
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            args.safe_radius,
            fill=False,
            edgecolor="#b22222",
            linestyle="--",
            linewidth=1.5,
            zorder=1,
        )
    )

    axis_limit = args.mode_center + args.safe_radius + 0.25
    ax.set_xlim(-axis_limit, axis_limit)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal")
    ax.set_xlabel(r"$a_1$")
    ax.set_ylabel(r"$a_2$")


def plot_results(
    actions: np.ndarray,
    gaussian_samples: np.ndarray,
    diffusion_samples: np.ndarray,
    metrics: Dict[str, Dict[str, float]],
    args: argparse.Namespace,
) -> Path:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(
        1,
        4,
        figsize=(18.0, 4.5),
        gridspec_kw={"width_ratios": [1.0, 1.0, 1.0, 0.9]},
    )

    scatter_count = min(2_000, len(actions), len(gaussian_samples), len(diffusion_samples))
    panels = [
        ("Offline safe data", actions[:scatter_count], "#1f77b4"),
        (
            f"Gaussian MLE\nbridge={metrics['Gaussian MLE']['unsafe_bridge_mass']:.3f}",
            gaussian_samples[:scatter_count],
            "#ff7f0e",
        ),
        (
            f"Diffusion BC\nbridge={metrics['Diffusion BC']['unsafe_bridge_mass']:.3f}",
            diffusion_samples[:scatter_count],
            "#9467bd",
        ),
    ]

    for ax, (title, samples, color) in zip(axs[:3], panels):
        add_geometry(ax, args)
        ax.scatter(samples[:, 0], samples[:, 1], s=10, alpha=0.35, color=color, edgecolors="none")
        ax.set_title(title)

    labels = ["safe mode 1", "safe mode 2", "unsafe bridge"]
    metric_keys = ["safe_mode_1_mass", "safe_mode_2_mass", "unsafe_bridge_mass"]
    x = np.arange(len(labels))
    width = 0.36

    gaussian_values = [metrics["Gaussian MLE"][key] for key in metric_keys]
    diffusion_values = [metrics["Diffusion BC"][key] for key in metric_keys]
    axs[3].bar(x - width / 2, gaussian_values, width, label="Gaussian MLE", color="#ff7f0e")
    axs[3].bar(x + width / 2, diffusion_values, width, label="Diffusion BC", color="#9467bd")
    axs[3].set_xticks(x)
    axs[3].set_xticklabels(labels, rotation=20, ha="right")
    axs[3].set_ylim(0.0, 1.0)
    axs[3].set_ylabel("sample mass")
    axs[3].set_title("Region occupancy")
    axs[3].legend(frameon=False)
    axs[3].grid(axis="y", alpha=0.25)

    fig.suptitle(
        "Disconnected safe modes: Gaussian bridge mass versus diffusion multimodality",
        y=1.03,
        fontsize=15,
    )
    fig.tight_layout()

    stem = f"safe_bridge_seed{args.seed}"
    pdf_path = output_dir / f"{stem}.pdf"
    png_path = output_dir / f"{stem}.png"
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(png_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return pdf_path


def save_metrics(metrics: Dict[str, Dict[str, float]], args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"safe_bridge_metrics_seed{args.seed}"

    with (output_dir / f"{stem}.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)

    fieldnames = ["policy"] + list(next(iter(metrics.values())).keys())
    with (output_dir / f"{stem}.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for policy, values in metrics.items():
            writer.writerow({"policy": policy, **values})


def maybe_apply_quick_mode(args: argparse.Namespace) -> None:
    if not args.quick:
        return

    args.num_data = min(args.num_data, 1_000)
    args.num_eval = min(args.num_eval, 1_000)
    args.epochs = min(args.epochs, 5)
    args.n_timesteps = min(args.n_timesteps, 10)


def main() -> None:
    args = parse_args()
    maybe_apply_quick_mode(args)

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = choose_device(args.device)
    print(f"Using device: {device}")

    data_sampler, actions = build_dataset(args, device)

    gaussian_mean, gaussian_cov = fit_full_gaussian(actions)
    rng = np.random.default_rng(args.seed + 1)
    gaussian_samples = rng.multivariate_normal(
        mean=gaussian_mean,
        cov=gaussian_cov,
        size=args.num_eval,
    ).astype(np.float32)

    diffusion_agent = train_diffusion_policy(args, data_sampler, device)
    diffusion_samples = sample_diffusion(diffusion_agent, args.num_eval, device)

    metrics = {
        "Gaussian MLE": compute_region_metrics(gaussian_samples, args),
        "Diffusion BC": compute_region_metrics(diffusion_samples, args),
    }

    figure_path = plot_results(actions, gaussian_samples, diffusion_samples, metrics, args)
    save_metrics(metrics, args)

    print("\nRegion masses")
    for policy, policy_metrics in metrics.items():
        print(
            f"{policy:>13s} | "
            f"S1={policy_metrics['safe_mode_1_mass']:.4f} "
            f"S2={policy_metrics['safe_mode_2_mass']:.4f} "
            f"bridge={policy_metrics['unsafe_bridge_mass']:.4f} "
            f"midpoint={policy_metrics['midpoint_set_mass']:.4f} "
            f"PL_bound={policy_metrics['prekopaleindler_half_bound']:.4f}"
        )

    print(f"\nSaved figure: {figure_path}")
    print(f"Saved metrics under: {os.path.abspath(args.output_dir)}")


if __name__ == "__main__":
    main()
