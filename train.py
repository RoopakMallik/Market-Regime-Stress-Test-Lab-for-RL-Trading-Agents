"""
Train RL models (DQN, PPO, A2C) for each market scenario.

Quick start:
  1. Open config.py and set TRAINING_MODE = "quick" | "dev" | "final"
  2. Run:  .venv/bin/python train.py

Command-line overrides (optional):
  .venv/bin/python train.py --mode dev
  .venv/bin/python train.py --timesteps 20000
  .venv/bin/python train.py --scenario bull_run
  .venv/bin/python train.py --agent PPO
  .venv/bin/python train.py --scenario recession --agent DQN --timesteps 10000
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone

from stable_baselines3 import A2C, DQN, PPO

from config import (
    AGENTS,
    INITIAL_BALANCE,
    MODELS_DIR,
    RANDOM_SEED,
    TICKERS,
    TIMESTEPS,
    TIMESTEPS_BY_MODE,
    TRAIN_RATIO,
    TRAINING_MODE,
    TRANSACTION_COST_BPS,
)
from features import fit_all_scalers
from scenarios import SCENARIOS
from trading_env import RandomTickerEnv

# Map short names to Stable Baselines3 classes.
AGENT_CLASSES = {"DQN": DQN, "PPO": PPO, "A2C": A2C}


def resolve_timesteps(mode: str | None, timesteps: int | None) -> int:
    """Pick training length: CLI --timesteps beats --mode beats config.py."""
    if timesteps is not None:
        return timesteps
    if mode is not None:
        if mode not in TIMESTEPS_BY_MODE:
            raise ValueError(f"Unknown mode '{mode}'. Use: {list(TIMESTEPS_BY_MODE)}")
        return TIMESTEPS_BY_MODE[mode]
    return TIMESTEPS


def write_config_json(agent_name: str, scenario_name: str, timesteps: int) -> None:
    """Save a small JSON record next to each model (for reproducibility)."""
    payload = {
        "agent": agent_name,
        "scenario": scenario_name,
        "tickers": TICKERS,
        "initial_balance": INITIAL_BALANCE,
        "train_ratio": TRAIN_RATIO,
        "timesteps": timesteps,
        "training_mode": TRAINING_MODE,
        "transaction_cost_bps": TRANSACTION_COST_BPS,
        "normalization": "zscore",
        "observation_lag": 1,
        "seed": RANDOM_SEED,
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    path = MODELS_DIR / f"{agent_name}_{scenario_name}_config.json"
    path.write_text(json.dumps(payload, indent=2))


def train_all(
    timesteps: int | None = None,
    scenarios: list[str] | None = None,
    agents: list[str] | None = None,
) -> None:
    """
    Train models and save to models/.

    Default: all 5 scenarios × 3 agents = 15 models.
    Pass scenarios=[...] or agents=[...] to train a subset only.
    """
    timesteps = timesteps or TIMESTEPS
    scenarios = scenarios or list(SCENARIOS.keys())
    agents = agents or AGENTS

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Training plan: {len(scenarios)} scenarios × {len(agents)} agents")
    print(f"Timesteps per model: {timesteps:,}")
    print(f"Starting portfolio: ${INITIAL_BALANCE:,}")

    for scenario_name in scenarios:
        if scenario_name not in SCENARIOS:
            raise ValueError(f"Unknown scenario '{scenario_name}'. Options: {list(SCENARIOS)}")

        # Fit z-score scalers on the train split for every ticker in this scenario.
        print(f"\n=== Fitting scalers for scenario: {scenario_name} ===")
        fit_all_scalers(scenario_name, TICKERS)

        for agent_name in agents:
            if agent_name not in AGENT_CLASSES:
                raise ValueError(f"Unknown agent '{agent_name}'. Options: {list(AGENT_CLASSES)}")

            print(f"Training {agent_name} on {scenario_name} ({timesteps:,} steps)...")

            # Each episode randomly picks a ticker — multi-ticker training, one model per scenario.
            env = RandomTickerEnv(scenario_name, split="train", seed=RANDOM_SEED)
            model = AGENT_CLASSES[agent_name](
                "MlpPolicy",
                env,
                verbose=1,
                seed=RANDOM_SEED,
            )
            model.learn(total_timesteps=timesteps)

            save_stem = MODELS_DIR / f"{agent_name}_{scenario_name}"
            model.save(str(save_stem))
            write_config_json(agent_name, scenario_name, timesteps)
            print(f"  Saved {save_stem}.zip")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Market Regime Stress-Test Lab RL models.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python train.py                      # uses TRAINING_MODE from config.py
  python train.py --mode dev           # 30,000 steps per model
  python train.py --mode final         # 100,000 steps per model
  python train.py --timesteps 20000    # custom step count
  python train.py --scenario bull_run --agent PPO
        """,
    )
    parser.add_argument(
        "--mode",
        choices=list(TIMESTEPS_BY_MODE.keys()),
        help='Training length preset: quick (5k), dev (30k), final (100k).',
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        help="Exact training steps per model (overrides --mode and config.py).",
    )
    parser.add_argument(
        "--scenario",
        choices=list(SCENARIOS.keys()),
        help="Train only this scenario (default: all five).",
    )
    parser.add_argument(
        "--agent",
        choices=AGENTS,
        help="Train only this agent type (default: DQN, PPO, A2C).",
    )
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    steps = resolve_timesteps(args.mode, args.timesteps)
    scenario_list = [args.scenario] if args.scenario else None
    agent_list = [args.agent] if args.agent else None
    train_all(timesteps=steps, scenarios=scenario_list, agents=agent_list)
