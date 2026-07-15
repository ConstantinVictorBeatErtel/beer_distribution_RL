"""Eval-only: force greedy (argmax) action selection for any regime.

Mirrors IPPOTrainer.evaluate cost accounting but never uses
`greedy=not signaling`. Does not modify training, rewards, or env dynamics.
"""

from __future__ import annotations

import numpy as np
import torch


def evaluate_matched_deterministic(
    trainer,
    *,
    n_episodes: int | None = None,
    seed: int | None = None,
) -> dict[str, float]:
    """Deterministic eval with the same seed offset as IPPOTrainer.evaluate."""
    n_episodes = n_episodes or trainer.cfg.eval_episodes
    seed = trainer.cfg.seed + 10_000 if seed is None else seed
    costs: list[float] = []

    for ep in range(n_episodes):
        states = trainer.core.reset(seed + ep)
        done = False
        sys_acc = 0.0
        steps = 0
        while not done:
            orders = {}
            signals = {} if trainer.signaling else None
            with torch.no_grad():
                for r in trainer.roles:
                    o = torch.as_tensor(
                        trainer._obs(states, r, trainer.core), device=trainer.device
                    ).unsqueeze(0)
                    a, _, _ = trainer._policy_act(r, o, greedy=True)
                    if trainer.signaling:
                        row = a.squeeze(0).cpu().numpy().astype(int)
                        orders[r] = trainer._decode_order(int(row[0]), states[r])
                        assert signals is not None
                        signals[r] = trainer._decode_signal(
                            states[r], int(row[1]), int(row[2]), int(row[3])
                        )
                    else:
                        orders[r] = trainer._decode_order(int(a.item()), states[r])
            states, _, done, info = trainer.core.step(orders, signals)
            sys_acc += info.system_cost
            steps += 1
        costs.append(sys_acc / max(steps, 1))

    return {
        "eval/mean_system_cost": float(np.mean(costs)),
        "eval/std_system_cost": float(np.std(costs)),
        "eval/n_episodes": float(n_episodes),
        "eval/greedy": 1.0,
    }
