"""
Usage:
  python run.py path/to/config.yaml
"""

from __future__ import annotations
import argparse
from typing import Any, Dict, List

import jax
import jax.numpy as jnp
# jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

from src.utils.load import load_yaml
from src.systems import DT_Plan_Sys, CT_Plan_Sys
from src.reachability import DT_Plan_Reach, CT_Plan_Reach
from src.reachability_baseline import DT_Plan_Reach_CROWN
from run_dyn import Launcher_DynSys

class Launcher_PlanSys(Launcher_DynSys):
    """High-level orchestrator for a single YAML-driven run."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        sim: bool = True,
        ver: bool = True,
        debug: bool = False,
    ) -> None:
        self.n_plan = 1
        self.action_seq = cfg.get("action_seq")
        super().__init__(cfg, sim=sim, ver=ver, debug=debug)

    def _load_action_seq(self) -> None:
        n_plan_steps = self.n_total_steps // self.n_steps_per_plan
        if self.action_seq is None:
            self.action_seq = jnp.zeros((self.n_plan, n_plan_steps, self.D_a))
            return

        action_seq = jnp.asarray(self.action_seq)
        expected_shape = (n_plan_steps, self.D_a)
        if action_seq.shape != expected_shape:
            raise ValueError(
                "action_seq shape mismatch: expected "
                f"{expected_shape}, got {action_seq.shape}."
            )
        self.action_seq = action_seq[None, :, :]

    def _build_system(self):
        if self.discrete:
            self.step_size: int = int(self.cfg["step_size"])
            assert self.step_size == 1, "Discrete-time dynamics only support step_size=1."
            self.dynamics_class = DT_Plan_Sys
            if self.cfg.get("engine", "") == "crown":
                print("[info] Using CROWN-based reachability engine.")
                self.reach_class = DT_Plan_Reach_CROWN
                self.output_dir += "_crown"
            else:
                self.reach_class = DT_Plan_Reach
        else:
            self.step_size: float = float(self.cfg["step_size"])
            self.dynamics_class = CT_Plan_Sys
            self.reach_class = CT_Plan_Reach

        self.n_steps_per_plan: int = int(self.cfg["n_steps_per_plan"])
        if self.n_total_steps % self.n_steps_per_plan != 0:
            # increase n_total_steps to be divisible by n_steps_per_plan
            self.n_total_steps += self.n_steps_per_plan - (self.n_total_steps % self.n_steps_per_plan)

        print(f"[info] Building reach function with step_size={self.step_size}, "
              f"n_total_steps={self.n_total_steps}, and B={self.B}...")
        if self.ver:
            self.system = self.reach_class(
                rhs=self.rhs,
                state_dim=self.D_s,
                action_dim=self.D_a,
                nn_dyn=self.nn_dyn,
                n_steps_per_plan=self.n_steps_per_plan,
                step_size=self.step_size,
                sr_window_size=self.sr_window_size
            )
        elif self.sim:
            self.system = self.dynamics_class(
                rhs=self.rhs, state_dim=self.D_s, action_dim=self.D_a, n_steps_per_plan=self.n_steps_per_plan, step_size=self.step_size
            )
        self._load_action_seq()

    def _simulate(self):
        super()._simulate(action_seq = self.action_seq)
        self.trajs = self.trajs.reshape(-1, self.n_total_steps + 1, self.D_s + self.D_a)  # (n_samples * n_plan, T+1, D)
        return 


    def _verify(self):
        super()._verify(action_seq = self.action_seq)
        self.lowers = self.lowers.reshape(-1, self.n_total_steps + 1, self.D_s + self.D_a)  # (B * n_plan, T+1, D)
        self.uppers = self.uppers.reshape(-1, self.n_total_steps + 1, self.D_s + self.D_a)  # (B * n_plan, T+1, D)
        return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument("--sim", action="store_true", help="Run trajectory simulation.")
    parser.add_argument("--ver", action="store_true", help="Run reachability analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging output.")
    parser.add_argument("--load", action="store_true", help="Load results from output directory instead of running.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    launcher = Launcher_PlanSys(cfg, sim=args.sim, ver=args.ver, debug=args.debug)
    if args.load:
        launcher.replay()
    else:
        launcher.run()


if __name__ == "__main__":
    main()
