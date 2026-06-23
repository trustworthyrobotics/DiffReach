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

from src.utils.load import (
    load_yaml,
    load_controller,
)

from src.systems import CT_Ctl_Sys, DT_Ctl_Sys
from src.reachability import CT_Ctl_Reach, DT_Ctl_Reach
from src.reachability_baseline import CT_Ctl_Reach_immrax
from run_dyn import Launcher_DynSys

class Launcher_CtlSys(Launcher_DynSys):
    def __init__(
        self,
        cfg: Dict[str, Any],
        sim: bool = True,
        ver: bool = True,
        debug: bool = False,
    ) -> None:
        self.reference_dim = int(cfg.get("reference_dim", 0))
        self.reference_seq = cfg.get("reference_seq")
        super().__init__(cfg, sim=sim, ver=ver, debug=debug)

    def _load_reference_seq(self) -> None:
        if self.reference_seq is None:
            if self.reference_dim == 0:
                return
            self.reference_seq = jnp.zeros((self.control_steps, self.reference_dim))
            return

        reference_seq = jnp.asarray(self.reference_seq)
        expected_shape = (self.control_steps, self.reference_dim)
        if reference_seq.shape != expected_shape:
            raise ValueError(
                "reference_seq shape mismatch: expected "
                f"{expected_shape}, got {reference_seq.shape}."
            )
        self.reference_seq = reference_seq

    def _build_system(self):
        assert self.D_a > 0, "This launcher is for systems with control."
        if self.discrete:
            self.step_size: int = int (self.cfg["step_size"])
            assert self.step_size == 1, "Discrete-time dynamics only support step_size=1."
            self.dynamics_class = DT_Ctl_Sys
            self.reach_class = DT_Ctl_Reach
        else:
            self.step_size: float = float(self.cfg["step_size"])
            self.dynamics_class = CT_Ctl_Sys
            if self.cfg.get("engine", "") == "immrax":
                print("[info] Using immrax-based reachability engine.")
                self.reach_class = CT_Ctl_Reach_immrax
                self.output_dir += "_immrax"
            else:
                self.reach_class = CT_Ctl_Reach

        self.controller_path: str = str(self.cfg["controller"])
        self.n_steps_per_control: int = int(self.cfg["n_steps_per_control"])
        self.control_steps: int = int(round(self.n_total_steps / self.n_steps_per_control))
        print(f"[info] Building reach function with n_total_steps={self.n_total_steps}, "
              f"step_size={self.step_size}, control_steps={self.control_steps}, n_steps_per_control={self.n_steps_per_control} "
              f"and B={self.B}...")
        self.model = load_controller(self.controller_path)

        if self.ver:
            self.system = self.reach_class(
                rhs=self.rhs,
                state_dim=self.D_s,
                action_dim=self.D_a,
                nn_dyn=self.nn_dyn,
                controller=self.model,
                n_steps_per_control=self.n_steps_per_control,
                step_size=self.step_size,
                init_remainder=self.init_remainder,
                frr_rounds=self.frr_rounds,
                frr_stop_ratio=self.frr_stop_ratio,
                sr_window_size=self.sr_window_size,
                reference_dim=self.reference_dim
            )
        elif self.sim:
            self.system = self.dynamics_class(
                rhs=self.rhs,
                state_dim=self.D_s,
                action_dim=self.D_a,
                controller=self.model,
                n_steps_per_control=self.n_steps_per_control,
                step_size=self.step_size,
                reference_dim=self.reference_dim
            )

        self._load_reference_seq()

    def _simulate(self):
        return super()._simulate(reference_seq = self.reference_seq) 

    def _verify(self):
        return super()._verify(reference_seq = self.reference_seq)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument("--sim", action="store_true", help="Run trajectory simulation.")
    parser.add_argument("--ver", action="store_true", help="Run reachability analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode.")
    parser.add_argument("--load", action="store_true", help="Load results from output directory instead of running.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    launcher = Launcher_CtlSys(cfg, sim=args.sim, ver=args.ver, debug=args.debug)

    if args.load:
        launcher.replay()
    else:
        launcher.run()

if __name__ == "__main__":
    main()
