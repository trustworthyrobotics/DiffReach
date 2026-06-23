"""
Usage:
  python run.py path/to/config.yaml
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path
from typing import Any, Dict, List

import jax
# jax.config.update('jax_platforms', 'cpu')
jax.config.update("jax_enable_x64", True)
jax.config.update("jax_default_matmul_precision", "highest")

import jax.numpy as jnp

from src.utils.box_set import prepare_initial_sets, calculate_volume
from src.utils.load import (
    load_yaml,
    load_analytic_dynamics,
    load_nn_dynamics
)
from src.utils.vis import visualize_flowpipe_time, visualize_flowpipe_xy
from src.systems import CT_Dyn_Sys, DT_Dyn_Sys
from src.reachability import CT_Dyn_Reach, DT_Dyn_Reach
from src.reachability_baseline import CT_Dyn_Reach_immrax
import src.settings as settings

class Launcher_DynSys:
    """High-level orchestrator for a single YAML-driven run."""

    def __init__(
        self,
        cfg: Dict[str, Any],
        sim: bool = True,
        ver: bool = True,
        debug: bool = False,
    ) -> None:
        assert sim or ver, "At least one of --sim or --ver must be specified."
        self.sim = sim
        self.ver = ver
        self.debug = debug
        settings.update_config({"DEBUG_LOG": debug})

        self.cfg = cfg
        self.D_s: int = int(cfg["state_dim"])
        self.D_a: int = int(cfg["action_dim"])
        self.D_total: int = self.D_s + self.D_a
        self.nn_dyn: bool = bool(cfg.get("nn_dyn", False))
        self.var_names: List[str] = list(cfg["variable_names"])

        self.n_total_steps: int = int(cfg["n_total_steps"])
        self.discrete: bool = bool(cfg.get("discrete", False))
        self.step_size: float = float(cfg["step_size"])

        init_remainder = cfg.get("init_remainder", 1e-1)
        if isinstance(init_remainder, list):
            self.init_remainder: jnp.array = jnp.array([float(x) for x in init_remainder])
        else:
            self.init_remainder: jnp.array = jnp.array([float(init_remainder)])
        self.frr_rounds: int = int(cfg.get("frr_rounds", 5))
        self.frr_stop_ratio: float = float(cfg.get("frr_stop_ratio", 0.95))
        self.sr_window_size: int = int(cfg.get("sr_window_size", 100))

        self.dyn_path: str = str(cfg["dynamics"])
        if "initial_set" in cfg:
            self.initial_set = jnp.array(cfg["initial_set"])
        else:
            assert "initial_x" in cfg and "eps" in cfg
            init_x = jnp.array(cfg["initial_x"])
            eps = jnp.array(cfg["eps"])
            self.initial_set = jnp.stack([init_x - eps, init_x + eps], axis=1)  # [D_total,2]

        self.splits_cfg: Dict[int,int] = dict(cfg.get("splits", {}))

        # prepare initial sets (handles dict→dense per-dim splits)
        self.x0_lo, self.x0_hi = prepare_initial_sets(
            initial_set=self.initial_set,
            splits_cfg=self.splits_cfg,
            D_total=self.D_total,
        )
        self.B = int(self.x0_lo.shape[0])

        # parse visualization config
        self.parse_vis_cfg()

        # build reach function once
        if self.nn_dyn:
            self.rhs = load_nn_dynamics(self.dyn_path, self.D_s, self.D_a, self.discrete)
        else:
            self.rhs = load_analytic_dynamics(self.dyn_path, self.D_s, self.D_a, self.discrete)
        self._build_system()

        self.ts = None
        self.trajs = None
        self.lowers = None
        self.uppers = None

        print(f"[info] initial-set splitting -> B={self.B} partitions.")

    def _build_system(self):
        assert self.D_a == 0, "This launcher is for dynamics without control."
        if self.discrete:
            self.step_size: int = int (self.cfg["step_size"])
            assert self.step_size == 1, "Discrete-time dynamics only support step_size=1."
            self.dynamics_class = DT_Dyn_Sys
            self.reach_class = DT_Dyn_Reach
        else:
            self.step_size: float = float(self.cfg["step_size"])
            self.dynamics_class = CT_Dyn_Sys
            if self.cfg.get("engine", "") == "immrax":
                print("[info] Using immrax-based reachability engine.")
                self.reach_class = CT_Dyn_Reach_immrax
                self.output_dir += "_immrax"
            else:
                self.reach_class = CT_Dyn_Reach

        print(f"[info] Building reach function with step_size={self.step_size}, "
              f"n_total_steps={self.n_total_steps}, and B={self.B}...")
        if self.ver:
            self.system = self.reach_class(
                rhs=self.rhs,
                state_dim=self.D_s,
                nn_dyn=self.nn_dyn,
                step_size=self.step_size,
                init_remainder=self.init_remainder,
                frr_rounds=self.frr_rounds,
                frr_stop_ratio=self.frr_stop_ratio,
                sr_window_size=self.sr_window_size
            )
        elif self.sim:
            self.system = self.dynamics_class(
                rhs=self.rhs, state_dim=self.D_s, step_size=self.step_size
            )

    def parse_vis_cfg(self):
        vis_cfg = dict(self.cfg.get("visualization", {}))
        self.vis_stride: int = int(vis_cfg.get("vis_stride", 10000))
        self.vis_agg: bool = bool(vis_cfg.get("aggregate_partitions", True))
        self.draw_dims: List[int] = list(vis_cfg.get("draw_dims", []))

        self.output_dir: str = str(self.cfg["output_dir"])
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)

    def _simulate(self, *args, **kwargs):
        # warmup
        x0_lo = self.initial_set[:, 0]
        x0_hi = self.initial_set[:, 1]
        n_samples = self.cfg.get("sim_n_samples", 32)
        t0 = time.time()
        if self.debug:
            sim_func = self.system.simulate
            with jax.disable_jit():
                ts, trajs = sim_func(
                    x0_lo, x0_hi, self.n_total_steps,
                    n_samples=n_samples,
                    sampler="random",
                    key =jax.random.PRNGKey(0),
                    *args, **kwargs
                )
        else:
            sim_func = jax.jit(self.system.simulate, static_argnames=("n_total_steps", "n_samples", "sampler"))
            ts, trajs = sim_func(
                x0_lo, x0_hi, self.n_total_steps,
                n_samples=n_samples,
                sampler="random",
                key =jax.random.PRNGKey(0),
                *args, **kwargs
            )
        jax.block_until_ready(trajs)
        t1 = time.time()
        print(f"[warmup] {t1-t0:.3f}s")

        if not self.debug:
            # timed
            t2 = time.time()
            ts, trajs = sim_func(
                x0_lo, x0_hi, self.n_total_steps,
                n_samples=n_samples,
                sampler="random",
                key =jax.random.PRNGKey(0),
                *args, **kwargs
            )
            jax.block_until_ready(trajs)
            t3 = time.time()
            print(f"[after-JIT] {t3-t2:.3f}s")
        self.ts = ts
        self.trajs = trajs
        return ts, trajs

    def _verify(self, *args, **kwargs):
        # warmup
        t0 = time.time()
        if self.debug:
            verify_func = self.system.verify
            with jax.disable_jit():
                ts, L, U, _, shrinked = verify_func(self.x0_lo, self.x0_hi, self.n_total_steps, *args, **kwargs)
        else:
            verify_func = jax.jit(self.system.verify, static_argnames=("n_total_steps",))
            ts, L, U, _, shrinked = verify_func(self.x0_lo, self.x0_hi, self.n_total_steps, *args, **kwargs)
        jax.block_until_ready(U)
        t1 = time.time()
        print(f"[warmup] {t1-t0:.3f}s")

        if not self.debug:
            # timed
            t2 = time.time()
            ts, L, U, _, shrinked = verify_func(self.x0_lo, self.x0_hi, self.n_total_steps, *args, **kwargs)
            jax.block_until_ready(U)
            t3 = time.time()
            print(f"[after-JIT] {t3-t2:.3f}s")

        self.ts = ts
        self.lowers = L
        self.uppers = U
        self.shrinked = shrinked
        return ts, L, U

    def _report(self) -> None:
        print("Total steps: ", self.n_total_steps)
        agg_L = jnp.min(self.lowers, axis=0)[None]  # [1, n_total_steps+1, D]
        agg_U = jnp.max(self.uppers, axis=0)[None]  # [1, n_total_steps+1, D]

        total_dim = agg_L.shape[-1]

        for i in range(total_dim):
            name = self.var_names[i]
            lo_i = float(agg_L[0, -1, i])
            hi_i = float(agg_U[0, -1, i])
            print(f"{name}(T) ∈ [{lo_i}, {hi_i}]")
        vol = float(calculate_volume(self.lowers[..., :self.D_s], self.uppers[..., :self.D_s], union_init=True))
        print(f"vag volume = {vol:.6e}")

        # Picard initial shrinkage report
        picard_shrinked_rate =  self.shrinked.sum() / self.shrinked.size
        if picard_shrinked_rate < 1:
            print(f"[info] Picard shrinked {picard_shrinked_rate*100:.2f}% of partitions across all steps. The result might be very conservative or unsound.")

        if self.trajs is not None:
            # soundness check
            trajs = self.trajs[..., :total_dim]  # [B,T,D]
            tol = 1e-5
            violations = jnp.logical_or(trajs < agg_L - tol, trajs > agg_U + tol)  # [B,T,D]
            violations = violations[:, 1:, :]  # ignore t=0
            for i in range(total_dim):
                name = self.var_names[i]
                n_viol = int(jnp.sum(violations[:, :, i]))
                if n_viol > 0:
                    print(f"[warning] {name}: {n_viol} violations from sampled trajectories.")
                    #  Indices: {jnp.nonzero(violations[:, :, i])}

    def _save_results(self) -> None:
        """Save flowpipe data to output_dir."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.ver:
            jnp.savez(
                f"{self.output_dir}/flowpipe_ver.npz",
                ts=self.ts,
                lowers=self.lowers,
                uppers=self.uppers,
                shrinked=self.shrinked,
            )
        if self.sim:
            jnp.savez(
                f"{self.output_dir}/trajectories_sim.npz",
                ts=self.ts,
                trajs=self.trajs,
            )
    
    def _load_results(self) -> None:
        """Load flowpipe data from output_dir."""
        if self.ver:
            data = jnp.load(f"{self.output_dir}/flowpipe_ver.npz")
            self.ts = data["ts"]
            self.lowers = data["lowers"]
            self.uppers = data["uppers"]
            self.shrinked = data["shrinked"]
        if self.sim:
            data = jnp.load(f"{self.output_dir}/trajectories_sim.npz")
            self.ts = data["ts"]
            self.trajs = data["trajs"]

    def _visualize(self) -> None:
        """Loop over dims and emit per-dim flowpipe plots."""
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        total_dim = self.lowers.shape[-1] if self.ver else self.trajs.shape[-1]
        for idx in range(total_dim):
            name = self.var_names[idx]
            # if idx != 0:
            #     continue
            outfile = f"{self.output_dir}/{name}_{self.n_total_steps}_{self.B}{'_agg' if self.vis_agg else ''}.png"
            visualize_flowpipe_time(
                times=self.ts,
                lowers=self.lowers if self.ver else None,
                uppers=self.uppers if self.ver else None,
                trajs=self.trajs if self.sim else None,
                state_idx=idx,
                file_name=outfile,
                print_boxes=False,
                draw_boxes=self.ver,
                aggregate_partitions=self.vis_agg,
                stride=self.vis_stride,
                draw_traj=self.sim,
            )
        if self.draw_dims:
            assert len(self.draw_dims) == 2, "draw_dims must have length 2."
            d0, d1 = self.draw_dims
            name0 = self.var_names[d0]
            name1 = self.var_names[d1]
            outfile = f"{self.output_dir}/{name0}_{name1}_{self.n_total_steps}_{self.B}{'_agg' if self.vis_agg else ''}.png"
            visualize_flowpipe_xy(
                times=self.ts,
                lowers=self.lowers if self.ver else None,
                uppers=self.uppers if self.ver else None,
                trajs=self.trajs if self.sim else None,
                x_idx=d0,
                y_idx=d1,
                file_name=outfile,
                print_boxes=False,
                draw_boxes=self.ver,
                aggregate_partitions=self.vis_agg,
                stride=self.vis_stride,
                draw_traj=self.sim,
            )

    def run(self) -> None:
        if self.sim:
            print(f"[info] Simulating trajectories...")
            self._simulate()
            print(f"[OK] Sampled {self.trajs.shape[0]} trajectories of length {self.trajs.shape[1]}.")
        if self.ver:
            print(f"[info] Verifying reachability...")
            self._verify()
            self._report()
        self._save_results()
        self._visualize()

    def replay(self) -> None:
        print(f"[info] Loading results from {self.output_dir}...")
        self._load_results()
        self._report()
        self._visualize()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="Path to YAML config.")
    parser.add_argument("--sim", action="store_true", help="Run trajectory simulation.")
    parser.add_argument("--ver", action="store_true", help="Run reachability analysis.")
    parser.add_argument("--debug", action="store_true", help="Enable debugging output.")
    parser.add_argument("--load", action="store_true", help="Load results from output directory instead of running.")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    launcher = Launcher_DynSys(cfg, sim=args.sim, ver=args.ver, debug=args.debug)
    if args.load:
        launcher.replay()
    else:
        launcher.run()


if __name__ == "__main__":
    main()
