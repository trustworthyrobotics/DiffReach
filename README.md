# DiffReach

Reachability Engine for:

**Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers**  

RSS 2026

#### [[Project Page]](https://trustworthyrobotics.github.io/diffreach_site/) [[arXiv]](https://arxiv.org/abs/2605.25346) [[Video]](https://www.youtube.com/watch?v=E6zeKt26TcM)

---

## Overview

DiffReach is a GPU-parallel and differentiable reachability framework for continuous- and discrete-time systems with analytical and neural-network dynamics and controllers.


## Installation
- We use [uv](https://docs.astral.sh/uv/), a fast Python package and project manager to manage dependencies and virtual environments.
A detailed uv tutorial is available [here](https://docs.astral.sh/uv/guides/install-python/#getting-started).

- You may run following commands to set up the environment and install dependencies. It will install JAX 0.8.1 with CUDA 12.
```bash
uv venv --python=3.12
source .venv/bin/activate
uv pip install -r pyproject.toml
uv pip install --no-deps linrax
```

## Usage

### Run reachability analysis

We offer scripts to run reachability analysis for various settings like continuous- and discrete-time systems with plain dynamics (analytical or neural), neural controllers, and planning problems. `--sim` and `--ver` flags will run the simulation and verification, respectively. You can also run them separately.

- Continuous-time dynamics of Van der Pol oscillator:
```bash
python run_dyn.py config/ct_dyn/van_der_pol.yaml --sim --ver
```

- Continuous-time control of cartpole from ARCH-COMP 2025:
```bash
python run_ctl.py config/ct_ctl/cartpole.yaml --sim --ver
```

- Continuous-time random neural dynamics with zero open-loop action plan:
```bash
python run_plan.py config/ct_plan/test.yaml --sim --ver
```

- Discrete-time random neural dynamics:
```bash
python run_dyn.py config/dt_dyn/test.yaml --sim --ver
```

- Discrete-time control with random neural dynamics and controller:
```bash
python run_ctl.py config/dt_ctl/test.yaml --sim --ver
```

- Discrete-time random neural dynamics with zero open-loop action plan:
```bash
python run_plan.py config/dt_plan/test.yaml --sim --ver
```

### Use reachability analysis as a component in Python

See the minimal cartpole notebooks for Python-first examples without YAML:

- [`cartpole_reach.ipynb`](cartpole_reach.ipynb): build a controlled cartpole reachability pipeline directly in Python and print final per-dimension ranges.
- [`cartpole_pgd.ipynb`](cartpole_pgd.ipynb): differentiate through reachability and use PGD to search for a nominal initial point with a smaller reachable tube.

## Citation

```bibtex
@inproceedings{shen2026diffreach,
  title={Parallel Differentiable Reachability for Learning and Planning with Certified Neural Dynamics and Controllers},
  author={Keyi Shen and Glen Chou},
  booktitle={Proceedings of Robotics: Science and Systems (RSS)},
  year={2026}
}
