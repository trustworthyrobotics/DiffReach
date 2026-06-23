import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def visualize_flowpipe_xy(
    times, lowers=None, uppers=None, trajs=None,
    x_idx=0, y_idx=1,
    file_name="flowpipe_xy.png",
    draw_boxes=True, print_boxes=False, aggregate_partitions=True,
    draw_traj=False, stride=5,
):
    """
    Draw a 2D projection of the flowpipe (x vs y).

    times: 1D array of time samples (same length as lowers[1]).
    lowers, uppers: array-like of shape (M, N, 2) for x and y bounds.
    segments: optional list of dicts with 't0' and 't1' for coarse per-step boxes.
    """

    box_facealpha=0.5
    box_edgecolor="C0"
    box_linewidth=0.6
    traj_alpha=0.8
    traj_linewidth=0.8
    figsize=(7.5,5.5)
    fig, ax = plt.subplots(1,1, figsize=figsize)

    if draw_boxes:
        assert lowers.ndim == 3 and uppers.ndim == 3, "This helper targets 2D state (x,y)."
        times = np.asarray(times)
    
        n_partitions_ori = n_partitions = lowers.shape[0]
        lowers = lowers[:, :, [x_idx, y_idx]]
        uppers = uppers[:, :, [x_idx, y_idx]]

        if aggregate_partitions:
            # Merge all partitions into one big box per time step
            lowers = np.min(lowers, axis=0, keepdims=True)  # (1, N, 2)
            uppers = np.max(uppers, axis=0, keepdims=True)  # (1, N, 2)
            n_partitions = 1

        idxs = range(0, len(times), max(1, int(stride)))
        for i in idxs:
            for j in range(n_partitions):  # over splits
                x_lo, y_lo = float(lowers[j,i,0]), float(lowers[j,i,1])
                x_up, y_up = float(uppers[j,i,0]), float(uppers[j,i,1])
                w = float(x_up - x_lo)
                h = float(y_up - y_lo)
                if w < 0 or h < 0:
                    continue
                if print_boxes:
                    print(f"lbox at t={times[i]:.3f}: x=[{x_lo:.4f}, {x_up:.4f}], y=[{y_lo:.4f}, {y_up:.4f}]")
                rect = Rectangle((x_lo, y_lo), w, h,
                                facecolor=box_edgecolor, alpha=box_facealpha,
                                edgecolor=box_edgecolor, linewidth=box_linewidth)
                ax.add_patch(rect)

        # after creating all rects, before save/tight_layout:
        x_min = float(np.min(lowers[:,:,0]))
        x_max = float(np.max(uppers[:,:,0]))
        y_min = float(np.min(lowers[:,:,1]))
        y_max = float(np.max(uppers[:,:,1]))

        # add a small margin
        dx = 0.02 * max(1e-12, (x_max - x_min))
        dy = 0.02 * max(1e-12, (y_max - y_min))
        ax.set_xlim(x_min - dx, x_max + dx)
        ax.set_ylim(y_min - dy, y_max + dy)
    if draw_traj:
        assert trajs is not None, "To draw_traj, must provide trajs."
        assert trajs.ndim == 3, "Expected trajs of shape (B, N, D)."
        assert trajs.shape[1] == len(times), "trajs time length must match times length."
        for b in range(trajs.shape[0]):
            ax.plot(trajs[b, :, x_idx], trajs[b, :, y_idx], 
                    alpha=traj_alpha, linewidth=traj_linewidth)

    ax.set_title(f"TM Flowpipe in state space (x–y) (Horizon {times[-1]:.2f})")
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_name)
    print(f"Saved figure to {file_name}")
    return fig, ax

def visualize_flowpipe_time(
    times, lowers=None, uppers=None, trajs=None,
    state_idx=0,
    file_name="flowpipe_t_x.png",
    draw_boxes=True, print_boxes=False, aggregate_partitions=True,
    draw_traj=False,
    stride=5,
):
    """
    Draw a time-vs-state projection of the flowpipe (t vs x[state_idx]).

    times: 1D array of time samples (length N)
    lowers, uppers: array-like of shape (M, N, D) for per-partition, per-time-step bounds.
                    We'll use [:, :, state_idx] to obtain (M, N) envelopes.
    state_idx: which state dimension to visualize against time.
    aggregate_partitions: if True, min/max over partitions per time step to a single band.
    """

    box_facealpha=0.5
    box_edgecolor="C0"
    box_linewidth=0.6
    traj_alpha=0.8
    traj_linewidth=0.8
    figsize=(7.5,5.5)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    if draw_boxes:
        assert lowers.ndim == 3 and uppers.ndim == 3, "Expected lowers/uppers of shape (M, N, D)."
        times = np.asarray(times)
        assert times.ndim == 1 and len(times) == lowers.shape[1], "times length must match N."

        n_partitions_ori, n_steps, _D = lowers.shape

        # Slice the chosen state dimension -> (M, N)
        low_1d = lowers[:, :, state_idx]
        up_1d  = uppers[:, :, state_idx]

        n_partitions = n_partitions_ori
        if aggregate_partitions:
            # Merge all partitions into one band per time step: (1, N)
            low_1d = np.min(low_1d, axis=0, keepdims=True)
            up_1d  = np.max(up_1d,  axis=0, keepdims=True)
            n_partitions = 1

        # Precompute half-widths around each time sample to form contiguous bands
        # Use centered widths where possible; endpoints borrow neighbor width.
        dt_right = np.empty(n_steps); dt_left = np.empty(n_steps)
        dt_right[:-1] = times[1:] - times[:-1]
        dt_right[-1]  = dt_right[-2] if n_steps > 1 else 0.0
        dt_left[1:]   = times[1:] - times[:-1]
        dt_left[0]    = dt_left[1]  if n_steps > 1 else 0.0
        half_w = 0.5 * (dt_left + dt_right)

        idxs = range(0, n_steps, max(1, int(stride)))
        for i in idxs:
            t_lo = float(times[i] - 0.5 * half_w[i])
            t_hi = float(times[i] + 0.5 * half_w[i])
            w = max(0.0, t_hi - t_lo)
            for j in range(n_partitions):
                y_lo = float(low_1d[j, i])
                y_up = float(up_1d[j, i])
                h = float(y_up - y_lo)
                if h < 0 or w <= 0:
                    continue
                if print_boxes:
                    print(f"t-band at t≈{times[i]:.4f}: y[{state_idx}]=[{y_lo:.6f}, {y_up:.6f}], Δt≈{w:.6f}")
                rect = Rectangle(
                    (t_lo, y_lo), w, h,
                    facecolor=box_edgecolor, alpha=box_facealpha,
                    edgecolor=box_edgecolor, linewidth=box_linewidth
                )
                ax.add_patch(rect)

        # Axis limits with small margins
        t_min, t_max = float(times[0]), float(times[-1])
        y_min = float(np.min(low_1d))
        y_max = float(np.max(up_1d))
        dtm = 0.02 * max(1e-12, (t_max - t_min))
        dym = 0.02 * max(1e-12, (y_max - y_min))
        ax.set_xlim(t_min - dtm, t_max + dtm)
        ax.set_ylim(y_min - dym, y_max + dym)
    if draw_traj:
        assert trajs is not None, "To draw_traj, must provide trajs."
        assert trajs.ndim == 3, "Expected trajs of shape (B, N, D)."
        assert trajs.shape[1] == len(times), "trajs time length must match times length."
        for b in range(trajs.shape[0]):
            ax.plot(times, trajs[b, :, state_idx],
                    alpha=traj_alpha, linewidth=traj_linewidth)

    ax.set_title(f"TM Flowpipe (time vs x[{state_idx}]) (Horizon {times[-1]:.2f})")
    ax.set_xlabel("time"); ax.set_ylabel(f"x[{state_idx}]")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(file_name)
    print(f"Saved figure to {file_name}")
    return fig, ax
