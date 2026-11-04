"""Metrics shared with DeepONet run 11597463, preserved for comparison."""
import numpy as np
from scipy.ndimage import distance_transform_edt

def relative_norm(error, truth):
    return float(np.linalg.norm(error) / max(float(np.linalg.norm(truth)), 1e-12))

def design_metrics(pred, truth, mask, y, z):
    valid = ~mask
    edge = valid & (distance_transform_edt(valid) <= 3)
    # Forward differences on the nonuniform physical coordinate grid (units h).
    # Both endpoints must be valid; gradients never cross the excluded trace.
    gy_valid = valid[:, 1:] & valid[:, :-1]
    gz_valid = valid[1:, :] & valid[:-1, :]
    dpy = np.diff(pred, axis=2) / np.diff(y)[None, None, :]
    dty = np.diff(truth, axis=2) / np.diff(y)[None, None, :]
    dpz = np.diff(pred, axis=1) / np.diff(z)[None, :, None]
    dtz = np.diff(truth, axis=1) / np.diff(z)[None, :, None]
    grad_error = np.concatenate(((dpy - dty)[:, gy_valid].ravel(),
                                 (dpz - dtz)[:, gz_valid].ravel()))
    grad_truth = np.concatenate((dty[:, gy_valid].ravel(), dtz[:, gz_valid].ravel()))
    return {
        "relative_l2": relative_norm((pred - truth)[:, valid], truth[:, valid]),
        "edge_relative_l2": relative_norm((pred - truth)[:, edge], truth[:, edge]),
        "gradient_relative_l2": relative_norm(grad_error, grad_truth),
        "masked_mse": float(np.mean((pred[:, valid] - truth[:, valid]) ** 2)),
    }

