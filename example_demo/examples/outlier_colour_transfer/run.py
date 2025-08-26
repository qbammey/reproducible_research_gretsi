#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import ot  # POT: Python Optimal Transport


def read_rgb(path: str) -> np.ndarray:
    """
    Read an image as float64 RGB in [0,1]. If it has an alpha channel, drop it.
    """
    img = plt.imread(path).astype(np.float64)
    # Some readers give [0,255], others already [0,1]
    if img.max() > 1.0:
        img = img / 255.0
    if img.ndim == 2:
        # grayscale -> fake RGB
        img = np.stack([img, img, img], axis=-1)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return np.clip(img, 0.0, 1.0)


def pairwise_cost(A: np.ndarray, B: np.ndarray, metric: str) -> np.ndarray:
    """
    Pairwise distance matrix between two (n,3) and (m,3) RGB point clouds.
    metric: 'l2' for Euclidean (as in the original code), 'l1' for Manhattan.
    """
    if metric == 'l2':
        return ot.dist(A, B)  # Euclidean distance
    elif metric == 'l1':
        # Manhattan distance
        return np.sum(np.abs(A[:, None, :] - B[None, :, :]), axis=2)
    else:
        raise ValueError("metric must be 'l2' or 'l1'")


def color_transfer_ot(
    base_img: np.ndarray,
    target_img: np.ndarray,
    downscale: int = 50,
    metric: str = 'l2',
    reg: float = 0.0,
    sinkhorn_max_iter: int = 1000,
) -> np.ndarray:
    """
    Match base_img colors to target_img colors via OT on downscaled images,
    then lift the per-color translation to full resolution by nearest neighbor
    in RGB space (same heuristic as in the original script).

    Parameters
    ----------
    base_img : np.ndarray
        Source image to be stylized, float64 in [0,1], shape (H,W,3).
    target_img : np.ndarray
        Style/target image providing the color distribution, float64 in [0,1], (H,W,3).
    downscale : int
        Downscaled width/height used to build color clouds (default 50 = original code).
        Makes the OT problem tractable: nd = downscale*downscale.
    metric : {'l2','l1'}
        Ground metric for OT. 'l2' (Euclidean) matches the original code.
        'l1' is often a bit more outlier-robust.
    reg : float
        Entropic regularization strength for Sinkhorn. If 0.0, use exact EMD (as original).
        If > 0, use Sinkhorn with this regularization (faster/smoother for larger clouds).
    sinkhorn_max_iter : int
        Maximum iterations for Sinkhorn if reg>0.

    Returns
    -------
    styl_full : np.ndarray
        Stylized image, same shape as base_img, float64 in [0,1].
    """
    H, W, _ = base_img.shape

    # --- downscale both images (like the original code) ---
    base_d = cv2.resize(base_img, (downscale, downscale), interpolation=cv2.INTER_AREA)
    target_d = cv2.resize(target_img, (downscale, downscale), interpolation=cv2.INTER_AREA)
    nd = downscale * downscale

    base_flat = base_d.reshape(nd, 3)
    target_flat = target_d.reshape(nd, 3)
    a = ot.unif(nd)  # uniform masses
    b = ot.unif(nd)

    # --- OT plan between base_d and target_d color clouds ---
    C = pairwise_cost(base_flat, target_flat, metric=metric)
    if reg > 0.0:
        # Entropic regularized OT (smooth, fast for larger nd)
        pi = ot.sinkhorn(a, b, C, reg=reg, numItermax=sinkhorn_max_iter)
    else:
        # Exact EMD (as in the original script)
        pi = ot.emd(a, b, C)

    # --- barycentric projection of base_d colors onto target distribution ---
    base_d_matched = (nd * pi @ target_flat).reshape(base_d.shape)
    base_d_matched = np.clip(base_d_matched, 0.0, 1.0)

    # --- lift to full resolution by nearest neighbor in RGB space (original heuristic) ---
    base_full_flat = base_img.reshape(H * W, 3)
    base_d_flat = base_d.reshape(nd, 3)
    base_dm_flat = base_d_matched.reshape(nd, 3)

    # nearest neighbor downscaled color for each full-res pixel
    D_full_to_low = pairwise_cost(base_full_flat, base_d_flat, metric=metric)
    nn_idx = np.argmin(D_full_to_low, axis=1)

    translations = base_dm_flat - base_d_flat
    styl_full_flat = base_full_flat + translations[nn_idx]
    styl_full_flat = np.clip(styl_full_flat, 0.0, 1.0)
    styl_full = styl_full_flat.reshape(H, W, 3)

    return styl_full


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Apply color style transfer via Optimal Transport (as in the provided script),\n"
            "matching the BASE image's colors to the TARGET image's colors. "
            "The method computes an OT map between downscaled color clouds "
            "and lifts the per-color translation to the full resolution."
        )
    )
    parser.add_argument(
        "--base",
        required=True,
        help="Path to the BASE image (the one to be stylized)."
    )
    parser.add_argument(
        "--target",
        required=True,
        help="Path to the TARGET image (provides the color distribution)."
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to save the stylized output image (e.g., output.jpg)."
    )
    parser.add_argument(
        "--downscale",
        type=int,
        default=50,
        help=(
            "Downscaled width/height used to build color clouds (default: 50). "
            "Larger values give finer color matching but increase memory/time "
            "(~O(downscale^4) for exact EMD)."
        )
    )
    parser.add_argument(
        "--metric",
        choices=["l2", "l1"],
        default="l2",
        help=(
            "Ground metric for OT: 'l2' (Euclidean, matches original) or 'l1' (Manhattan). "
            "L1 can be slightly more robust to outliers."
        )
    )
    parser.add_argument(
        "--reg",
        type=float,
        default=0.0,
        help=(
            "Entropic regularization strength for Sinkhorn. "
            "Set 0.0 to use exact EMD (original behavior). "
            "Set >0.0 (e.g., 1e-2 to 5e-2) for faster/smoother transport on larger problems."
        )
    )
    parser.add_argument(
        "--sinkhorn-max-iter",
        type=int,
        default=1000,
        help="Maximum iterations for Sinkhorn if --reg > 0 (default: 1000)."
    )

    args = parser.parse_args()

    # Load images
    base_img = read_rgb(args.base)
    target_img = read_rgb(args.target)

    # Run style transfer
    styl = color_transfer_ot(
        base_img,
        target_img,
        downscale=args.downscale,
        metric=args.metric,
        reg=args.reg,
        sinkhorn_max_iter=args.sinkhorn_max_iter,
    )

    # Ensure output directory exists
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    # Save result (plt.imsave expects [0,1] float for RGB)
    plt.imsave(args.output, styl)
    print(f"Saved stylized image to: {args.output}")


if __name__ == "__main__":
    main()

