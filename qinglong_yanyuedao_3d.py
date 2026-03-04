"""用 Python 生成青龍偃月刀 3D 模型示意圖。

依賴：
    pip install numpy matplotlib

執行：
    python qinglong_yanyuedao_3d.py
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  # 啟用 3D 投影


def rotation_matrix_y(theta: float) -> np.ndarray:
    """繞 y 軸旋轉的 3x3 矩陣。"""
    c, s = np.cos(theta), np.sin(theta)
    return np.array(
        [
            [c, 0, s],
            [0, 1, 0],
            [-s, 0, c],
        ]
    )


def cylinder_mesh(
    length: float,
    radius: float,
    center: tuple[float, float, float],
    axis: str = "z",
    n_theta: int = 40,
    n_len: int = 2,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """建立沿指定軸的圓柱網格。"""
    theta = np.linspace(0, 2 * np.pi, n_theta)
    lin = np.linspace(0, length, n_len)
    tt, ll = np.meshgrid(theta, lin)

    x = radius * np.cos(tt)
    y = radius * np.sin(tt)
    z = ll

    if axis == "z":
        X, Y, Z = x, y, z
    elif axis == "x":
        X, Y, Z = z, x, y
    elif axis == "y":
        X, Y, Z = y, z, x
    else:
        raise ValueError("axis 必須是 'x'、'y' 或 'z'")

    X = X + center[0]
    Y = Y + center[1]
    Z = Z + center[2]
    return X, Y, Z


def blade_surface(
    length: float = 1.2,
    width: float = 0.18,
    thickness: float = 0.03,
    curve: float = 0.35,
    n_u: int = 70,
    n_v: int = 24,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """生成帶弧度與刀鋒收束的偃月刀刀身曲面。"""
    u = np.linspace(0, 1, n_u)
    v = np.linspace(-1, 1, n_v)
    U, V = np.meshgrid(u, v)

    # 主體：沿 x 延伸，帶向上弧度（z）。
    X = length * U
    centerline = curve * np.sin(0.8 * np.pi * U) * (0.35 + 0.65 * U)

    # 刀身寬度由根部到前端逐漸變窄，並在前 20% 更尖。
    local_width = width * (1.0 - 0.65 * U)
    tip_factor = np.clip((1.0 - U) / 0.2, 0, 1)
    local_width = local_width * (0.35 + 0.65 * tip_factor)

    Y = 0.5 * local_width * V

    # 厚度中央較厚，邊緣漸薄，呈現刀脊。
    ridge = (1 - np.abs(V) ** 1.6)
    local_thickness = thickness * (0.7 + 0.3 * (1 - U))
    Z = centerline + ridge * local_thickness

    # 增加刀鋒下切感（前端下彎）。
    Z -= 0.08 * (U**1.8) * (V > 0)

    return X, Y, Z


def dragon_ornament(
    base: tuple[float, float, float],
    scale: float = 0.1,
    turns: int = 4,
    points: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """以螺旋曲線模擬刀根附近青龍紋飾。"""
    t = np.linspace(0, 2 * np.pi * turns, points)
    x = base[0] + scale * 0.45 * np.cos(t)
    y = base[1] + scale * 0.22 * np.sin(t)
    z = base[2] + scale * 0.12 * t / np.pi
    return x, y, z


def make_qinglong_yanyuedao() -> None:
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 1) 長桿（木柄）
    shaft_len = 3.8
    shaft_radius = 0.04
    shaft = cylinder_mesh(length=shaft_len, radius=shaft_radius, center=(0, 0, -3.3), axis="z")
    ax.plot_surface(*shaft, color="#8b5a2b", linewidth=0, alpha=0.95, shade=True)

    # 2) 尾端金屬配重（圓錐近似）
    h = np.linspace(0, 0.35, 30)
    th = np.linspace(0, 2 * np.pi, 40)
    TH, H = np.meshgrid(th, h)
    r = 0.06 * (1 - H / H.max())
    Xb = r * np.cos(TH)
    Yb = r * np.sin(TH)
    Zb = -3.65 - H
    ax.plot_surface(Xb, Yb, Zb, color="#bfa14a", linewidth=0, alpha=1.0, shade=True)

    # 3) 刀根護手（短圓柱）
    guard = cylinder_mesh(length=0.18, radius=0.09, center=(0, 0, 0.42), axis="z")
    ax.plot_surface(*guard, color="#c9c9c9", linewidth=0, alpha=1.0, shade=True)

    # 4) 刀身（偃月）
    X, Y, Z = blade_surface()

    # 把刀身旋轉到與桿件上端連接的角度。
    blade_pts = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=0)
    R = rotation_matrix_y(np.deg2rad(-16))
    blade_rot = R @ blade_pts
    Xr = blade_rot[0, :].reshape(X.shape) + 0.02
    Yr = blade_rot[1, :].reshape(Y.shape)
    Zr = blade_rot[2, :].reshape(Z.shape) + 0.45

    ax.plot_surface(Xr, Yr, Zr, color="#d9e3ea", linewidth=0, antialiased=True, shade=True)

    # 5) 刀背青龍紋飾
    dx, dy, dz = dragon_ornament(base=(0.12, 0.0, 0.55), scale=0.11)
    ax.plot3D(dx, dy, dz, color="#1f8f55", linewidth=2.2)

    # 視覺設定
    ax.set_title("青龍偃月刀 3D 模型（Python）", fontsize=15, pad=14)
    ax.set_axis_off()
    ax.view_init(elev=17, azim=-63)

    # 讓比例看起來較自然
    ax.set_box_aspect((3.2, 0.8, 4.2))

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    make_qinglong_yanyuedao()
