import cv2
import numpy as np


def get_rotation_rodrigues(n: np.ndarray, theta: float) -> np.ndarray:
    """
    与えられた単位ベクトルnと角度thetaに基づいて、ロドリゲスの回転公式を使用して回転行列を計算する。
    引数:
        n: 単位ベクトル (numpy.ndarray, shape=(3,))
        theta: 回転角度 [ラジアン]
    戻り値:
        R: 3x3 の回転行列 (numpy.ndarray)
    """
    assert n.shape == (3,), "n must be a 3D vector"

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    a = 1 - cos_theta
    R = np.array(
        [
            [
                n[0] * n[0] * a + cos_theta,
                n[0] * n[1] * a - n[2] * sin_theta,
                n[0] * n[2] * a + n[1] * sin_theta,
            ],
            [
                n[0] * n[1] * a + n[2] * sin_theta,
                n[1] * n[1] * a + cos_theta,
                n[1] * n[2] * a - n[0] * sin_theta,
            ],
            [
                n[0] * n[2] * a - n[1] * sin_theta,
                n[1] * n[2] * a + n[0] * sin_theta,
                n[2] * n[2] * a + cos_theta,
            ],
        ]
    )
    return R


def get_rotation_matrix(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """
    与えられたヨー（y軸）、ピッチ（x軸）、ロール（z軸）の順に右手系で回転する回転行列を返す。

    引数:
        yaw: ヨー角（y軸周りの回転） [ラジアン]
        pitch: ピッチ角（x軸周りの回転） [ラジアン]
        roll: ロール角（z軸周りの回転） [ラジアン]

    戻り値:
        R: 3x3 の回転行列（numpy.ndarray）
    """

    # ヨー: y軸周り
    Ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])

    # ピッチ: y軸周りに回転後のx軸周り
    xdir = Ry @ np.array([1, 0, 0])
    Rx = get_rotation_rodrigues(xdir, pitch)  # ピッチ: x軸周りの回転行列

    # ロール: x軸周りに回転後のz軸周り
    Rxy = Rx @ Ry  # ヨーとピッチを適用した後の回転行列
    zdir = Rxy @ np.array([0, 0, 1])
    Rz = get_rotation_rodrigues(zdir, roll)  # ロール: z軸周りの回転行列

    R = Rz @ Rxy
    return R


def calc_warp_map(
    eq_size: tuple[int, int],
    pe_size: tuple[int, int],
    v_fov_deg: float,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    equirectangular画像を透視投影画像に変換するためのワープマップを計算する。

    引数:
        eq_size: equirectangular画像のサイズ (幅, 高さ)
        pe_size: 透視投影画像のサイズ (幅, 高さ)
        v_fov_deg: 垂直方向の画角 [度]
        yaw_deg: ヨー角 [度]
        pitch_deg: ピッチ角 [度]
        roll_deg: ロール角 [度]

    戻り値:
        mapx: x座標のワープマップ
        mapy: y座標のワープマップ
    """

    assert len(eq_size) == 2, "eq_size must be a tuple of (height, width)"
    assert len(pe_size) == 2, "pe_size must be a tuple of (height, width)"
    assert 0 < v_fov_deg < 180, "v_fov_deg must be between 0 and 180 degrees"

    w_eq, h_eq = eq_size
    w_pe, h_pe = pe_size

    # 透視投影画像のカメラ座標から正距円筒図法画像のカメラ座標への回転行列を計算
    R_ep = get_rotation_matrix(np.deg2rad(yaw_deg), np.deg2rad(pitch_deg), np.deg2rad(roll_deg))

    # 垂直方向のスケールを計算
    fov_scale = np.tan(np.deg2rad(v_fov_deg) / 2) / (h_pe / 2)

    # 透視投影画像のX座標の範囲
    x_p_range = np.arange(-w_pe / 2, w_pe / 2)
    # 透視投影画像のY座標の範囲
    y_p_range = np.arange(-h_pe / 2, h_pe / 2)
    # 透視投影画像のX座標とY座標のメッシュグリッド
    x_p_mesh, y_p_mesh = np.meshgrid(x_p_range, y_p_range)
    x_p_mesh = x_p_mesh.reshape(-1, 1)
    y_p_mesh = y_p_mesh.reshape(-1, 1)
    # 透視投影画像の各座標(画像中心が原点)
    xy_p = np.concatenate([x_p_mesh, y_p_mesh], axis=1)
    # 透視投影画像の各座標を3次元座標に変換
    xyz_p = (
        np.array([0, 0, 1])
        + xy_p[:, 0:1] * np.array([fov_scale, 0, 0])
        + xy_p[:, 1:2] * np.array([0, fov_scale, 0])
    )

    # 透視投影画像の各点を正距円筒座標に変換
    xyz_e = (R_ep @ xyz_p.T).T

    # 方位角
    azimuth_e = np.arctan2(xyz_e[:, 0], xyz_e[:, 2])
    # 仰俯角
    elevation_e = np.arctan2(xyz_e[:, 1], np.sqrt(xyz_e[:, 0] ** 2 + xyz_e[:, 2] ** 2))
    # 正距円筒座標画像のX座標(画像中心が原点)
    x_e = azimuth_e * (w_eq / (2 * np.pi))
    # 正距円筒座標画像のY座標(画像中心が原点)
    y_e = elevation_e * (h_eq / np.pi)
    # 正距円筒座標画像のU座標(画像左上が原点)に変換
    mapx = (x_e + w_eq / 2).reshape(h_pe, w_pe)
    # 正距円筒座標画像のV座標(画像左上が原点)に変換
    mapy = (y_e + h_eq / 2).reshape(h_pe, w_pe)

    return mapx.astype(np.float32), mapy.astype(np.float32)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument(
        "--output_size",
        type=int,
        nargs=2,
        default=(500, 500),
        help="Output image size (width, height)",
    )
    parser.add_argument(
        "--v_fov_deg",
        type=float,
        default=120,
        help="Vertical field of view in degrees",
    )
    parser.add_argument(
        "--yaw_deg",
        type=float,
        default=0,
        help="Yaw angle in degrees (rotation around y-axis)",
    )
    parser.add_argument(
        "--pitch_deg",
        type=float,
        default=0,
        help="Pitch angle in degrees (rotation around x-axis)",
    )
    parser.add_argument(
        "--roll_deg",
        type=float,
        default=0,
        help="Roll angle in degrees (rotation around z-axis)",
    )
    parser.add_argument("--show", action="store_true", help="Show the output image")
    args = parser.parse_args()

    eq_img = cv2.imread(args.input)
    assert eq_img is not None, f"Failed to read image from {args.input}"
    assert eq_img.shape[0] * 2 == eq_img.shape[1], "Input image must be equirectangular (H x 2H)"

    mapx, mapy = calc_warp_map(
        eq_size=(eq_img.shape[1], eq_img.shape[0]),
        pe_size=(args.output_size[0], args.output_size[1]),
        v_fov_deg=args.v_fov_deg,
        yaw_deg=args.yaw_deg,
        pitch_deg=args.pitch_deg,
        roll_deg=args.roll_deg,
    )
    pe_img = cv2.remap(
        eq_img,
        mapx,
        mapy,
        cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE,
    )

    if args.show:
        cv2.imshow("warped image", pe_img)
        cv2.waitKey(0)
    cv2.imwrite(args.output, pe_img)
    print(f"Output saved to {args.output}")
