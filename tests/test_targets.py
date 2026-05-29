import numpy as np

from sensorforge.capture.targets import checkerboard, save_png, uniform_field


def test_checkerboard_shape_follows_corner_convention():
    # (9, 6) inner corners -> 10x7 squares -> *square_px pixels each.
    board = checkerboard(inner_corners=(9, 6), square_px=10)
    assert board.shape == (7 * 10, 10 * 10)
    assert board.dtype == np.float32


def test_checkerboard_alternates():
    board = checkerboard(inner_corners=(2, 2), square_px=1, light=1.0, dark=0.0)
    # Top-left is light; both orthogonal neighbors must differ from it.
    assert board[0, 0] == 1.0
    assert board[0, 1] != board[0, 0]
    assert board[1, 0] != board[0, 0]
    # Only two distinct values exist.
    assert set(np.unique(board)) == {0.0, 1.0}


def test_uniform_field_is_constant():
    field = uniform_field(height=32, width=48, level=0.4)
    assert field.shape == (32, 48)
    assert np.all(field == np.float32(0.4))


def test_save_png_roundtrips_levels(tmp_path):
    import cv2

    field = uniform_field(height=8, width=8, level=0.5)
    out = tmp_path / "field.png"
    save_png(field, out)

    loaded = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    # 0.5 * 255 + 0.5 rounds to 128.
    assert loaded.shape == (8, 8)
    assert np.all(loaded == 128)
