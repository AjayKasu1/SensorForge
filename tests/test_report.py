import numpy as np

from sensorforge.isp.params import ISPParams
from sensorforge.metrics.emva1288 import PhotonTransfer
from sensorforge.metrics.report import write_convergence_gif, write_report


def _img(seed):
    return (np.random.default_rng(seed).random((32, 32, 3)) * 255).astype(np.uint8)


def test_report_writes_markdown_and_comparison(tmp_path):
    metrics = {"SSIM": 0.83, "PSNR": 29.4, "deltaE2000": 4.21}
    path = write_report(tmp_path, _img(0), _img(1), metrics, ISPParams())
    assert path.exists()
    text = path.read_text()
    assert "SSIM" in text and "4.21" in text
    assert (tmp_path / "comparison.png").exists()
    # Params are embedded so a run is self-describing.
    assert "awb_gain_r" in text


def test_report_includes_photon_transfer_when_given(tmp_path):
    pt = PhotonTransfer(
        gain_dn_per_e=0.8,
        dark_variance_dn2=2.0,
        mean_signal_dn=np.array([100.0, 500.0, 1000.0]),
        temporal_variance_dn2=np.array([82.0, 402.0, 802.0]),
    )
    path = write_report(tmp_path, _img(2), _img(3), {"SSIM": 0.9}, ISPParams(), photon_transfer=pt)
    assert (tmp_path / "photon_transfer.png").exists()
    assert "System gain K" in path.read_text()


def test_convergence_gif_has_one_frame_per_iteration(tmp_path):
    from PIL import Image

    frames = [_img(i) for i in range(4)]
    out = write_convergence_gif(tmp_path / "demo.gif", frames, _img(9))
    assert out.exists()
    assert Image.open(out).n_frames == 4
