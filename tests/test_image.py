"""Diffusion image model: data, architecture, forward process, training job, PNG export."""
import numpy as np
import torch
from PIL import Image

from conftest import wait_job
from nexora.image.data import synthetic_shapes, to_tensor
from nexora.image.model import CaptionTokenizer, Diffusion, ImageModelConfig, UNet


def test_synthetic_dataset():
    imgs, caps = synthetic_shapes(20, 16, seed=1)
    assert imgs.shape == (20, 16, 16, 3) and imgs.dtype == np.uint8 and len(caps) == 20
    again, caps2 = synthetic_shapes(20, 16, seed=1)
    assert np.array_equal(imgs, again) and caps == caps2
    x = to_tensor(imgs)
    assert x.min() >= -1 and x.max() <= 1


def test_caption_tokenizer():
    t = CaptionTokenizer.build(["a red circle", "a blue square"], max_len=6)
    ids = t.encode("a red Triangle")
    assert len(ids) == 6 and ids[2] == CaptionTokenizer.UNK
    assert t.encode("")[0] == CaptionTokenizer.NULL
    assert CaptionTokenizer.from_json(t.to_json()).words == t.words


def test_unet_shapes_and_text_conditioning():
    torch.manual_seed(0)
    cfg = ImageModelConfig(resolution=16, base_channels=16, channel_mults=[1, 2], text_dim=32, vocab_size=20, timesteps=50)
    m = UNet(cfg)
    x = torch.randn(2, 3, 16, 16)
    ids = torch.tensor([[3, 4, 0, 0] + [0] * 12, [5, 6, 7, 0] + [0] * 12])
    out = m(x, torch.tensor([3, 40]), ids)
    assert out.shape == x.shape
    # make the network non-trivial, then check the caption changes the output
    for p in m.parameters():
        p.data.add_(torch.randn_like(p) * 0.05)
    a = m(x[:1], torch.tensor([10]), ids[:1]); b = m(x[:1], torch.tensor([10]), ids[1:])
    assert not torch.allclose(a, b)


def test_forward_noise_process():
    d = Diffusion(100)
    assert d.ac[0] > 0.99 and d.ac[-1] < 0.01 and torch.all(d.ac[1:] <= d.ac[:-1])
    x0 = torch.ones(1, 3, 4, 4)
    n = torch.randn_like(x0)
    assert torch.allclose(d.q_sample(x0, torch.tensor([0]), n), x0, atol=0.1)
    assert (d.q_sample(x0, torch.tensor([99]), n) - n).abs().mean() < 0.1


def test_loss_backward():
    cfg = ImageModelConfig(resolution=16, base_channels=16, channel_mults=[1, 2], text_dim=32, vocab_size=20, timesteps=50)
    m, d = UNet(cfg), Diffusion(50)
    loss = d.loss(m, torch.randn(2, 3, 16, 16), torch.randint(0, 20, (2, 16)))
    loss.backward()
    assert torch.isfinite(loss) and m.inp.weight.grad is not None


def test_image_training_job_and_generation(app):
    info = app.images.prepare("synthetic_shapes", "shapes", n=120)
    assert info["images"] == 120
    j = wait_job(app, app.jobs.submit("image_train", {"dataset": "shapes"}))
    assert j["status"] == "completed", j
    v = app.images.active()
    assert v["id"] == "img0001" and v["metrics"]["total_steps"] == 30
    r = app.images.generate("a small red circle on a blue background", n=2, seed=0)
    assert len(r["files"]) == 2 and r["warning"]  # honest: barely trained
    img = Image.open(app.paths.generated / r["files"][0])
    assert img.format == "PNG" and img.size == (256, 256)
    r2 = app.images.generate("a zebra", n=1, seed=0)
    assert "zebra" in r2["unknown_words"]
    # continue training an existing version (e.g. later on better hardware)
    j2 = wait_job(app, app.jobs.submit("image_train", {"dataset": "shapes", "resume_from": "img0001", "steps": 10}))
    assert j2["status"] == "completed"
    assert [x for x in app.images.versions() if x["id"] == "img0002"][0]["metrics"]["total_steps"] == 40
