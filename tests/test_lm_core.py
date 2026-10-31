"""Tokenizer, Transformer, training loop, save/load, generation, evaluation."""
import math

import numpy as np
import pytest
import torch

from conftest import CORPUS
from nexora.lm.data import BatchSampler, fingerprint, validate_text
from nexora.lm.evaluate import eval_qa, eval_text
from nexora.lm.model import LMConfig, NexoraLM, load_model, save_model
from nexora.lm.tokenizer import EOT, BPETokenizer
from nexora.lm.train import train

TEXT = CORPUS.read_text(encoding="utf-8")[:60_000]


@pytest.fixture(scope="module")
def tok():
    return BPETokenizer.train(TEXT, 300)


# ------------------------------------------------------------------ tokenizer
def test_tokenizer_learns_merges_and_vocab_size(tok):
    assert tok.vocab_size == 300
    assert len(tok.merges) == 300 - 257
    assert " the" in [tok.decode([i]) for i in range(256, tok.vocab_size - 1)]


@pytest.mark.parametrize("s", ["Hello, world!", "Zażółć gęślą jaźń 🙂", "  spaces\n\nnew\tlines ", "", "x" * 500])
def test_tokenizer_roundtrip(tok, s):
    assert tok.decode(tok.encode(s)) == s


def test_tokenizer_compresses_and_handles_eot(tok):
    ids = tok.encode(TEXT[:5000])
    assert len(ids) < 5000 * 0.8  # merges actually compress (only 43 merges here)
    ids = tok.encode(f"a{EOT}b")
    assert tok.eot_id in ids and tok.decode(ids) == f"a{EOT}b"
    assert max(tok.encode(TEXT[:2000])) < tok.vocab_size


def test_tokenizer_save_load(tok, tmp_path):
    tok.save(tmp_path / "t.json")
    t2 = BPETokenizer.load(tmp_path / "t.json")
    assert t2.encode(TEXT[:3000]) == tok.encode(TEXT[:3000])
    (tmp_path / "bad.json").write_text('{"type":"other"}')
    with pytest.raises(ValueError):
        BPETokenizer.load(tmp_path / "bad.json")


# ---------------------------------------------------------------------- model
CFG = LMConfig(vocab_size=300, block_size=32, n_layer=2, n_head=2, n_embd=32, dropout=0.0)


def test_model_random_initialisation():
    torch.manual_seed(0)
    a, b = NexoraLM(CFG), NexoraLM(CFG)
    w = a.blocks[0].attn.qkv.weight
    assert abs(w.std().item() - 0.02) < 0.004 and abs(w.mean().item()) < 0.003
    assert not torch.equal(a.tok_emb.weight, b.tok_emb.weight)  # really random
    assert a.head.weight is a.tok_emb.weight  # tied
    assert torch.all(a.blocks[0].ln1.weight == 1)


def test_initial_loss_is_near_uniform():
    torch.manual_seed(0)
    m = NexoraLM(CFG)
    x = torch.randint(0, 300, (4, 32))
    y = torch.randint(0, 300, (4, 32))
    _, loss = m(x, y)
    assert abs(loss.item() - math.log(300)) < 0.3


def test_causal_mask_blocks_the_future():
    torch.manual_seed(0)
    m = NexoraLM(CFG).eval()
    x = torch.randint(0, 300, (1, 16))
    y = x.clone(); y[0, 10:] = (y[0, 10:] + 7) % 300
    la, _ = m(x); lb, _ = m(y)
    assert torch.allclose(la[0, :10], lb[0, :10], atol=1e-5)
    assert not torch.allclose(la[0, 10:], lb[0, 10:])


def test_forward_backward_shapes_and_grads():
    m = NexoraLM(CFG)
    x = torch.randint(0, 300, (3, 20))
    logits, loss = m(x, x)
    assert logits.shape == (3, 20, 300)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in m.parameters())
    with pytest.raises(ValueError):
        m(torch.zeros(1, 33, dtype=torch.long))


def test_loss_mask():
    m = NexoraLM(CFG)
    x = torch.randint(0, 300, (1, 10))
    mask = torch.zeros(1, 10); mask[0, -3:] = 1
    _, l1 = m(x, x, loss_mask=mask)
    logits, _ = m(x)
    ce = torch.nn.functional.cross_entropy(logits[0, -3:], x[0, -3:])
    assert torch.allclose(l1, ce, atol=1e-5)


def test_training_reduces_loss_and_changes_weights(tok):
    torch.manual_seed(0)
    m = NexoraLM(CFG)
    before = {k: v.clone() for k, v in m.state_dict().items()}
    arr = np.asarray(tok.encode(TEXT[:20000]), dtype=np.uint16)
    s = BatchSampler([(arr, 1.0)], 32, 8, seed=0)
    first = eval_text(m, arr, max_windows=16)["loss"]
    res = train(m, s, max_steps=80, lr=3e-3, warmup=5)
    assert res["status"] == "completed" and res["step"] == 80
    after = eval_text(m, arr, max_windows=16)["loss"]
    assert after < first - 0.5, (first, after)
    changed = sum(not torch.equal(before[k], v) for k, v in m.state_dict().items())
    assert changed >= len(before) - 2


def test_save_load_and_resume(tmp_path, tok):
    torch.manual_seed(0)
    m = NexoraLM(CFG)
    save_model(m, tmp_path / "m.pt")
    m2 = load_model(tmp_path / "m.pt")
    for (k, a), (_, b) in zip(m.state_dict().items(), m2.state_dict().items()):
        assert torch.equal(a, b), k
    # resume: 20 steps + 20 resumed steps == 40 uninterrupted steps
    arr = np.asarray(tok.encode(TEXT[:20000]), dtype=np.uint16)

    from nexora.jobs.control import NullControl, StopTraining

    class StopAt(NullControl):  # simulates the user pressing "stop" after 20 steps
        n = 0

        def tick(self, save_fn=None):
            self.n += 1
            if self.n == 21:
                raise StopTraining("cancelled")

    def run(ck, model, control=None):
        return train(model, BatchSampler([(arr, 1.0)], 32, 4, seed=1), max_steps=40, lr=1e-3,
                     warmup=2, checkpoint_every=10, ckpt_path=ck, control=control)
    torch.manual_seed(0); a = NexoraLM(CFG); run(tmp_path / "a.pt", a)
    torch.manual_seed(0); b = NexoraLM(CFG); r1 = run(tmp_path / "b.pt", b, StopAt())
    assert r1["status"] == "cancelled" and r1["step"] == 20
    torch.manual_seed(0); c = NexoraLM(CFG); r = run(tmp_path / "b.pt", c)
    assert r["step"] == 40 and r["status"] == "completed"
    for k in a.state_dict():
        assert torch.allclose(a.state_dict()[k], c.state_dict()[k], atol=1e-5), k


def test_generation_uses_model_and_is_seeded(tok):
    torch.manual_seed(0)
    m = NexoraLM(CFG)
    x = torch.tensor([tok.encode("Hello")])
    g1 = m.generate(x, 20, generator=torch.Generator().manual_seed(5))
    g2 = m.generate(x, 20, generator=torch.Generator().manual_seed(5))
    assert torch.equal(g1, g2) and g1.shape[1] == x.shape[1] + 20
    assert int(g1.max()) < CFG.vocab_size
    greedy = m.generate(x, 10, temperature=0)
    assert torch.equal(greedy, m.generate(x, 10, temperature=0))
    long = m.generate(torch.randint(0, 300, (1, 40))[:, :32], 10)  # sliding window beyond block
    assert long.shape[1] == 42


def test_evaluation_is_reproducible(tok):
    torch.manual_seed(0)
    m = NexoraLM(CFG)
    arr = np.asarray(tok.encode(TEXT[:8000]), dtype=np.uint16)
    assert eval_text(m, arr) == eval_text(m, arr)
    items = [{"prompt": "KING:\n", "answer": "Good morrow to you all."}] * 3
    a, b = eval_qa(m, tok, items), eval_qa(m, tok, items)
    assert a == b and a["items"] == 3
    m.train()
    assert eval_text(m, arr)["loss"] == eval_text(m, arr)["loss"]  # dropout disabled in eval


# ----------------------------------------------------------------------- data
def test_dataset_validation_and_dedup():
    assert validate_text("Too short")[0] is False
    assert validate_text("1234567890 " * 10)[0] is False
    assert validate_text("spam " * 100)[1] == "highly repetitive"
    assert validate_text("The quick brown fox jumps over the lazy dog, again and again today.")[0]
    assert fingerprint("Hello   World") == fingerprint("hello world")


def test_batch_sampler_mixes_sources_deterministically():
    a = np.zeros(1000, dtype=np.uint16); b = np.ones(1000, dtype=np.uint16)
    s1 = BatchSampler([(a, 0.5), (b, 0.5)], 16, 64, seed=3)
    s2 = BatchSampler([(a, 0.5), (b, 0.5)], 16, 64, seed=3)
    x1, y1 = s1.next(); x2, _ = s2.next()
    assert torch.equal(x1, x2) and x1.shape == (64, 16) and y1.shape == (64, 16)
    frac = (x1[:, 0] == 1).float().mean().item()
    assert 0.2 < frac < 0.8
    with pytest.raises(ValueError):
        BatchSampler([(np.zeros(5, dtype=np.uint16), 1.0)], 16, 4)
