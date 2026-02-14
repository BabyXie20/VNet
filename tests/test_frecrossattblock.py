import torch

from model import DWT3D, IDWT3D, FreCrossAttBlock


def _run_case(use_shift: bool):
    torch.manual_seed(0)
    b, c, d, h, w = 2, 48, 96, 96, 96
    e_l = torch.randn(b, c, d, h, w, requires_grad=True)
    u_l = torch.randn(b, 2 * c, d // 2, h // 2, w // 2, requires_grad=True)

    block = FreCrossAttBlock(
        dim=c,
        dwt3d=DWT3D(),
        idwt3d=IDWT3D(),
        num_heads=6,
        window_size=(6, 6, 6),
        use_shift=use_shift,
    )

    out = block(e_l, u_l)
    assert out.shape == (b, c, d, h, w)

    loss = out.mean()
    loss.backward()

    grads = [p.grad for p in block.parameters() if p.requires_grad]
    assert any(g is not None for g in grads)


def test_frecrossattblock_no_shift():
    _run_case(use_shift=False)


def test_frecrossattblock_shift():
    _run_case(use_shift=True)
