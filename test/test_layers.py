import torch
import pytest

from src.core.layers import (
    LTCNLayer,
    NeuralODELayer,
)
from src.core.models import (
    NeuralODEController,
)


def test_ltcn_layer_basic():
    """LTCNLayer basic functionality test."""
    in_dim, k, num_blocks = 5, 8, 3
    N = num_blocks * k  # total hidden dimension

    layer = LTCNLayer(in_dim, k, num_blocks)

    # Test forward pass
    y = torch.randn(N)
    u_t = torch.randn(in_dim)

    y_next = layer(y, u_t, dt=0.01, n_steps=1)

    # Check output shape
    assert y_next.shape == (N,)
    assert torch.isfinite(y_next).all()


def test_ltcn_layer_batch():
    """LTCNLayer batch processing test."""
    batch_size = 4
    in_dim, k, num_blocks = 3, 6, 2
    N = num_blocks * k

    layer = LTCNLayer(in_dim, k, num_blocks)

    # Test batch forward pass
    y = torch.randn(batch_size, N)
    u_t = torch.randn(batch_size, in_dim)

    y_next = layer(y, u_t, dt=0.02, n_steps=5)

    # Check output shape
    assert y_next.shape == (batch_size, N)
    assert torch.isfinite(y_next).all()


def test_ltcn_layer_activations():
    """Test different activation functions for LTCNLayer."""
    in_dim, k, num_blocks = 4, 5, 2
    N = num_blocks * k

    for activation in ["tanh", "relu", "sigmoid", "htanh"]:
        layer = LTCNLayer(in_dim, k, num_blocks, activation=activation)

        y = torch.randn(N)
        u_t = torch.randn(in_dim)

        y_next = layer(y, u_t, dt=0.01, n_steps=1)

        assert y_next.shape == (N,)
        assert torch.isfinite(y_next).all()


def test_ltcn_layer_clamping():
    """Test output clamping for LTCNLayer."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k

    layer = LTCNLayer(in_dim, k, num_blocks, clamp_output=1.0)

    # Start with large values
    y = torch.randn(N) * 10
    u_t = torch.randn(in_dim) * 10

    y_next = layer(y, u_t, dt=0.05, n_steps=10)

    # Check clamping
    assert torch.all(y_next <= 1.0 + 1e-6)
    assert torch.all(y_next >= -1.0 - 1e-6)


def test_ltcn_layer_gradient_flow():
    """Test gradient flow through LTCNLayer."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k

    layer = LTCNLayer(in_dim, k, num_blocks)

    y = torch.randn(N, requires_grad=True)
    u_t = torch.randn(in_dim, requires_grad=True)

    y_next = layer(y, u_t, dt=0.01, n_steps=1)
    loss = y_next.sum()

    loss.backward()

    # Check gradients exist
    assert y.grad is not None
    assert u_t.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_ltcn_layer_no_input():
    """Test LTCNLayer with no external input (u_t=None)."""
    in_dim, k, num_blocks = 3, 4, 2
    N = num_blocks * k

    layer = LTCNLayer(in_dim, k, num_blocks)

    y = torch.randn(N)

    y_next = layer(y, u_t=None, dt=0.01, n_steps=1)

    assert y_next.shape == (N,)
    assert torch.isfinite(y_next).all()


# ============== Neural ODE Tests ==============


def test_neural_ode_layer_basic():
    """Test NeuralODELayer basic functionality."""
    in_dim, hidden_dim = 8, 16

    layer = NeuralODELayer(in_dim, hidden_dim, num_hidden_layers=2)

    y = torch.randn(hidden_dim)
    u_t = torch.randn(in_dim)

    y_next = layer(y, u_t, dt=0.1, n_steps=1)

    assert y_next.shape == (hidden_dim,)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_batch():
    """Test NeuralODELayer with batch input."""
    batch_size = 4
    in_dim, hidden_dim = 5, 10

    layer = NeuralODELayer(in_dim, hidden_dim)

    y = torch.randn(batch_size, hidden_dim)
    u_t = torch.randn(batch_size, in_dim)

    y_next = layer(y, u_t, dt=0.05, n_steps=2)

    assert y_next.shape == (batch_size, hidden_dim)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_no_input():
    """Test NeuralODELayer with no external input."""
    hidden_dim = 12

    layer = NeuralODELayer(8, hidden_dim)

    y = torch.randn(hidden_dim)

    y_next = layer(y, u_t=None, dt=0.1, n_steps=1)

    assert y_next.shape == (hidden_dim,)
    assert torch.isfinite(y_next).all()


def test_neural_ode_layer_gradient_flow():
    """Test gradient flow through NeuralODELayer."""
    in_dim, hidden_dim = 6, 10

    layer = NeuralODELayer(in_dim, hidden_dim)

    y = torch.randn(hidden_dim, requires_grad=True)
    u_t = torch.randn(in_dim, requires_grad=True)

    y_next = layer(y, u_t, dt=0.1, n_steps=1)
    loss = y_next.sum()

    loss.backward()

    assert y.grad is not None
    assert u_t.grad is not None
    assert all(p.grad is not None for p in layer.parameters())


def test_neural_ode_controller():
    """Test NeuralODEController end-to-end."""
    B, T, C, H_frame, W_frame = 2, 3, 3, 64, 64
    hidden_dim, output_dim = 16, 6

    controller = NeuralODEController(
        frame_height=H_frame,
        frame_width=W_frame,
        hidden_dim=hidden_dim,
        output_dim=output_dim,
    )

    frames = torch.randn(B, T, C, H_frame, W_frame)

    controls, final_hidden = controller(frames)

    assert controls.shape == (B, T, output_dim)
    assert final_hidden.shape == (B, hidden_dim)
    assert torch.isfinite(controls).all()
    assert torch.isfinite(final_hidden).all()
