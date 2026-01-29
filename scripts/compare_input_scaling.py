
import torch
import torch.nn as nn
from src.core.layers.ltcn_layer import LTCNLayer
from src.core.layers.node_layer import NeuralODELayer

def compare_input_response():
    input_dim = 10
    hidden_dim = 64
    dt = 0.1
    n_steps = 1
    
    # Instantiate layers
    # LTCN: 4 blocks, k=16 -> 64 hidden
    ltcn = LTCNLayer(in_dim=input_dim, k=16, num_blocks=4)
    # NODE: match dim
    node = NeuralODELayer(in_dim=input_dim, hidden_dim=hidden_dim, num_hidden_layers=1, mlp_hidden_dim=10)
    
    # Initialize zero state
    y_ltcn = torch.zeros(1, hidden_dim)
    y_node = torch.zeros(1, hidden_dim)
    
    # Random input u_t (order of magnitude 1.0)
    torch.manual_seed(42)
    u_t = torch.randn(1, input_dim)
    
    print(f"Input norm: {u_t.norm().item():.4f}")
    
    # Forward pass LTCN
    # We want to see how much y changes from 0 based on u_t
    with torch.no_grad():
        y_ltcn_next = ltcn(y_ltcn, u_t, dt=dt, n_steps=n_steps)
        delta_ltcn = y_ltcn_next - y_ltcn
        print(f"LTCN Delta norm: {delta_ltcn.norm().item():.4f}")
        
    # Forward pass NODE
    with torch.no_grad():
        y_node_next = node(y_node, u_t, dt=dt, n_steps=n_steps)
        delta_node = y_node_next - y_node
        print(f"NODE Delta norm: {delta_node.norm().item():.4f}")

    # Check weights magnitude to be fair
    print("\nWeight Stats:")
    print(f"LTCN W_in mean abs: {ltcn.W_in.weight.abs().mean():.4f}")
    print(f"NODE input_proj mean abs: {node.input_proj.weight.abs().mean():.4f}")

if __name__ == "__main__":
    compare_input_response()
