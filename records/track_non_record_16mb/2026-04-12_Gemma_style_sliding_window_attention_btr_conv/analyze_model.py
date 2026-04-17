import torch
import math

def format_size(size):
    return f"{size / (1024 * 1024):.2f} MB"

def analyze_model(model_path):
    print(f"Loading {model_path}...")
    try:
        data = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print(f"Failed to load: {e}")
        return

    # If it's the raw state dict
    state_dict = data if isinstance(data, dict) else data.state_dict()
        
    print(f"\n================ MODEL ANALYSIS ================")
    
    total_params = 0
    total_bytes = 0
    
    skip_weights = None
    tied_embed = None
    
    layer_types = {'embed': 0, 'attn': 0, 'mlp': 0, 'other': 0}
    layer_bytes = {'embed': 0, 'attn': 0, 'mlp': 0, 'other': 0}
    
    for name, param in state_dict.items():
        if isinstance(param, torch.Tensor):
            numel = param.numel()
            bytes = numel * param.element_size()
            total_params += numel
            total_bytes += bytes
            
            if 'tok_emb' in name or 'lm_head' in name:
                c = 'embed'
                if 'tok_emb' in name: tied_embed = param
            elif '.mlp.' in name:
                c = 'mlp'
            elif '.attn.' in name or '.proj.' in name:
                c = 'attn'
            else:
                c = 'other'
                
            layer_types[c] += numel
            layer_bytes[c] += bytes
            
            if 'skip_weights' in name:
                skip_weights = param
                print(f"[{name}] Shape: {list(param.shape)}, Type: {param.dtype}")
                # Analyze skip weight magnitudes
                norms = torch.norm(param, dim=1).float()
                print(f"   Magnitudes (L2 norm per skip): {[round(float(x), 3) for x in norms]}")
            
            if 'parallel_post_lambdas' in name:
                print(f"[{name}] parallel mappings:")
                try:
                    for i, lam in enumerate(param):
                        # typically 2x2: (attn->attn, attn->mlp, mlp->attn, mlp->mlp)
                        flat = lam.flatten()
                        print(f"   layer {i}: {flat.tolist()}")
                except:
                    pass

    print("\n--- Summary ---")
    print(f"Total Parameters: {total_params / 1e6:.2f} M")
    print(f"Total Size (FP16/BF16): {format_size(total_bytes)}")
    
    print("\n--- Distribution ---")
    for k in layer_types:
        print(f"  {k:5s}: {layer_types[k]/1e6:5.2f} M params ({layer_types[k]/total_params*100:5.1f}%) | {format_size(layer_bytes[k])}")
        
    if tied_embed is not None:
        print(f"\nEmbedding Mag (L2): {torch.norm(tied_embed.float()).item():.2f}")
    
    print("================================================\n")

if __name__ == '__main__':
    analyze_model('final_model.pt')
