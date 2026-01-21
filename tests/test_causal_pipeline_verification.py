
import torch
import torch.nn as nn
from mtr.config import cfg
from mtr.models.motion_decoder.mtr_decoder import MTRDecoder

def test_causal_pipeline():
    print("="*60)
    print("Testing Causal MTR Pipeline Integration")
    print("="*60)

    # 1. Mock Config
    model_cfg = cfg.MODEL.MOTION_DECODER
    # Enable Causal components
    model_cfg.USE_CAUSAL_SCORER = True
    model_cfg.SCORER_NUM_HEADS = 4
    model_cfg.SCORER_HIDDEN_DIM = 128
    model_cfg.USE_CAUSAL_PLANNING_LOSS = True
    model_cfg.SAFETY_THRESHOLD = 1.5
    model_cfg.LOSS_WEIGHTS = {
        'cls': 1.0, 'reg': 1.0, 'vel': 0.5,
        'causal_imitation': 1.0, 'causal_safety': 10.0
    }
    
    # 2. Initialize Decoder
    print("\n[1] Initializing MTRDecoder...")
    decoder = MTRDecoder(in_channels=256, config=model_cfg).cuda()
    print("    Success! CausalScorer initialized:", decoder.causal_scorer is not None)
    
    # 3. Mock Inputs
    B, N, M = 2, 32, 128  # Batch, Agents, Map Polylines
    C = 256
    
    batch_dict = {
        'obj_feature': torch.randn(B, N, C).cuda(),
        'obj_mask': torch.ones(B, N).bool().cuda(),
        'obj_pos': torch.randn(B, N, 3).cuda(),
        'map_feature': torch.randn(B, M, C).cuda(),
        'map_mask': torch.ones(B, M).bool().cuda(),
        'map_pos': torch.randn(B, M, 3).cuda(),
        'center_objects_feature': torch.randn(B, C).cuda(),
        'input_dict': {
            'center_objects_type': ['TYPE_VEHICLE'] * B,
            # Training specific inputs
            'center_gt_trajs': torch.randn(B, 80, 4).cuda(), # GT
            'center_gt_trajs_mask': torch.ones(B, 80).cuda(),
            'center_gt_final_valid_idx': torch.full((B,), 79).long().cuda(),
            # Dense future inputs
            'obj_trajs_future_state': torch.randn(B, N, 80, 4).cuda(),
            'obj_trajs_future_mask': torch.ones(B, N, 80).cuda(),
        }
    }
    
    # 4. Test Training Forward (Automatic Candidate Generation)
    print("\n[2] Testing Training Forward (Simulating Training Step)...")
    decoder.train()
    out = decoder(batch_dict)
    
    print("    Forward pass successful!")
    print("    Automatic candidates generated:", 'ego_future_candidates' in batch_dict['input_dict'])
    if 'ego_future_candidates' in batch_dict['input_dict']:
        shape = batch_dict['input_dict']['ego_future_candidates'].shape
        print(f"    Candidate shape: {shape} (Expected: B, 6, 80, 2)")
        
    # 5. Test Loss Calculation
    print("\n[3] Testing Loss Calculation...")
    loss, tb_dict, _ = decoder.get_loss()
    print(f"    Total Loss: {loss.item():.4f}")
    print("    Causal Loss included:", 'loss_causal_planning' in tb_dict)
    if 'loss_causal_planning' in tb_dict:
        print(f"    Causal Loss Value: {tb_dict['loss_causal_planning']:.4f}")

    # 6. Test Inference Forward
    print("\n[4] Testing Inference Forward (Simulating Validation)...")
    decoder.eval()
    # Remove generated candidates to force regeneration using FrenetSampler
    del batch_dict['input_dict']['ego_future_candidates']
    
    with torch.no_grad():
        out = decoder(batch_dict)
        
    print("    Inference pass successful!")
    print("    Selected world indices:", out['selected_world_idx'])
    print("    Collision costs:", out['collision_costs'].mean().item())
    
    print("\n" + "="*60)
    print("VERIFICATION COMPLETE: SYSTEM IS READY")
    print("="*60)

if __name__ == "__main__":
    test_causal_pipeline()
