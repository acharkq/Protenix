# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from typing import Dict, List


def convert_weights_original_to_fused(original_state_dict: Dict[str, torch.Tensor], 
                                    transition_prefixes: List[str]) -> Dict[str, torch.Tensor]:
    """
    Convert weights from original Transition format to FusedTransition format.
    
    Original Transition operation: F.silu(a) * b
    FusedTransition operation: F.silu(right_half) * left_half
    
    Therefore:
    - FusedTransition left_half should get linear_b weights (the multiplier)  
    - FusedTransition right_half should get linear_a weights (the one that gets SiLU)
    
    Args:
        original_state_dict (Dict[str, torch.Tensor]): State dict with original Transition weights
        transition_prefixes (List[str]): List of module prefixes for transitions to convert
                                       (e.g., ["pairformer_stack.blocks.0.pair_transition"])
    
    Returns:
        Dict[str, torch.Tensor]: Updated state dict with converted weights for FusedTransition
    """
    converted_dict = original_state_dict.copy()
    
    for module_prefix in transition_prefixes:
        # Keys for original transition
        ln_weight_key = f"{module_prefix}.layernorm1.weight"
        ln_bias_key = f"{module_prefix}.layernorm1.bias"
        linear_a_key = f"{module_prefix}.linear_no_bias_a.weight"  # Gets SiLU applied
        linear_b_key = f"{module_prefix}.linear_no_bias_b.weight"  # Gets multiplied
        linear_out_key = f"{module_prefix}.linear_no_bias.weight"
        
        # Keys for fused transition (when use_layernormlinear=False)
        fused_ln_weight_key = f"{module_prefix}.fused_transition.ff.0.weight"
        fused_ln_bias_key = f"{module_prefix}.fused_transition.ff.0.bias"
        fused_linear_combined_key = f"{module_prefix}.fused_transition.ff.1.weight"
        fused_linear_out_key = f"{module_prefix}.fused_transition.ff.3.weight"
        
        # Check if this transition exists in the state dict
        if all(key in original_state_dict for key in [ln_weight_key, ln_bias_key, 
                                                     linear_a_key, linear_b_key, linear_out_key]):
            
            print(f"Converting transition weights for: {module_prefix}")
            
            # Convert LayerNorm weights (direct mapping)
            converted_dict[fused_ln_weight_key] = original_state_dict[ln_weight_key]
            converted_dict[fused_ln_bias_key] = original_state_dict[ln_bias_key]
            
            # Combine the two parallel linear layers with correct order
            # Original: F.silu(a) * b
            # FusedTransition: F.silu(right_half) * left_half
            # So: left_half = b_weights, right_half = a_weights
            linear_a_weight = original_state_dict[linear_a_key]  # [n*c_in, c_in] - gets SiLU
            linear_b_weight = original_state_dict[linear_b_key]  # [n*c_in, c_in] - gets multiplied
            
            # Concatenate: [b_weights (left_half), a_weights (right_half)]
            combined_weight = torch.cat([linear_b_weight, linear_a_weight], dim=0)  # [2*n*c_in, c_in]
            converted_dict[fused_linear_combined_key] = combined_weight
            
            # Convert output linear layer (direct mapping)
            converted_dict[fused_linear_out_key] = original_state_dict[linear_out_key]
            
            # Remove original keys
            keys_to_remove = [ln_weight_key, ln_bias_key, linear_a_key, linear_b_key, linear_out_key]
            for key in keys_to_remove:
                if key in converted_dict:
                    del converted_dict[key]
    
    return converted_dict


def convert_weights_fused_to_original(fused_state_dict: Dict[str, torch.Tensor], 
                                    transition_prefixes: List[str]) -> Dict[str, torch.Tensor]:
    """
    Convert weights from FusedTransition format to original Transition format.
    
    Args:
        fused_state_dict (Dict[str, torch.Tensor]): State dict with FusedTransition weights
        transition_prefixes (List[str]): List of module prefixes for transitions to convert
    
    Returns:
        Dict[str, torch.Tensor]: Updated state dict with converted weights for original Transition
    """
    converted_dict = fused_state_dict.copy()
    
    for module_prefix in transition_prefixes:
        # Keys for fused transition
        fused_ln_weight_key = f"{module_prefix}.fused_transition.ff.0.weight"
        fused_ln_bias_key = f"{module_prefix}.fused_transition.ff.0.bias"
        fused_linear_combined_key = f"{module_prefix}.fused_transition.ff.1.weight"
        fused_linear_out_key = f"{module_prefix}.fused_transition.ff.3.weight"
        
        # Keys for original transition
        ln_weight_key = f"{module_prefix}.layernorm1.weight"
        ln_bias_key = f"{module_prefix}.layernorm1.bias"
        linear_a_key = f"{module_prefix}.linear_no_bias_a.weight"
        linear_b_key = f"{module_prefix}.linear_no_bias_b.weight"
        linear_out_key = f"{module_prefix}.linear_no_bias.weight"
        
        # Check if this fused transition exists in the state dict
        if all(key in fused_state_dict for key in [fused_ln_weight_key, fused_ln_bias_key,
                                                  fused_linear_combined_key, fused_linear_out_key]):
            
            print(f"Converting fused transition weights for: {module_prefix}")
            
            # Convert LayerNorm weights (direct mapping)
            converted_dict[ln_weight_key] = fused_state_dict[fused_ln_weight_key]
            converted_dict[ln_bias_key] = fused_state_dict[fused_ln_bias_key]
            
            # Split the combined linear layer back into two parallel layers
            combined_weight = fused_state_dict[fused_linear_combined_key]  # [2*n*c_in, c_in]
            n_features = combined_weight.shape[0] // 2
            
            # FusedTransition format: [left_half, right_half] = [b_weights, a_weights]
            linear_b_weight = combined_weight[:n_features]    # left_half -> linear_b
            linear_a_weight = combined_weight[n_features:]    # right_half -> linear_a
            
            converted_dict[linear_a_key] = linear_a_weight
            converted_dict[linear_b_key] = linear_b_weight
            
            # Convert output linear layer (direct mapping)
            converted_dict[linear_out_key] = fused_state_dict[fused_linear_out_key]
            
            # Remove fused keys
            keys_to_remove = [fused_ln_weight_key, fused_ln_bias_key, 
                             fused_linear_combined_key, fused_linear_out_key]
            for key in keys_to_remove:
                if key in converted_dict:
                    del converted_dict[key]
    
    return converted_dict


def find_transition_module_prefixes(state_dict: Dict[str, torch.Tensor]) -> List[str]:
    """
    Find all transition module prefixes in the state dict.
    
    Args:
        state_dict: Model state dictionary
        
    Returns:
        List of transition module prefixes
    """
    prefixes = set()
    
    for key in state_dict.keys():
        # Look for transition-related keys
        if any(suffix in key for suffix in ['.layernorm1.weight', '.linear_no_bias_a.weight', 
                                          '.fused_transition.ff.0.weight']):
            # Extract the module prefix (everything before the layer-specific part)
            if '.layernorm1.weight' in key:
                prefix = key.replace('.layernorm1.weight', '')
            elif '.linear_no_bias_a.weight' in key:
                prefix = key.replace('.linear_no_bias_a.weight', '')
            elif '.fused_transition.ff.0.weight' in key:
                prefix = key.replace('.fused_transition.ff.0.weight', '')
            else:
                continue
                
            prefixes.add(prefix)
    
    return sorted(list(prefixes))


def auto_convert_checkpoint_weights(checkpoint_path: str, target_format: str = "fused", 
                                  output_path: str = None) -> Dict[str, torch.Tensor]:
    """
    Automatically convert checkpoint between original and fused transition formats.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        target_format: "fused" to convert to FusedTransition, "original" to convert to original
        output_path: Optional path to save converted checkpoint
        
    Returns:
        Dictionary containing converted state dict
    """
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    
    # Find transition modules
    transition_prefixes = find_transition_module_prefixes(state_dict)
    print(f"Found {len(transition_prefixes)} transition modules to convert")
    
    # Convert weights
    if target_format == "fused":
        converted_state_dict = convert_weights_original_to_fused(state_dict, transition_prefixes)
        print("Converted to FusedTransition format")
    elif target_format == "original":
        converted_state_dict = convert_weights_fused_to_original(state_dict, transition_prefixes)
        print("Converted to original Transition format")
    else:
        raise ValueError(f"Invalid target_format: {target_format}. Must be 'fused' or 'original'")
    
    # Update checkpoint
    if 'model' in checkpoint:
        checkpoint['model'] = converted_state_dict
    else:
        checkpoint = converted_state_dict
    
    # Save if output path provided
    if output_path:
        torch.save(checkpoint, output_path)
        print(f"Saved converted checkpoint to: {output_path}")
    
    return checkpoint