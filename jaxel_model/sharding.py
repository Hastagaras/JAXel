import re
import jax
import numpy as np
from jax.sharding import PartitionSpec as P
from flax.traverse_util import flatten_dict, unflatten_dict
from transformers import (
    LlamaConfig, MistralConfig, Qwen2Config, Gemma2Config, Phi3Config, GraniteConfig, Qwen3Config
)

# --- Sharding Constants ---
# 'mp' = Model/Tensor Parallel (Splitting large matrices across cores)
# 'fsdp' = Fully Sharded Data Parallel (Splitting the batch/optimizer states)
# None = Replicated (Copied to all devices)

# Standard sharding rules for Decoder-only transformers (Llama, Qwen, Mistral)
STANDARD_DECODER_RULES = (
    # Embeddings: Split hidden dimension (mp), Data parallel on batch (fsdp)
    ("embed_tokens/embedding", P("mp", "fsdp")),
    
    # Attention Q/K/V Projections [hidden, heads * head_dim]
    ("self_attn/(q_proj|k_proj|v_proj)/kernel", P("fsdp", "mp")),
    
    # Attention Output Projection [heads * head_dim, hidden]
    ("self_attn/o_proj/kernel", P("mp", "fsdp")),
    
    # MLP Gate/Up Projections [hidden, intermediate]
    ("mlp/(gate_proj|up_proj)/kernel", P("fsdp", "mp")),
    
    # MLP Down Projection [intermediate, hidden]
    ("mlp/down_proj/kernel", P("mp", "fsdp")),
    
    # LM Head (Final output layer)
    ("lm_head/kernel", P("fsdp", "mp")),
    
    # Normalization & Biases (Small tensors, usually FSDP sharded or replicated)
    ("norm/kernel", P("fsdp")), 
    ("bias", P("fsdp")),
)

# Map Hugging Face Config classes to our rules
CONFIG_TO_RULES = [
    (Qwen2Config, STANDARD_DECODER_RULES),
    (Qwen3Config, STANDARD_DECODER_RULES),
    (LlamaConfig, STANDARD_DECODER_RULES),
    (MistralConfig, STANDARD_DECODER_RULES),
    (Gemma2Config, STANDARD_DECODER_RULES),
    (Phi3Config, STANDARD_DECODER_RULES),
    (GraniteConfig, STANDARD_DECODER_RULES),
]

def create_jaxel_mesh(model_axis: int = 2) -> jax.sharding.Mesh:
    """
    Creates a 2D Device Mesh for JAXel (FSDP, MP).
    
    Args:
        model_axis: Number of devices to use for Tensor Parallelism (MP).
                    (e.g., 2 on a v3-8 TPU leaves 4 for FSDP).
    """
    total_devices = jax.device_count()
    
    if total_devices % model_axis != 0:
        raise ValueError(
            f"JAXel Error: Total devices ({total_devices}) is not divisible "
            f"by model_axis ({model_axis})."
        )
        
    fsdp_axis = total_devices // model_axis
    devices = np.array(jax.devices()).reshape(fsdp_axis, model_axis)
    
    # We define axis names 'fsdp' and 'mp' here, referenced in the rules above.
    mesh = jax.sharding.Mesh(devices, ('fsdp', 'mp'))
    print(f"JAXel Mesh Initialized: {mesh.shape} (FSDP={fsdp_axis}, MP={model_axis})")
    return mesh

def get_partition_rules(config):
    """Finds the correct partition rules for a given model config."""
    for config_class, rule in CONFIG_TO_RULES:
        if isinstance(config, config_class):
            return rule
    
    print(f"JAXel Warning: No specific sharding rules found for {type(config)}. Defaulting to FSDP-only.")
    # Fallback: Shard everything on the data parallel axis
    return [("(.*)", P("fsdp"))]

def match_partition_rules(rules, params):
    """
    Applies regex rules to the Flax parameter tree.
    Returns: A PyTree of PartitionSpecs.
    """
    # Flatten keys to slash-separated paths: "model/layers/0/self_attn/q_proj/kernel"
    flat_params = flatten_dict(params, sep='/')
    flat_specs = {}

    for param_name in flat_params.keys():
        matched_spec = P("fsdp") # Default safety net
        
        for pattern, spec in rules:
            if re.search(pattern, param_name):
                matched_spec = spec
                break
        
        flat_specs[param_name] = matched_spec

    return unflatten_dict(flat_specs)
