import jax
import jax.numpy as jnp
from transformers import AutoConfig, FlaxAutoModelForCausalLM
from jax.sharding import NamedSharding

# Import from our new clean sibling file
from .sharding import create_jaxel_mesh, get_partition_rules, match_partition_rules

def load_jaxel_model(
    model_id: str, 
    model_axis: int = 2, 
    use_cache: bool = False, 
    dtype: jnp.dtype = jnp.bfloat16
):
    """
    Loads a Hugging Face model and shards it across JAX devices using JAXel rules.
    
    Args:
        model_id: HF Model ID (e.g. "Qwen/Qwen2.5-3B")
        model_axis: Tensor Parallelism degree (default 2)
        use_cache: Set True for inference, False for training.
        dtype: Computation data type (bfloat16 recommended for TPU).
        
    Returns:
        (sharded_params, model_def, mesh)
    """
    print(f"--- JAXel: Loading Model {model_id} ---")
    
    # 1. Initialize Hardware Mesh
    mesh = create_jaxel_mesh(model_axis=model_axis)

    # 2. Load Configuration
    config = AutoConfig.from_pretrained(model_id)
    config.use_cache = use_cache
    
    # 3. Load Weights to Host (CPU RAM)
    # We use from_pt=True so we don't need Flax-specific weights on HF Hub
    print("JAXel: Downloading/Loading parameters to Host (CPU)...")
    model_def = FlaxAutoModelForCausalLM.from_pretrained(
        model_id, 
        config=config, 
        dtype=dtype, 
        from_pt=True 
    )
    params = model_def.params
    print("JAXel: Parameters loaded to RAM.")

    # 4. Determine Sharding Strategy
    print("JAXel: Calculating partition specs...")
    rules = get_partition_rules(config)
    partition_specs = match_partition_rules(rules, params)

    # 5. Shard (Move from CPU -> TPU/GPU)
    print(f"JAXel: Sharding parameters across {jax.device_count()} devices...")
    
    with mesh:
        sharded_params = jax.tree_map(
            lambda p, spec: jax.device_put(p, NamedSharding(mesh, spec)),
            params,
            partition_specs
        )
        
    print("JAXel: Model successfully sharded and ready.")
    return sharded_params, model_def, mesh
