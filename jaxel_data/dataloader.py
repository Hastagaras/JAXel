import os
import logging
import numpy as np
import jax
from typing import Optional, Dict, Any, Iterator, Tuple
from datasets import load_dataset, load_from_disk, Dataset as HFDataset

# Import our JAXel dataset wrapper
from .dataset import JAXelDataset

# Setup a named logger for this module
logger = logging.getLogger("JAXelDataLoader")
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

class PretokenizedDatasetAdapter:
    """
    Adapts a pre-tokenized Hugging Face dataset to behave like JAXelDataset.
    Ensures __getitem__ returns NumPy arrays and handles formatting.
    """
    def __init__(self, hf_dataset: HFDataset):
        self.data = hf_dataset
        # We assume the dataset columns are already correct (input_ids, labels, attention_mask)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        # Explicitly cast to int32 numpy arrays for JAX
        return {
            "input_ids": np.array(item["input_ids"], dtype=np.int32),
            "labels": np.array(item["labels"], dtype=np.int32),
            "attention_mask": np.array(item["attention_mask"], dtype=np.int32)
        }

class JAXelDataLoader:
    """
    A simple, pure-Python DataLoader that yields batches of NumPy arrays.
    Replaces torch.utils.data.DataLoader.
    """
    def __init__(self, dataset, batch_size: int, shuffle: bool = True, drop_last: bool = True, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.seed = seed
        self.epoch = 0

    def __len__(self):
        if self.drop_last:
            return len(self.dataset) // self.batch_size
        else:
            return (len(self.dataset) + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """Yields collated batches."""
        indices = np.arange(len(self.dataset))
        
        if self.shuffle:
            # Deterministic shuffling based on epoch + seed
            rng = np.random.default_rng(self.seed + self.epoch)
            rng.shuffle(indices)
        
        # Generator loop
        for start_idx in range(0, len(indices), self.batch_size):
            end_idx = start_idx + self.batch_size
            
            if self.drop_last and end_idx > len(indices):
                break
            
            # Allow partial batches if drop_last is False
            if end_idx > len(indices):
                end_idx = len(indices)
                
            batch_indices = indices[start_idx:end_idx]
            
            # Collate (Stack individual items into a batch)
            # This fetches item [0], item [1], etc. from JAXelDataset
            batch_items = [self.dataset[i] for i in batch_indices]
            
            collated_batch = {
                key: np.stack([item[key] for item in batch_items])
                for key in batch_items[0]
            }
            
            yield collated_batch
        
        self.epoch += 1

def setup_jaxel_dataloaders(
    train_dataset_raw, 
    eval_dataset_raw, 
    tokenizer, 
    flags: Dict[str, Any]
) -> Tuple[JAXelDataLoader, Optional[JAXelDataLoader]]:
    """
    Main entry point to setup JAXel data loaders.
    Handles 'USE_PRETOKENIZED_DATA' logic and JAX global batch size calculation.
    """
    logger.info("Starting JAXel dataloader setup...")
    
    use_pretokenized = flags.get('USE_PRETOKENIZED_DATA', False)
    pretokenized_path = flags.get('PRETOKENIZED_DATA_PATH', None)
    
    # --- 1. Determine Global Batch Size ---
    # In JAX (pmap), we feed a global batch that covers all local devices.
    per_device_batch = flags.get("BATCH_SIZE", 1)
    
    # Check if we are running on JAX devices (TPU/GPU) or CPU fallback
    try:
        device_count = jax.local_device_count() 
    except:
        device_count = 1

    global_batch_size = per_device_batch * device_count
    
    logger.info(f"Per-device batch: {per_device_batch}, Local Devices: {device_count}")
    logger.info(f"Global Training Batch Size: {global_batch_size}")
    
    # Update flags for downstream reference
    flags['GLOBAL_BATCH_SIZE'] = global_batch_size

    train_data_obj = None
    eval_data_obj = None

    # --- 2. Load Data (Pre-tokenized or On-the-fly) ---
    if use_pretokenized:
        logger.info(f"Attempting to load pre-tokenized data from: {pretokenized_path}")
        if not pretokenized_path:
            raise ValueError("USE_PRETOKENIZED_DATA is True, but PRETOKENIZED_DATA_PATH is missing.")

        try:
            # Handle Local Disk vs Hugging Face Hub
            if os.path.isdir(pretokenized_path):
                train_path = os.path.join(pretokenized_path, "train")
                eval_path = os.path.join(pretokenized_path, "eval")
                
                if not os.path.isdir(train_path):
                    raise FileNotFoundError(f"Train path not found: {train_path}")
                
                # Load and wrap
                train_ds_hf = load_from_disk(train_path)
                train_data_obj = PretokenizedDatasetAdapter(train_ds_hf)
                
                if os.path.isdir(eval_path):
                    eval_ds_hf = load_from_disk(eval_path)
                    eval_data_obj = PretokenizedDatasetAdapter(eval_ds_hf)
                else:
                    logger.warning(f"Eval path not found: {eval_path}")
            else:
                # Load from Hub
                train_ds_hf = load_dataset(pretokenized_path, split='train')
                train_data_obj = PretokenizedDatasetAdapter(train_ds_hf)
                
                # Try finding validation split
                for split_name in ['validation', 'test']:
                    try:
                        eval_ds_hf = load_dataset(pretokenized_path, split=split_name)
                        eval_data_obj = PretokenizedDatasetAdapter(eval_ds_hf)
                        break
                    except Exception:
                        pass

            logger.info("Pre-tokenized data loaded successfully.")

        except Exception as e:
            logger.error(f"Failed to load pre-tokenized data: {e}", exc_info=True)
            raise

    else:
        # On-the-fly Processing
        logger.info("Processing data on-the-fly using JAXelDataset.")
        if train_dataset_raw is None:
            raise ValueError("USE_PRETOKENIZED_DATA is False, but 'train_dataset_raw' is None.")

        # Train config 
        train_flags_cfg = flags.copy()
        # Ensure PACKING is explicitly handled (default to False if not in flags)
        train_flags_cfg['PACKING'] = flags.get('PACKING', False) 
        
        train_data_obj = JAXelDataset(tokenizer, train_dataset_raw, train_flags_cfg)

        if eval_dataset_raw:
            eval_flags_cfg = flags.copy()
            eval_flags_cfg['PACKING'] = False # Usually don't pack eval
            eval_data_obj = JAXelDataset(tokenizer, eval_dataset_raw, eval_flags_cfg)

    # --- 3. Create JAXel Data Loaders ---
    if not train_data_obj or len(train_data_obj) == 0:
        raise ValueError("Training dataset is empty.")

    logger.info(f"Training samples: {len(train_data_obj)}")
    
    # Train Loader
    train_loader = JAXelDataLoader(
        train_data_obj,
        batch_size=global_batch_size,
        shuffle=True,
        drop_last=True,
        seed=flags.get('SEED', 42)
    )
    
    # Eval Loader
    eval_loader = None
    if eval_data_obj and len(eval_data_obj) > 0:
        logger.info(f"Validation samples: {len(eval_data_obj)}")
        # Calculate eval batch size (can be same as train or larger)
        val_batch_size = flags.get("VAL_BATCH", per_device_batch) * device_count
        
        eval_loader = JAXelDataLoader(
            eval_data_obj,
            batch_size=val_batch_size,
            shuffle=False,
            drop_last=True
        )

    logger.info(f"Train Steps per Epoch: {len(train_loader)}")
    
    # Update flags for the training loop
    flags['STEPS'] = len(train_loader)
    
    return train_loader, eval_loader