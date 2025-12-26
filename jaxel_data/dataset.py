# D:\AI\JAX\data_pipeline\dataset.py

import numpy as np
from typing import List, Dict, Any, Optional, Iterator, Tuple
from .config import JAXelFlags
from .templates import get_template_config
from .utils import get_logger
from .core import process_conversation_content, generate_chat_labels, pack_sequences

logger = get_logger(__name__)

class JAXelDataset:
    def __init__(self, tokenizer: Any, dataset: List[Dict], flags: Optional[Dict] = None):
        """
        Initializes the dataset, processes all items, and prepares them for iteration.
        """
        self.tokenizer = tokenizer
        self.raw_dataset = dataset
        self.flags_dict = flags or {}
        
        # 1. Initialize Configuration
        self.config = JAXelFlags(
            max_length_chat=self.flags_dict.get('MAX_LENGTH_CHAT', 8192),
            max_length_completion=self.flags_dict.get('MAX_LENGTH_COMPLETION', 8192),
            label_masking_mode=self.flags_dict.get('LABEL_MASKING_MODE', 'all_assistant'),
            packing_enabled=self.flags_dict.get('PACKING', False),
            template_name=self.flags_dict.get('TEMPLATE', 'llama3').lower(),
            max_turns=self.flags_dict.get('MAX_TURNS', 17),
            ignore_index=self.flags_dict.get('IGNORE_INDEX', -100),
            append_system_to_user=self.flags_dict.get("APPEND_SYSTEM_TO_USER", False),
            include_assistant_start_marker_in_labels=self.flags_dict.get("LABEL_INCLUDE_ASSISTANT_START_MARKER", False),
            include_assistant_end_marker_in_labels=self.flags_dict.get("LABEL_INCLUDE_ASSISTANT_END_MARKER", True)
        )
        
        # Determine target length (for padding)
        self.padding_max_length = max(self.config.max_length_chat, self.config.max_length_completion)

        # 2. Tokenizer Setup
        self.special_tokens = self._configure_special_tokens()
        self.template = get_template_config(self.config.template_name, self.tokenizer)
        
        # Get newline token for formatting adjustments
        nl_ids = self.tokenizer.encode("\n", add_special_tokens=False)
        self.nl_id = nl_ids[:1] if nl_ids else []
        
        # 3. Pre-process Data
        # (For massive datasets, you might move this into __iter__ for streaming)
        logger.info(f"Processing {len(self.raw_dataset)} items with template '{self.config.template_name}'...")
        self.samples = self._prepare_data()
        logger.info(f"Final dataset size: {len(self.samples)} samples.")

    def _configure_special_tokens(self) -> Dict[str, Any]:
        """Extracts special tokens from flags or tokenizer."""
        bos_str = self.flags_dict.get('BOS_TOKEN')
        eos_str = self.flags_dict.get('EOS_TOKEN')
        pad_str = self.flags_dict.get('PAD_TOKEN')
        
        bos_ids = self.tokenizer.encode(bos_str, add_special_tokens=False) if bos_str else []
        eos_ids = self.tokenizer.encode(eos_str, add_special_tokens=False) if eos_str else []
        
        pad_id = self.tokenizer.pad_token_id
        if pad_str:
            ids = self.tokenizer.encode(pad_str, add_special_tokens=False)
            if len(ids) == 1: pad_id = ids[0]
            
        # Fallback for pad_id
        if pad_id is None:
            if self.tokenizer.eos_token_id is not None:
                pad_id = self.tokenizer.eos_token_id
                logger.warning(f"PAD token not found. Using EOS token ID ({pad_id}) as PAD.")
            else:
                pad_id = 0
                logger.warning("PAD and EOS tokens not found. Defaulting PAD to 0.")
        
        return {'bos': bos_ids, 'eos': eos_ids, 'pad': pad_id}

    def _prepare_data(self) -> List[Tuple[List[int], List[int]]]:
        """Runs the core processing logic on all items."""
        individual_samples = []
        
        for i, item_wrapper in enumerate(self.raw_dataset):
            if not isinstance(item_wrapper, dict): continue
            
            item_type = item_wrapper.get('type', 'chat') # Default to chat
            data = item_wrapper.get('data', {})
            
            token_ids, labels = None, None
            
            # --- Chat / Reasoning Processing ---
            if item_type in ['chat', 'reasoning']:
                res = process_conversation_content(
                    data.get("conversations") or data.get("messages"), 
                    self.config, self.template, self.tokenizer, 
                    self.special_tokens, self.nl_id
                )
                
                if res:
                    token_ids, last_content = res
                    # Reasoning Check
                    if item_type == 'reasoning' and '<think>' not in last_content:
                        continue 
                        
                    labels = generate_chat_labels(
                        token_ids, self.config, self.template, 
                        self.tokenizer, self.special_tokens, self.nl_id, item_type
                    )

            # --- Completion Processing (Simplistic) ---
            elif item_type == 'completion':
                text = data.get("text") or data.get("completion")
                if isinstance(text, str):
                    t_ids = self.tokenizer.encode(text, add_special_tokens=False)
                    token_ids = self.special_tokens['bos'] + t_ids + self.special_tokens['eos']
                    # Simple label generation (shift right is handled by model usually, here we just copy)
                    labels = list(token_ids) 
                    # Mask global BOS if requested
                    if self.config.mask_global_bos and self.special_tokens['bos']:
                        for k in range(len(self.special_tokens['bos'])): labels[k] = self.config.ignore_index
            
            # Add to list if valid
            if token_ids and labels and len(token_ids) == len(labels):
                individual_samples.append((token_ids, labels, item_type))

        # Packing
        if self.config.packing_enabled:
            logger.info("Packing enabled. Compressing samples...")
            return pack_sequences(
                individual_samples, 
                self.padding_max_length, 
                self.config.packing_separator_ids, 
                self.config.packing_separator_labels
            )
        
        # If no packing, just return the list stripping the type info
        return [(s[0], s[1]) for s in individual_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self) -> Iterator[Dict[str, np.ndarray]]:
        """
        Yields JAX-ready dictionaries containing NumPy arrays.
        """
        pad_id = self.special_tokens['pad']
        ignore_idx = self.config.ignore_index
        target_len = self.padding_max_length

        for seq, lbl in self.samples:
            current_len = len(seq)
            
            # Create fixed-size NumPy arrays (int32 is standard for JAX indices)
            input_ids = np.full((target_len,), pad_id, dtype=np.int32)
            labels = np.full((target_len,), ignore_idx, dtype=np.int32)
            attention_mask = np.zeros((target_len,), dtype=np.int32)
            
            # Fill content
            valid_len = min(current_len, target_len)
            input_ids[:valid_len] = seq[:valid_len]
            labels[:valid_len] = lbl[:valid_len]
            attention_mask[:valid_len] = 1
            
            yield {
                "input_ids": input_ids,
                "labels": labels,
                "attention_mask": attention_mask
            }
