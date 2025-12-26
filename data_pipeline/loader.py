import logging
import random
from typing import Dict, Any, List, Optional, Tuple, Union
from datasets import load_dataset, Dataset, concatenate_datasets, IterableDataset
from .utils import get_logger

logger = get_logger(__name__)

# --- Standardization Logic ---

def standardize_format(
    example: Dict[str, Any], 
    dataset_type: str = 'chat', 
    text_column: str = 'text',
    system_column: Optional[str] = None, 
    prompt_column: str = 'prompt', 
    response_column: str = 'response'
) -> Optional[Dict[str, Any]]:
    """
    Standardizes examples to the format expected by JAXConversationDataset.
    Returns: {'conversations': [...]} or {'text': ...}
    """
    if dataset_type == 'completion':
        text = example.get(text_column)
        if isinstance(text, str) and text.strip():
            return {"text": text.strip()}
        return None

    # Process 'chat' and 'reasoning'
    elif dataset_type in ['chat', 'reasoning']:
        conversations = None
        processed_conv = []
        valid = True

        # 1. Try standard 'conversations'/'messages' list format
        source_list = example.get('conversations') or example.get('messages') or example.get('dialogs')
        if isinstance(source_list, list):
            for msg in source_list:
                if isinstance(msg, dict):
                    role = msg.get('role') or msg.get('from')
                    value = msg.get('content') or msg.get('value')
                    
                    if role is not None and value is not None:
                        role_str = str(role).lower().strip()
                        value_str = str(value).strip()
                        if role_str and value_str:
                            processed_conv.append({"from": role_str, "value": value_str})
                        else: 
                            valid = False; break
                    else: 
                        valid = False; break
                else: 
                    valid = False; break
            
            if valid and processed_conv:
                conversations = processed_conv

        # 2. Fallback: Try 'prompt'/'response' columns
        elif not conversations:
            prompt_val = example.get(prompt_column)
            response_val = example.get(response_column)
            system_val = example.get(system_column) if system_column else None

            if isinstance(prompt_val, str) and isinstance(response_val, str):
                prompt_str, response_str = prompt_val.strip(), response_val.strip()
                system_str = str(system_val).strip() if system_val is not None else ""

                if prompt_str and response_str:
                    conv = []
                    if system_str: conv.append({"from": "system", "value": system_str})
                    conv.append({"from": "human", "value": prompt_str})
                    conv.append({"from": "assistant", "value": response_str})
                    conversations = conv

        return {"conversations": conversations} if conversations else None

    # Process 'single_turn' (converts to chat format)
    elif dataset_type == 'single_turn':
        prompt_val = example.get(prompt_column)
        response_val = example.get(response_column)
        system_val = example.get(system_column) if system_column else None

        if isinstance(prompt_val, str) and isinstance(response_val, str):
            prompt_str, response_str = prompt_val.strip(), response_val.strip()
            system_str = str(system_val).strip() if system_val is not None else ""

            if prompt_str and response_str:
                conv = []
                if system_str: conv.append({"from": "system", "value": system_str})
                conv.append({"from": "human", "value": prompt_str})
                conv.append({"from": "assistant", "value": response_str})
                return {"conversations": conv}
                
    return None

# --- Dataset Loading & Processing ---

def process_dataset(ds_name: str, ds_config: Dict[str, Any], seed: int) -> Optional[Dataset]:
    """Loads, samples, and standardizes a single dataset from Hugging Face."""
    try:
        dtype = ds_config.get('type', 'chat')
        limit = ds_config.get('limit')
        stream = ds_config.get('stream', False)
        load_args = {"name": ds_config.get('config_name')} if ds_config.get('config_name') else {}

        # Column mapping
        text_col = ds_config.get('text_column', 'text')
        system_col = ds_config.get('system_column')
        prompt_col = ds_config.get('prompt_column', 'prompt')
        response_col = ds_config.get('response_column', 'response')

        logger.info(f"Loading: {ds_name} (Type: {dtype}, Limit: {limit})")

        # Load
        try:
            dataset = load_dataset(ds_name, split='train', streaming=stream, **load_args)
        except Exception as e:
            logger.error(f"Failed to load {ds_name}: {e}")
            return None

        # Limit & Shuffle
        if stream:
            if limit: dataset = dataset.take(limit)
        else:
            dataset = dataset.shuffle(seed=seed)
            if limit and limit < len(dataset):
                dataset = dataset.select(range(limit))

        # Standardize
        valid_examples = []
        skipped = 0
        
        # Determine output type tag (single_turn becomes chat)
        output_type = 'chat' if dtype == 'single_turn' else dtype

        # Iterate and Transform
        # Note: For very large datasets, map() is faster, but this manual loop 
        # is safer for complex standardization logic across heterogenous sources.
        for ex in dataset:
            std = standardize_format(
                ex, dataset_type=dtype, text_column=text_col,
                system_column=system_col, prompt_column=prompt_col, response_column=response_col
            )
            if std:
                valid_examples.append({"data": std, "type": output_type})
            else:
                skipped += 1

        logger.info(f"  -> {len(valid_examples)} valid examples ({skipped} skipped).")

        if not valid_examples:
            return None

        if not stream:
            random.shuffle(valid_examples)

        return Dataset.from_list(valid_examples)

    except Exception as e:
        logger.error(f"Error processing {ds_name}: {e}", exc_info=True)
        return None

def interleave_datasets(datasets_with_names: List[Tuple[Dataset, str]]) -> Dataset:
    """Interleaves multiple datasets into one."""
    active_iters = {}
    active_sources = []
    
    for ds, name in datasets_with_names:
        if ds and len(ds) > 0:
            active_iters[name] = iter(ds)
            active_sources.append(name)

    if not active_sources:
        return Dataset.from_list([])

    interleaved_data = []
    
    while active_sources:
        sources_to_drop = []
        for source in list(active_sources):
            try:
                item = next(active_iters[source])
                # We can optionally inject the source name here if needed for debugging
                # item['source'] = source 
                interleaved_data.append(item)
            except StopIteration:
                sources_to_drop.append(source)
            except Exception as e:
                logger.error(f"Error reading from {source}: {e}")
                sources_to_drop.append(source)
        
        for s in sources_to_drop:
            if s in active_sources: active_sources.remove(s)

    random.shuffle(interleaved_data)
    return Dataset.from_list(interleaved_data)

def load_and_prepare_datasets(
    datasets_config: Dict[str, Dict], 
    seed: int = 42, 
    test_size: float = 0.05
) -> Tuple[Dataset, Optional[Dataset]]:
    """
    Main entry point. Loads all datasets in config, splits them, 
    and returns (train_dataset, eval_dataset).
    """
    random.seed(seed)
    train_splits = []
    eval_splits = []

    for name, config in datasets_config.items():
        ds = process_dataset(name, config, seed)
        if not ds: continue

        # Attempt Split
        try:
            # Only split if we have enough data (e.g., > 100 samples)
            if len(ds) > 100 and test_size > 0:
                split = ds.train_test_split(test_size=test_size, seed=seed)
                train_splits.append((split['train'], name))
                eval_splits.append(split['test'])
                logger.info(f"  -> Split {name}: Train={len(split['train'])}, Eval={len(split['test'])}")
            else:
                train_splits.append((ds, name))
                logger.info(f"  -> Used all {len(ds)} for training (too small to split).")
        except Exception as e:
            logger.warning(f"Split failed for {name}: {e}. Using all for training.")
            train_splits.append((ds, name))

    # Merge
    logger.info("Interleaving Training Data...")
    final_train = interleave_datasets(train_splits)
    
    final_eval = None
    if eval_splits:
        logger.info("Concatenating Eval Data...")
        final_eval = concatenate_datasets(eval_splits).shuffle(seed=seed)

    return final_train, final_eval
