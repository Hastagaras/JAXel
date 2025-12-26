from typing import List, Dict, Optional, Tuple, Any
from .config import TemplateConfig, JAXelFlags, ROLE_MAPPING
from .utils import get_logger, clean_text, find_sublist_indices

logger = get_logger(__name__)

def process_conversation_content(
    messages: List[Dict], 
    flags: JAXelFlags, 
    template: TemplateConfig, 
    tokenizer: Any,
    special_tokens: Dict[str, List[int]],
    nl_id: List[int]
) -> Optional[Tuple[List[int], str]]:
    """
    Processes a raw list of messages into a single list of token IDs based on the template.
    Returns: (all_token_ids, last_assistant_content_string) or None if invalid.
    """
    # 1. Validation & Cleaning
    validated_messages = []
    for msg in messages:
        if not isinstance(msg, dict) or "from" not in msg or "value" not in msg: continue
        role = str(msg.get("from", "")).lower()
        if role not in ROLE_MAPPING: continue
        mapped_role = ROLE_MAPPING[role]
        cleaned_val = clean_text(msg.get("value"))
        
        # Skip empty user/assistant turns
        if mapped_role in ["user", "assistant"] and not cleaned_val: continue
        validated_messages.append({"role": mapped_role, "value": cleaned_val})

    if not validated_messages: return None

    # 2. System Prompt Handling
    # Fix odd edge case: System followed immediately by Assistant (no user)
    if len(validated_messages) >= 2 and validated_messages[0]["role"] == "system" and validated_messages[1]["role"] == "assistant":
        validated_messages[0]["role"] = "user" 

    system_prompt = ""
    final_message_list = []
    
    # Extract system prompt if present
    if validated_messages[0]["role"] == "system":
        system_prompt = validated_messages.pop(0)["value"]
        # If template supports explicit system role, add it back as a turn
        if system_prompt and not flags.append_system_to_user and flags.template_name in ["chatml", "deepseek", "chat_glm", "granite", "llama3", "falcon"]:
            final_message_list.append({"role": "system", "value": system_prompt})
            system_prompt = ""

    first_non_system_added = False
    for msg in validated_messages:
        role, val = msg["role"], msg["value"]
        # Prepend system prompt to first user message if configured
        if system_prompt and flags.append_system_to_user and role == "user" and not first_non_system_added:
            val = f"{system_prompt}\n\n{val}".strip()
            system_prompt = ""
        final_message_list.append({"role": role, "value": val})
        if role != "system": first_non_system_added = True

    if not final_message_list: return None
    if final_message_list[-1]["role"] != "assistant": return None
    
    # 3. Truncation (Max Turns)
    # We only count User/Assistant pairs
    num_ua_turns = sum(1 for m in final_message_list if m['role'] in ['user', 'assistant'])
    if num_ua_turns > flags.max_turns:
        # Simple truncation: keep first N turns
        trunc_limit = flags.max_turns + (1 if final_message_list[0]['role'] == 'system' else 0)
        final_message_list = final_message_list[:trunc_limit]
        
        # Ensure we end with assistant
        if final_message_list and final_message_list[-1]['role'] == 'user':
            final_message_list.pop()
            
        if not final_message_list or final_message_list[-1]['role'] != 'assistant':
            return None

    original_last_assistant_content = final_message_list[-1]["value"]

    # 4. Tokenization Loop
    all_token_ids = list(special_tokens.get('bos', []))
    max_len_effective = flags.max_length_chat - len(special_tokens.get('eos', []))

    for i, turn in enumerate(final_message_list):
        role, content = turn["role"], turn["value"]
        content_ids = tokenizer.encode(content, add_special_tokens=False)
        turn_tokens = []
        is_last_turn = (i == len(final_message_list) - 1)
        
        # --- Template Logic ---
        if flags.template_name in ["gemma", "chatml"]:
            s_ids, e_ids = [], []
            if role == "system": s_ids, e_ids = template.system_start_ids, template.system_end_ids
            elif role == "user": s_ids, e_ids = template.user_start_ids, template.user_end_ids
            elif role == "assistant": s_ids, e_ids = template.assistant_start_ids, template.assistant_end_ids
            
            if s_ids or (role == "system" and content_ids):
                turn_tokens.extend(s_ids)
                # Newline after start marker
                if nl_id and s_ids: turn_tokens.extend(nl_id)
                turn_tokens.extend(content_ids)
                turn_tokens.extend(e_ids)
                # Separating newline
                if not is_last_turn and nl_id and e_ids:
                    if flags.template_name == "chatml" or flags.template_name == "gemma":
                        turn_tokens.extend(nl_id)

        elif flags.template_name == "llama3":
            role_ids = tokenizer.encode(role, add_special_tokens=False)
            nl2 = tokenizer.encode("\n\n", add_special_tokens=False)
            if content_ids or role == 'system':
                turn_tokens.extend(template.header_start_ids)
                turn_tokens.extend(role_ids)
                turn_tokens.extend(template.header_end_ids)
                turn_tokens.extend(nl2)
                turn_tokens.extend(content_ids)
                turn_tokens.extend(template.turn_separator_ids)

        elif flags.template_name == "deepseek":
            if role == "system":
                turn_tokens.extend(content_ids)
                if nl_id: turn_tokens.extend(nl_id)
            elif role == "user":
                turn_tokens.extend(template.ds_user_marker_ids)
                turn_tokens.extend(content_ids)
            elif role == "assistant":
                turn_tokens.extend(template.ds_assistant_marker_ids)
                turn_tokens.extend(content_ids)
                turn_tokens.extend(template.ds_end_assistant_marker_ids)

        elif flags.template_name == "falcon":
            s_ids = []
            if role == "system": s_ids = template.system_start_ids
            elif role == "user": s_ids = template.user_start_ids
            elif role == "assistant": s_ids = template.assistant_start_ids
            
            if content_ids or role == 'system':
                turn_tokens.extend(s_ids)
                if nl_id: turn_tokens.extend(nl_id)
                turn_tokens.extend(content_ids)
                if role == "assistant":
                    turn_tokens.extend(template.assistant_end_ids)
                if not is_last_turn and nl_id:
                    turn_tokens.extend(nl_id)

        # Fallback for unexpected or unimplemented templates
        if not turn_tokens and content_ids:
            logger.debug(f"Template logic resulted in empty tokens for {role}, using raw content.")
            turn_tokens = content_ids

        # Length Check
        if len(all_token_ids) + len(turn_tokens) > max_len_effective:
            remaining = max_len_effective - len(all_token_ids)
            if remaining > 0:
                all_token_ids.extend(turn_tokens[:remaining])
            break # Stop processing turns if full
        
        all_token_ids.extend(turn_tokens)

    # 5. Finalize with EOS
    eos_ids = special_tokens.get('eos', [])
    if eos_ids:
        all_token_ids.extend(eos_ids)
    
    return all_token_ids, original_last_assistant_content

def generate_chat_labels(
    sequence_ids: List[int], 
    flags: JAXelFlags, 
    template: TemplateConfig, 
    tokenizer: Any,
    special_tokens: Dict[str, List[int]],
    nl_id: List[int],
    sample_type: str
) -> List[int]:
    """Generates the label mask for the sequence."""
    seq_len = len(sequence_ids)
    labels = [flags.ignore_index] * seq_len
    
    mode = flags.label_masking_mode
    if sample_type == 'reasoning': 
        mode = 'final_assistant'

    if mode == 'full':
        return list(sequence_ids)

    # 1. Identify Assistant Start/End patterns based on template
    start_pattern = []
    end_pattern = []
    
    if flags.template_name in ["gemma", "chatml"]:
        start_pattern = template.assistant_start_ids
        end_pattern = template.assistant_end_ids
    elif flags.template_name == "llama3":
        role_tok = tokenizer.encode("assistant", add_special_tokens=False)
        start_pattern = template.header_start_ids + role_tok + template.header_end_ids
        end_pattern = template.turn_separator_ids
    elif flags.template_name == "deepseek":
        start_pattern = template.ds_assistant_marker_ids
        end_pattern = template.ds_end_assistant_marker_ids
    elif flags.template_name == "falcon":
        start_pattern = template.assistant_start_ids
        end_pattern = template.assistant_end_ids

    if not start_pattern:
        return labels # Return all masked if pattern not found

    # 2. Find Assistant Turns
    start_indices = find_sublist_indices(sequence_ids, start_pattern)
    if not start_indices:
        return labels

    target_indices = start_indices
    # If mode is 'final_assistant', only label the last one
    if mode == 'final_assistant':
        target_indices = start_indices[-1:]
    elif mode == 'mask_before_first_assistant':
        # Special case: Mask everything before first assistant, then show everything
        label_start = start_indices[0]
        if not flags.include_assistant_start_marker_in_labels:
            label_start += len(start_pattern)
        labels[label_start:] = sequence_ids[label_start:]
        return labels

    # 3. Apply Labels for selected turns
    start_offset = 0 if flags.include_assistant_start_marker_in_labels else len(start_pattern)

    for start_idx in target_indices:
        # Calculate where content starts
        label_start = start_idx + start_offset
        
        # Calculate where this turn ends
        search_slice = sequence_ids[label_start:]
        
        # Look for explicit end marker
        end_relative_indices = find_sublist_indices(search_slice, end_pattern) if end_pattern else []
        
        # Also look for next user turn as implicit end
        next_turn_start = []
        if flags.template_name in ["gemma", "chatml", "falcon"]: next_turn_start = template.user_start_ids
        elif flags.template_name == "llama3": 
             role_u = tokenizer.encode("user", add_special_tokens=False)
             next_turn_start = template.header_start_ids + role_u + template.header_end_ids
        
        next_turn_indices = find_sublist_indices(search_slice, next_turn_start) if next_turn_start else []
        
        # Determine strict end
        possible_ends = []
        if end_relative_indices: possible_ends.append(end_relative_indices[0])
        if next_turn_indices: possible_ends.append(next_turn_indices[0])
        
        if possible_ends:
            # End at the earliest marker found
            label_end = label_start + min(possible_ends)
            # If we ended on the explicit assistant end marker, maybe include it?
            if end_relative_indices and min(possible_ends) == end_relative_indices[0]:
                if flags.include_assistant_end_marker_in_labels:
                    label_end += len(end_pattern)
        else:
            label_end = seq_len # Implicit end at sequence end

        # Clamp
        label_start = min(max(0, label_start), seq_len)
        label_end = min(max(0, label_end), seq_len)

        if label_start < label_end:
            labels[label_start:label_end] = sequence_ids[label_start:label_end]

    # 4. Global BOS/EOS masking overrides
    bos = special_tokens.get('bos', [])
    if bos and flags.mask_global_bos and len(sequence_ids) >= len(bos) and sequence_ids[:len(bos)] == bos:
        for i in range(len(bos)): labels[i] = flags.ignore_index
        
    eos = special_tokens.get('eos', [])
    eos_len = len(eos)
    if eos and flags.mask_global_eos and len(sequence_ids) >= eos_len and sequence_ids[-eos_len:] == eos:
         for i in range(seq_len - eos_len, seq_len): labels[i] = flags.ignore_index

    return labels

def pack_sequences(
    samples: List[Tuple[List[int], List[int], str]], 
    max_len: int, 
    separator_ids: List[int], 
    separator_labels: List[int]
) -> List[Tuple[List[int], List[int]]]:
    """
    Greedy packing of samples into max_len.
    Returns list of (packed_tokens, packed_labels).
    """
    packed_results = []
    current_toks = []
    current_labs = []
    
    for tokens, labels, _ in samples:
        potential_toks = []
        potential_labs = []
        
        # Add separator if we are appending to existing content
        if current_toks:
            potential_toks.extend(separator_ids)
            potential_labs.extend(separator_labels)
        
        potential_toks.extend(tokens)
        potential_labs.extend(labels)
        
        # Check fit
        if len(current_toks) + len(potential_toks) <= max_len:
            current_toks.extend(potential_toks)
            current_labs.extend(potential_labs)
        else:
            # Commit current pack
            if current_toks:
                packed_results.append((list(current_toks), list(current_labs)))
            
            # Start new pack with current item
            current_toks = list(tokens)
            current_labs = list(labels)
            
    # Commit leftovers
    if current_toks:
        packed_results.append((current_toks, current_labs))
        
    return packed_results
