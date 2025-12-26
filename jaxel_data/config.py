from dataclasses import dataclass, field
from typing import List, Dict, Optional

@dataclass
class TemplateConfig:
    """
    Holds specific token IDs for chat template markers.
    Used to determine where turns start/end for masking labels.
    """
    system_start_ids: List[int] = field(default_factory=list)
    system_end_ids: List[int] = field(default_factory=list)
    user_start_ids: List[int] = field(default_factory=list)
    user_end_ids: List[int] = field(default_factory=list)
    assistant_start_ids: List[int] = field(default_factory=list)
    assistant_end_ids: List[int] = field(default_factory=list)
    header_start_ids: List[int] = field(default_factory=list)
    header_end_ids: List[int] = field(default_factory=list)
    turn_separator_ids: List[int] = field(default_factory=list)
    ds_user_marker_ids: List[int] = field(default_factory=list)
    ds_assistant_marker_ids: List[int] = field(default_factory=list)
    ds_end_assistant_marker_ids: List[int] = field(default_factory=list)

@dataclass
class JAXelFlags:
    """
    Consolidated configuration flags for the JAXel pipeline.
    """
    # Length configurations
    max_length_chat: int = 8192
    max_length_completion: int = 8192
    
    # Label Masking configurations
    label_masking_mode: str = 'all_assistant'  # Options: 'full', 'all_assistant', 'final_assistant', 'mask_before_first_assistant'
    include_assistant_start_marker_in_labels: bool = False
    include_assistant_end_marker_in_labels: bool = True
    mask_global_bos: bool = True
    mask_global_eos: bool = True
    
    # Chat specific configurations
    max_turns: int = 17
    template_name: str = 'llama3'
    append_system_to_user: bool = False
    
    # Packing configurations
    packing_enabled: bool = False
    packing_separator_str: str = ""
    packing_separator_ids: List[int] = field(default_factory=list)
    packing_separator_labels: List[int] = field(default_factory=list)
    
    # General
    ignore_index: int = -100

# Constant mapping for standardizing role names from dataset
ROLE_MAPPING = {
    "system": "system", 
    "human": "user", 
    "user": "user",
    "gpt": "assistant", 
    "assistant": "assistant", 
    "model": "assistant"
}