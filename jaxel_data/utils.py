import re
import logging
from typing import List, Any

def get_logger(name: str) -> logging.Logger:
    """
    Configures and returns a logger with a standard format.
    Ensures we don't duplicate handlers if called multiple times.
    """
    logger = logging.getLogger(name)
    if not logger.hasHandlers():
        logging.basicConfig(
            level=logging.INFO, 
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    return logger

def find_sublist_indices(main_list: List[int], sub_list: List[int]) -> List[int]:
    """
    Finds all start indices of a sublist (pattern) within a main list.
    Used to locate where assistant tokens start/end for masking.
    """
    if not sub_list or not main_list or len(sub_list) > len(main_list):
        return []
    
    indices = []
    len_sub = len(sub_list)
    len_main = len(main_list)
    
    # Simple sliding window check
    for i in range(len_main - len_sub + 1):
        if main_list[i] == sub_list[0] and main_list[i:i+len_sub] == sub_list:
            indices.append(i)
            
    return indices

def clean_text(text: Any) -> str:
    """
    Sanitizes input text: fixes smart quotes, ellipses, and excessive newlines.
    """
    if not isinstance(text, str):
        try: 
            text = str(text)
        except Exception: 
            return ""
            
    # Normalize quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    
    # Normalize ellipsis
    text = text.replace('\u2026', '...')
    
    # Collapse 3+ newlines into 2
    text = re.sub(r'\n{3,}', '\n\n', text)
    
    return text.strip()