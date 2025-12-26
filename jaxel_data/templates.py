from typing import List, Any
from .config import TemplateConfig
from .utils import get_logger

logger = get_logger(__name__)

def get_template_config(template_name: str, tokenizer: Any) -> TemplateConfig:
    """
    Returns a TemplateConfig object populated with the token IDs 
    specific to the requested template_name and the provided tokenizer.
    """
    
    # Helper to safely tokenize a string into IDs (without special tokens like BOS)
    def _tok(s: str) -> List[int]:
        if not s: return []
        try: 
            return tokenizer.encode(s, add_special_tokens=False)
        except Exception as e: 
            logger.error(f"Template tokenization error for marker '{s}': {e}")
            return []

    name = template_name.lower()
    logger.debug(f"Getting template config for: '{name}'")
    
    if name == "gemma":
        return TemplateConfig(
            user_start_ids=_tok("<start_of_turn>user"), 
            user_end_ids=_tok("<end_of_turn>"), 
            assistant_start_ids=_tok("<start_of_turn>model"), 
            assistant_end_ids=_tok("<end_of_turn>")
        )
        
    elif name == "chatml":
        return TemplateConfig(
            system_start_ids=_tok("<|im_start|>system"), 
            system_end_ids=_tok("<|im_end|>"),
            user_start_ids=_tok("<|im_start|>user"), 
            user_end_ids=_tok("<|im_end|>"),
            assistant_start_ids=_tok("<|im_start|>assistant"), 
            assistant_end_ids=_tok("<|im_end|>")
        )
        
    elif name == "falcon":
        return TemplateConfig(
            system_start_ids=_tok("<|system|>"), 
            user_start_ids=_tok("<|user|>"),
            assistant_start_ids=_tok("<|assistant|>"), 
            assistant_end_ids=_tok("<|endoftext|>")
        )
        
    elif name == "llama3":
        return TemplateConfig(
            header_start_ids=_tok("<|start_header_id|>"), 
            header_end_ids=_tok("<|end_header_id|>"),
            turn_separator_ids=_tok("<|eot_id|>")
        )
        
    elif name == "deepseek":
        return TemplateConfig(
            ds_user_marker_ids=_tok("[INST]"), 
            ds_assistant_marker_ids=_tok("[/INST]"),
            ds_end_assistant_marker_ids=_tok("</s>")
        )
        
    elif name == "chat_glm":
        return TemplateConfig(
            system_start_ids=_tok("<|system|>"), 
            user_start_ids=_tok("<|user|>"),
            assistant_start_ids=_tok("<|assistant|>")
        )
        
    elif name == "granite":
        return TemplateConfig(
            header_start_ids=_tok("<|start_of_role|>"), 
            header_end_ids=_tok("<|end_of_role|>"),
            turn_separator_ids=_tok("<|end_of_text|>")
        )
        
    else:
        logger.warning(f"Unknown template name '{name}'. Returning empty template config.")
        return TemplateConfig()