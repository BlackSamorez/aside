"""
ASIDE Model Architectures Implementation

This module contains the core architectural implementations for the ASIDE paper experiments:

1. ForwardRotMixin: Main ASIDE method (forward_rot) with π/2 rotation
2. ISEMixin: ISE baseline method with learnable segment embeddings  
3. Model-specific classes: Llama, Qwen, Mistral, Gemma variants
4. Legacy CustomLLaMA: Double embedding approach (double_emb)

Key Architecture Types:
- forward_rot: ASIDE method - applies orthogonal rotation to data token embeddings
- ise: ISE baseline - adds learnable segment embeddings to token embeddings
- single_emb: Vanilla baseline - standard model without architectural changes
- double_emb: Legacy approach - separate embedding matrices for instructions/data

The implementations follow the paper's methodology for creating distinct embedding
subspaces for instruction and data tokens without adding learnable parameters (ASIDE)
or with minimal parameter addition (ISE).

References:
    ASIDE: Architectural Separation of Instructions and Data in Language Models
    Section 3: Architecturally Separated Instruction-Data Embeddings
"""

import os

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PretrainedConfig,
)

from embeddings_init import (
    generate_isoclinic_rotation_matrix,
    rotate_embeddings_in_multiple_planes,
    rotate_embeddings_independently,
)


def texts_to_prepared_ids(
    text_sequences, tokenizer, max_length=512, model_type="double_emb"
):
    """
    Prepares tokenized inputs with instruction/data separation for ASIDE models.
    
    This function is central to ASIDE training and evaluation. It processes sequences
    of (text, type) pairs and creates the appropriate input representations based on
    the model architecture. For ASIDE models, it generates segment_ids that determine
    which tokens receive rotated embeddings.
    
    Args:
        text_sequences (List[Tuple[str, str]]): List of (text, role) pairs where:
                                               - text: The actual text content
                                               - role: "inst" for instructions, "data" for data
        tokenizer: HuggingFace tokenizer for the model
        max_length (int): Maximum sequence length (default: 512)
        model_type (str): Architecture type:
                         - "forward_rot": ASIDE method (main contribution)
                         - "ise": ISE baseline method
                         - "single_emb": Vanilla baseline
                         - "double_emb": Legacy double embedding
    
    Returns:
        tuple: (input_ids, attention_mask, segment_ids)
               - input_ids: Token IDs for the model
               - attention_mask: Attention mask (1 for real tokens, 0 for padding)
               - segment_ids: Role indicators (0 for instructions, 1 for data)
                             Used by ASIDE and ISE to determine embedding treatment
    
    Note:
        segment_ids are crucial for ASIDE as they determine which tokens get
        rotated embeddings during the forward pass. Only tokens with segment_id=1
        (data tokens) receive the π/2 rotation.
        
    Example:
        >>> sequences = [("Translate this text:", "inst"), ("Hello world", "data")]
        >>> ids, mask, segments = texts_to_prepared_ids(sequences, tokenizer, 512, "forward_rot")
        >>> # segment_ids will be [0, 0, 0, 1, 1] for instruction vs data tokens
    """
    remaining_length = max_length
    tokenized_seq = []

    # Tokenize each sequence while tracking remaining length budget
    for text_seq in text_sequences:
        tokenized = tokenizer(
            text_seq[0],
            return_tensors="pt",
            max_length=remaining_length,
            padding="longest",
            truncation=True,
        )
        tokenized_seq.append(tokenized)
        
        # Update remaining length budget
        seq_len = tokenized["input_ids"].shape[1]
        remaining_length -= seq_len
        
        if remaining_length <= 0:
            break

    # Process based on model architecture type
    if model_type == "single_emb":
        # Vanilla baseline: single sequence, no segment separation
        assert len(text_sequences) == 1
        input_ids, attention_mask = (
            tokenized_seq[0]["input_ids"],
            tokenized_seq[0]["attention_mask"],
        )
        segment_ids = None
        
    elif model_type == "double_emb":
        # Legacy double embedding: separate vocabulary spaces
        token_sequences = [
            (tokenized_seq[i]["input_ids"], text_sequences[i][1])
            for i in range(len(tokenized_seq))
        ]
        input_ids, attention_mask = prepare_input_ids(token_sequences, tokenizer)
        segment_ids = None
        
    elif model_type in ("ise", "forward_rot", "attention_rot", "position_shift"):
        # ASIDE and ISE: use segment_ids for role-based processing
        token_sequences = [
            (tokenized_seq[i]["input_ids"], text_sequences[i][1])
            for i in range(len(tokenized_seq))
        ]
        
        # Concatenate sequences and handle BOS tokens
        input_ids = torch.hstack(
            [tokenized_seq[i]["input_ids"] for i in range(len(tokenized_seq))]
        )
        # Replace BOS tokens with PAD tokens except in first sequence
        input_ids[1:] = torch.where(
            input_ids[1:] == tokenizer.bos_token_id,
            torch.full_like(input_ids[1:], tokenizer.pad_token_id),
            input_ids[1:],
        )
        
        attention_mask = torch.hstack(
            [tokenized_seq[i]["attention_mask"] for i in range(len(tokenized_seq))]
        )
        
        # Create segment_ids: 0 for instructions, 1 for data
        # This is what ASIDE uses to determine which embeddings to rotate
        segment_ids = torch.hstack(
            [
                torch.zeros_like(ts[0]) if (ts[1] == "inst") else torch.ones_like(ts[0])
                for ts in token_sequences
            ]
        )
    else:
        raise NotImplementedError(f"Not implemented for model type {model_type}")

    return input_ids, attention_mask, segment_ids


def prepare_input_ids(token_sequences, tokenizer):
    """
    Prepares input for double embedding models (legacy approach).
    
    This function handles the legacy double embedding approach where separate
    embedding matrices are used for instruction and data tokens. Data token IDs
    are shifted by vocab_size to index into the second embedding matrix.
    
    Args:
        token_sequences (List[Tuple[torch.Tensor, str]]): List of (input_ids, role) pairs
        tokenizer: HuggingFace tokenizer
        
    Returns:
        tuple: (input_ids, attention_mask) with shifted token IDs for data tokens
        
    Note:
        This is a legacy approach superseded by ASIDE's rotation method.
        Data tokens get their IDs shifted by vocab_size to access separate embeddings.
    """
    pad_token_id = tokenizer.pad_token_id
    bos_token_id = tokenizer.bos_token_id
    vocab_size = len(tokenizer)
    declared_vocab_size = tokenizer.vocab_size
    specials_vocab_size = vocab_size - declared_vocab_size
    
    if bos_token_id is None:
        raise ValueError("bos_token_id is None. Please ensure the tokenizer has a bos_token_id set.")

    batch_size = token_sequences[0][0].size(0)
    for input_ids, _ in token_sequences:
        assert input_ids.size(0) == batch_size, "All input_ids must have the same batch size"

    batch_input_ids = []
    batch_attention_masks = []

    for batch_idx in range(batch_size):
        non_pad_tokens_list = []
        total_pad_count = 0

        for seq_idx, (input_ids, token_type) in enumerate(token_sequences):
            seq = input_ids[batch_idx].clone()

            # Shift data token IDs to access second embedding matrix
            if token_type == "data":
                seq = seq + vocab_size
                seq_pad_token_id = pad_token_id + vocab_size
                seq_bos_token_id = bos_token_id + vocab_size if bos_token_id is not None else None
            else:
                seq_pad_token_id = pad_token_id
                seq_bos_token_id = bos_token_id
                
            # Handle BOS token positioning
            bos_positions = (seq == seq_bos_token_id).nonzero(as_tuple=False).squeeze(-1)
            if bos_positions.numel() > 0:
                bos_position = bos_positions[0].item()
                non_pad_tokens = seq[bos_position + 1 :]
            else:
                bos_position = -1
                non_pad_tokens = seq

            # Replace BOS with PAD in subsequent sequences
            if seq_idx > 0 and seq_bos_token_id is not None:
                seq[seq == seq_bos_token_id] = seq_pad_token_id

            pad_count = seq.size(0) - non_pad_tokens.size(0)
            total_pad_count += pad_count
            non_pad_tokens_list.append(non_pad_tokens)

        # Build final sequences with padding at beginning
        concatenated_non_pad_tokens = torch.cat(non_pad_tokens_list, dim=0)
        
        input_ids = torch.cat([
            torch.full((total_pad_count,), pad_token_id, 
                      dtype=concatenated_non_pad_tokens.dtype,
                      device=concatenated_non_pad_tokens.device),
            concatenated_non_pad_tokens,
        ], dim=0)

        attention_mask = torch.cat([
            torch.zeros(total_pad_count, dtype=torch.bool, 
                       device=concatenated_non_pad_tokens.device),
            torch.ones(concatenated_non_pad_tokens.size(0), dtype=torch.bool,
                      device=concatenated_non_pad_tokens.device),
        ], dim=0)

        batch_input_ids.append(input_ids)
        batch_attention_masks.append(attention_mask)

    # Concatenate batch and handle special tokens
    input_ids = torch.cat([elem.unsqueeze(0) for elem in batch_input_ids], dim=0)
    if specials_vocab_size > 0:
        input_ids[input_ids >= declared_vocab_size * 2 + specials_vocab_size] %= vocab_size

    attention_mask = torch.cat([elem.unsqueeze(0) for elem in batch_attention_masks], dim=0)
    return input_ids, attention_mask


########################
# Core Architecture Mixins
########################

class ISEMixin:
    """
    ISE (Instructional Segment Embedding) baseline implementation.
    
    This mixin implements the ISE method from Wu et al. (2024) as a baseline
    comparison to ASIDE. ISE adds learnable segment embeddings to token embeddings
    based on whether tokens are part of instructions or data.
    
    Key Differences from ASIDE:
    - Uses learnable parameters (segment embeddings) vs ASIDE's parameter-free rotation
    - Additive combination vs ASIDE's geometric transformation
    - Less effective separation in deeper layers (as shown in paper Section 6)
    
    Architecture:
        final_embedding = token_embedding + segment_embedding(segment_id)
        where segment_id = 0 for instructions, 1 for data
    """
    
    def forward(self, *args, input_ids=None, segment_ids=None, labels=None, **kwargs):
        """
        Forward pass with ISE segment embedding addition.
        
        Args:
            input_ids (torch.Tensor, optional): Token IDs 
            segment_ids (torch.Tensor): Segment role indicators (0=instruction, 1=data)
            labels (torch.Tensor, optional): Labels for language modeling loss
            **kwargs: Additional arguments passed to parent forward
            
        Returns:
            Model outputs with ISE-modified embeddings
            
        Note:
            segment_ids are required for ISE models to determine which tokens
            receive which type of segment embedding.
        """
        if segment_ids is None:
            raise ValueError("For ISE models, segment_ids can't be None")

        inputs_embeds = kwargs.pop("inputs_embeds", None)

        # Handle input_ids case (normal forward pass)
        if input_ids is not None:
            # Adjust segment_ids for single token generation (when using cache)
            if input_ids.shape[1] == 1:
                segment_ids = segment_ids[:, -1:]

            if segment_ids.shape != input_ids.shape:
                raise Exception(
                    f"Mismatched shapes: segment_ids {segment_ids.shape}, input_ids {input_ids.shape}"
                )

            # Get token embeddings and add segment embeddings
            if inputs_embeds is None:
                token_embeddings = self.model.embed_tokens(input_ids)
                seg_embeddings = self.segment_embedding(segment_ids)
                inputs_embeds = token_embeddings + seg_embeddings

        # Handle inputs_embeds case (e.g., from adversarial attacks)
        elif inputs_embeds is not None:
            if segment_ids.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"Segment IDs shape {segment_ids.shape} doesn't match inputs_embeds shape {inputs_embeds.shape[:2]}"
                )

            # Add segment embeddings to provided embeddings
            seg_embeddings = self.segment_embedding(segment_ids)
            inputs_embeds = inputs_embeds + seg_embeddings
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Clean up kwargs and forward to parent
        kwargs.pop("segment_ids", None)
        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)

        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, **kwargs
        )
        return outputs


class ForwardRotMixin:
    """
    ASIDE (forward_rot) method implementation - Main contribution of the paper.
    
    This mixin implements the core ASIDE architecture that applies orthogonal rotations
    to data token embeddings. It creates geometrically separated embedding subspaces
    for instructions and data without adding any learnable parameters.
    
    Key Features:
    - Parameter-free: Uses fixed orthogonal rotation matrices (typically π/2)
    - Geometric separation: Creates distinct subspaces via isoclinic rotation
    - Conditional application: Only rotates embeddings where segment_ids == 1 (data)
    - Direction control: Supports left/right matrix multiplication
    
    Mathematical Operation:
        For data tokens (segment_id == 1):
            rotated_embedding = embedding @ rotation_matrix  (if direction="right")
            rotated_embedding = rotation_matrix @ embedding  (if direction="left")
        For instruction tokens (segment_id == 0):
            embedding remains unchanged
            
    This implements Section 3 of the ASIDE paper: "Architecturally Separated 
    Instruction-Data Embeddings"
    """
    
    def forward(self, *args, input_ids=None, segment_ids=None, labels=None, **kwargs):
        """
        Forward pass with ASIDE rotation applied to data embeddings.
        
        This is the core ASIDE forward pass that applies π/2 rotation to data token
        embeddings while leaving instruction token embeddings unchanged. This creates
        the geometric separation that enables better instruction-data distinction.
        
        Args:
            input_ids (torch.Tensor, optional): Token IDs for embedding lookup
            segment_ids (torch.Tensor): Role indicators (0=instruction, 1=data)
                                       Determines which tokens get rotated
            labels (torch.Tensor, optional): Labels for language modeling loss
            **kwargs: Additional arguments, including optional inputs_embeds
            
        Returns:
            Model outputs with ASIDE-rotated embeddings for data tokens
            
        Note:
            The rotation is only applied where segment_ids == 1 (data tokens).
            Instruction tokens (segment_ids == 0) use standard embeddings.
            This selective application is key to ASIDE's effectiveness.
        """
        if segment_ids is None:
            raise ValueError("For ForwardRot models, segment_ids can't be None")

        inputs_embeds = kwargs.get("inputs_embeds", None)

        # Case 1: Standard forward pass with input_ids
        if input_ids is not None:
            # Handle single token generation (when using KV cache)
            if input_ids.shape[1] == 1:
                segment_ids = segment_ids[:, -1:]

            if segment_ids.shape != input_ids.shape:
                raise Exception(
                    f"Mismatched shape of segment_ids:{segment_ids.shape, input_ids.shape}"
                )

            if inputs_embeds is None:
                inputs_embeds = self.model.embed_tokens(input_ids)

                # Apply ASIDE rotation only to data tokens (segment_ids == 1)
                mask = segment_ids == 1

                if self.rotation_direction == "right":
                    # Right multiplication: embedding @ rotation_matrix
                    new_embeds = inputs_embeds.clone()
                    new_embeds[mask] = torch.matmul(
                        inputs_embeds[mask], self.rotation_matrix
                    )
                    inputs_embeds = new_embeds
                elif self.rotation_direction == "left":
                    # Left multiplication: rotation_matrix @ embedding
                    new_embeds = inputs_embeds.clone()
                    new_embeds[mask] = (self.rotation_matrix @ inputs_embeds[mask].T).T
                    inputs_embeds = new_embeds
                else:
                    raise ValueError(
                        f"Unknown rotation_direction: {self.rotation_direction}"
                    )

                # Optional: Add linear shift (experimental feature)
                if self.add_linear_shift:
                    seg_embeddings = self.segment_embedding(segment_ids)
                    inputs_embeds = inputs_embeds + seg_embeddings

        # Case 2: Direct inputs_embeds case (e.g., adversarial evaluation)
        elif inputs_embeds is not None:
            if segment_ids.shape != inputs_embeds.shape[:2]:
                raise Exception(
                    f"Mismatched shape of segment_ids:{segment_ids.shape} vs inputs_embeds:{inputs_embeds.shape[:2]}"
                )

            # Apply rotation to data embeddings
            mask = segment_ids == 1
            new_embeds = inputs_embeds.clone()

            if self.rotation_direction == "right":
                new_embeds[mask] = torch.matmul(
                    inputs_embeds[mask], self.rotation_matrix
                )
            elif self.rotation_direction == "left":
                new_embeds[mask] = (self.rotation_matrix @ inputs_embeds[mask].T).T
            else:
                raise ValueError(
                    f"Unknown rotation_direction: {self.rotation_direction}"
                )

            inputs_embeds = new_embeds

            # Optional linear shift
            if self.add_linear_shift:
                seg_embeddings = self.segment_embedding(segment_ids)
                inputs_embeds = inputs_embeds + seg_embeddings

        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Clean up processed arguments
        kwargs.pop("inputs_embeds", None)
        kwargs.pop("segment_ids", None)
        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)

        # Forward to parent model with rotated embeddings
        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, **kwargs
        )

        return outputs
    

from transformers import AttentionInterface
from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS

def rotated_attention_forward(
    segment_ids: torch.Tensor,
    rotation_matrix: torch.Tensor,
    module: nn.Module,
    query: torch.Tensor,  # [batch_size, num_heads, seq_len, hidden_size]
    key: torch.Tensor,  # [batch_size, num_heads_kv, seq_len, hidden_size]
    value: torch.Tensor,  # [batch_size, num_heads_kv, seq_len, hidden_size]
    attention_mask: torch.Tensor,
    scaling: float,
    dropout: float = 0.0,
    **kwargs,
):
    data_mask = (segment_ids == 1).unsqueeze(1).unsqueeze(-1)  # [batch_size, 1, seq_len, 1]
    query = torch.where(
        data_mask,
        F.linear(query.reshape(-1, rotation_matrix.shape[1]), rotation_matrix).view_as(query),
        query
    )
    key = torch.where(
        data_mask,
        F.linear(key.reshape(-1, rotation_matrix.shape[1]), rotation_matrix).view_as(key),
        key
    )
    
    return ALL_ATTENTION_FUNCTIONS[module.config.orig_attention_implementation](
        module, query, key, value, attention_mask, scaling, dropout, **kwargs
    )


class AttentionRotMixin:    
    def forward(self, *args, position_ids=None,input_ids=None, segment_ids=None, labels=None, **kwargs):
        if segment_ids is None:
            raise ValueError("For AttentionRot models, segment_ids can't be None")

        inputs_embeds = kwargs.pop("inputs_embeds", None)

        # Handle input_ids case (normal forward pass)
        if input_ids is not None:
            # Adjust segment_ids for single token generation (when using cache)
            if input_ids.shape[1] == 1:
                segment_ids = segment_ids[:, -1:]

            if segment_ids.shape != input_ids.shape:
                raise Exception(
                    f"Mismatched shapes: segment_ids {segment_ids.shape}, input_ids {input_ids.shape}"
                )

            # Get token embeddings and add segment embeddings
            if inputs_embeds is None:
                AttentionInterface.register(
                    "rotated_attention",
                    lambda *args, **kwargs: rotated_attention_forward(segment_ids, self.rotation_matrix, *args, **kwargs),
                )
                inputs_embeds = self.model.embed_tokens(input_ids)

        # Handle inputs_embeds case (e.g., from adversarial attacks)
        elif inputs_embeds is not None:
            if segment_ids.shape[:2] != inputs_embeds.shape[:2]:
                raise ValueError(
                    f"Segment IDs shape {segment_ids.shape} doesn't match inputs_embeds shape {inputs_embeds.shape[:2]}"
                )

            AttentionInterface.register(
                "rotated_attention",
                lambda *args, **kwargs: rotated_attention_forward(segment_ids, self.rotation_matrix, *args, **kwargs),
            )
            inputs_embeds = inputs_embeds
        else:
            raise ValueError("Either input_ids or inputs_embeds must be provided")

        # Clean up kwargs and forward to parent
        kwargs.pop("segment_ids", None)
        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)

        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, **kwargs
        )
        return outputs
    

class PositionShiftMixin:
    def forward(
        self, *args,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        past_key_values=None,
        inputs_embeds=None, 
        segment_ids=None, labels=None,
        **kwargs
    ):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.model.embed_tokens(input_ids)

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            ).unsqueeze(0)
            
        nonzero_mask = segment_ids != 0
        cumsum = nonzero_mask.cumsum(dim=-1)
        shift_mask = cumsum > 0
        
        if inputs_embeds.shape[1] == 1:
            # Always shift when generating
            shift_mask = torch.where(shift_mask[:, -1:], True, True)
        
        position_shift = shift_mask.to(torch.int64) * self.data_shift
        position_ids = position_ids + position_shift
        
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        ) + self.data_shift
        kwargs["cache_position"] = cache_position
        
        kwargs.pop("segment_ids", None)
        if hasattr(self.model, "delete_num_items_in_batch"):
            kwargs.pop("num_items_in_batch", None)
            
        outputs = super().forward(
            *args, input_ids=None, inputs_embeds=inputs_embeds, labels=labels, position_ids=position_ids, past_key_values=past_key_values, **kwargs
        )
        return outputs


##################
# Llama Model Implementations
##################

from transformers.models.llama.configuration_llama import LlamaConfig
from transformers.models.llama.modeling_llama import LlamaForCausalLM


class CustomLlamaConfig(LlamaConfig):
    """Extended Llama configuration for ASIDE experiments."""
    model_type = "llama"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class LlamaBase(LlamaForCausalLM):
    """
    Base Llama model with tokenizer customization for ASIDE experiments.
    
    This base class handles tokenizer setup for Llama models, ensuring proper
    padding and special token configuration across Llama-2 and Llama-3 variants.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_name_or_path):
        """
        Customizes tokenizer for ASIDE experiments with proper special tokens.
        
        Args:
            tokenizer: HuggingFace tokenizer to customize
            model_name_or_path (str): Model path to determine Llama version
            
        Sets up:
        - Padding tokens (different for Llama-2 vs Llama-3)
        - EOS tokens with proper IDs
        - Ensures compatibility across model variants
        """
        # Set padding tokens based on Llama version
        if tokenizer.pad_token is None:
            if "llama-3" in model_name_or_path.lower() or "llama_3" in model_name_or_path.lower():
                tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(
                    "<|reserved_special_token_3|>"
                )
            elif "llama-2" in model_name_or_path.lower() or "llama_2" in model_name_or_path.lower():
                tokenizer.pad_token_id = tokenizer.unk_token_id
                tokenizer.pad_token = tokenizer.unk_token
            else:
                raise NotImplementedError("Pad token supports only llama now, fix")
        else:
            tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
            
        # Set EOS tokens based on Llama version  
        if tokenizer.eos_token_id is None:
            if "llama-3" in model_name_or_path.lower() or "llama_3" in model_name_or_path.lower():
                tokenizer.eos_token = "<|end_of_text|>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("<|end_of_text|>")
            elif "llama-2" in model_name_or_path.lower() or "llama_2" in model_name_or_path.lower():
                tokenizer.eos_token = "</s>"
                tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
            else:
                raise NotImplementedError("Pad token supports only llama 3 and 2 now, fix")


class LlamaISE(ISEMixin, LlamaBase):
    """
    Llama model with ISE baseline implementation.
    
    Combines the Llama base model with ISE segment embeddings for baseline
    comparison with ASIDE. Used in paper experiments as one of the baselines.
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class LlamaForwardRot(ForwardRotMixin, LlamaBase):
    """
    Llama model with ASIDE (ForwardRot) implementation - Main experimental model.
    
    This is the main ASIDE implementation for Llama models used in the paper.
    It applies π/2 orthogonal rotations to data token embeddings while keeping
    instruction embeddings unchanged.
    
    Key Configuration:
    - rotation_alpha: Rotation angle (π/2 for standard ASIDE)
    - gradual_rotation: Whether to gradually increase rotation during training
    - learned_rotation: Whether rotation matrix is learnable (typically False)
    - rotation_direction: "right" or "left" matrix multiplication
    - add_linear_shift: Whether to add ISE-style segment embeddings (typically False)
    """
    
    def __init__(self, config: LlamaConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Optional segment embeddings for linear shift experiments
        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        # Initialize rotation matrix
        if self.gradual_rotation:
            # Start with no rotation for gradual training
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            # Use full rotation angle (typically π/2)
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
            
        # Register as parameter or buffer based on configuration
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix)


###########
# Qwen Model Implementations  
###########

from transformers import Qwen2Config, Qwen2ForCausalLM


class CustomQwenConfig(Qwen2Config):
    """Extended Qwen configuration for ASIDE experiments."""
    model_type = "qwen2"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class QwenBase(Qwen2ForCausalLM):
    """Base Qwen model with tokenizer customization."""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_path):
        """Customizes Qwen tokenizer for ASIDE experiments."""
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<|endoftext|>")
        tokenizer.bos_token = "<|endoftext|>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids(tokenizer.bos_token)
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


class QwenISE(ISEMixin, QwenBase):
    """Qwen model with ISE baseline implementation."""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class QwenForwardRot(ForwardRotMixin, QwenBase):
    """Qwen model with ASIDE implementation."""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
            
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix)


class QwenAttentionRot(AttentionRotMixin, QwenBase):
    """Qwen model with ASIDE implementation."""
    
    def __init__(self, config: Qwen2Config):
        config.orig_attention_implementation = config._attn_implementation
        config._attn_implementation = "rotated_attention"
        super().__init__(config)
        self.config = config
        
        head_dim = config.hidden_size // config.num_attention_heads
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        rotation_matrix = generate_isoclinic_rotation_matrix(
            head_dim, self.global_rotation_alpha, device, model_dtype
        ).detach()
        
        self.register_buffer("rotation_matrix", rotation_matrix)


class QwenPositionShift(PositionShiftMixin, QwenBase):
    """Qwen model with ASIDE implementation."""
    
    def __init__(self, config: Qwen2Config):
        super().__init__(config)
        self.config = config
        self.data_shift = config.data_shift


###########
# Mistral Model Implementations
###########

from transformers import MistralConfig, MistralForCausalLM


class CustomMistralConfig(MistralConfig):
    """Extended Mistral configuration for ASIDE experiments."""
    model_type = "mistral"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


class MistralBase(MistralForCausalLM):
    """Base Mistral model with tokenizer customization."""
    
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.delete_num_items_in_batch = True

    @classmethod
    def _customize_tokenizer(cls, tokenizer, model_path):
        """Customizes Mistral tokenizer for ASIDE experiments."""
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token


class MistralISE(ISEMixin, MistralBase):
    """Mistral model with ISE baseline implementation."""
    
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class MistralForwardRot(ForwardRotMixin, MistralBase):
    """Mistral model with ASIDE implementation."""
    
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
            
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix)


###########
# Gemma Model Implementations
###########

import json
import os
from huggingface_hub import hf_hub_download
from transformers import (
    Gemma3Config,
    Gemma3ForCausalLM,
    Gemma3ForConditionalGeneration,
    Gemma3TextConfig,
)
from transformers.models.gemma.modeling_gemma import GemmaForCausalLM


class CustomGemmaConfig(Gemma3TextConfig):
    """Extended Gemma configuration for ASIDE experiments."""
    model_type = "gemma3_text"


class GemmaBase(Gemma3ForCausalLM):
    """Base Gemma model with configuration handling."""
    
    def __init__(self, config: Gemma3TextConfig):
        config.attention_bias = True
        super().__init__(config)
        print("Config in GemmaBase init:", config)

    @staticmethod
    def _customize_tokenizer(tokenizer, model_path):
        """Customizes Gemma tokenizer for ASIDE experiments."""
        tokenizer.pad_token = "<pad>"
        tokenizer.pad_token_id = tokenizer.convert_tokens_to_ids("<pad>")
        tokenizer.bos_token = "<s>"
        tokenizer.bos_token_id = tokenizer.convert_tokens_to_ids("<s>")
        tokenizer.eos_token = "</s>"
        tokenizer.eos_token_id = tokenizer.convert_tokens_to_ids("</s>")
        tokenizer.padding_side = "left"
        print("Special tokens", tokenizer.convert_tokens_to_ids(["<pad>", "<s>", "</s>"]))
        assert tokenizer.convert_tokens_to_ids("<pad>") >= 0, "Invalid <pad> token"


class GemmaISE(ISEMixin, GemmaBase):
    """Gemma model with ISE baseline implementation."""
    
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.num_segments = getattr(config, "num_segments", 2)
        self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)


class GemmaForwardRot(ForwardRotMixin, GemmaBase):
    """Gemma model with ASIDE implementation."""
    
    def __init__(self, config: Gemma3TextConfig):
        super().__init__(config)
        self.config = config
        dim = config.hidden_size
        self.global_rotation_alpha = config.rotation_alpha
        self.rotation_alpha = None
        self.gradual_rotation = getattr(config, "gradual_rotation", False)
        self.learned_rotation = getattr(config, "learned_rotation", False)
        self.add_linear_shift = getattr(config, "add_linear_shift", False)
        self.rotation_direction = getattr(self.config, "rotation_direction", "right")
        device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if self.add_linear_shift:
            self.num_segments = getattr(config, "num_segments", 2)
            self.segment_embedding = nn.Embedding(self.num_segments, config.hidden_size)

        if self.gradual_rotation:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, 0, device, model_dtype
            ).detach()
        else:
            rotation_matrix = generate_isoclinic_rotation_matrix(
                dim, self.global_rotation_alpha, device, model_dtype
            ).detach()
            
        if self.learned_rotation:
            self.rotation_matrix = nn.Parameter(rotation_matrix)
        else:
            self.register_buffer("rotation_matrix", rotation_matrix)
