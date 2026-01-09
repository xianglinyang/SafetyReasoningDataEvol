import os
import torch
from transformers import Trainer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class R2DTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
    
    def compute_loss(self, model, inputs):

        input_ids = inputs.get("input_ids").to(self.args.device)
        labels = inputs.get("labels").to(self.args.device)

        outputs = model(input_ids=input_ids)

        logits = outputs.logits.float()
        vocab_size = logits.shape[-1]
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(shift_logits.device)
        
        loss = torch.nn.functional.cross_entropy(shift_logits, shift_labels, ignore_index=-100, reduction="mean")

        safe_index = vocab_size - 8
        unsafe_index = vocab_size - 8 + 1
        # rethink_index = vocab_size - 8 + 2

        indices_safe = (shift_labels == safe_index)
        indices_unsafe = (shift_labels == unsafe_index)

        if indices_safe.any():
            logits_diff_safe = shift_logits[indices_safe, safe_index] - shift_logits[indices_safe, unsafe_index]
            safe_loss = -torch.nn.functional.logsigmoid(logits_diff_safe).mean()
            loss += safe_loss
            logger.info(f"safe_loss: {safe_loss:.4f}")
        
        if indices_unsafe.any():
            logits_diff_unsafe = shift_logits[indices_unsafe, unsafe_index] - shift_logits[indices_unsafe, safe_index]
            unsafe_loss = -torch.nn.functional.logsigmoid(logits_diff_unsafe).mean()
            loss += unsafe_loss
            logger.info(f"unsafe_loss: {unsafe_loss:.4f}")
        
        logger.info(f"total_loss: {loss:.4f}")

        return loss

    def save_model(self, output_dir: str, _internal_call: bool = False):
        """
        Event called after a checkpoint save. Saves PEFT adapter weights.
        """
        # The Trainer automatically saves the full state (optimizer, scheduler etc)
        # in the checkpoint_dir. We only need to save the PEFT adapter weights
        # using save_pretrained within that directory.

        # Save PEFT adapter model
        peft_save_path = os.path.join(output_dir, "adapter_model")
        try:
            logging.info(f"Saving PEFT adapter weights to {peft_save_path}...")
            self.model.save_pretrained(peft_save_path)
            logging.info("PEFT adapter weights saved successfully.")
        except Exception as e:
            logging.error(f"Error saving PEFT adapter model: {e}")

        # Save Tokenizer (often needed with the model)
        # Use processing_class (new API) or tokenizer (old API) for compatibility
        tokenizer = getattr(self, 'processing_class', None) or getattr(self, 'tokenizer', None)
        if tokenizer is not None:
            tokenizer_save_path = os.path.join(output_dir, "tokenizer")
            try:
                logging.info(f"Saving tokenizer to {tokenizer_save_path}...")
                tokenizer.save_pretrained(tokenizer_save_path)
                logging.info("Tokenizer saved successfully.")
            except Exception as e:
                logging.error(f"Error saving tokenizer: {e}")




