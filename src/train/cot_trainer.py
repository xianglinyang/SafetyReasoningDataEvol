import os
import shutil
import torch
from transformers import Trainer
from tqdm import tqdm
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

logger = logging.getLogger(__name__)

class SafetyCoTTrainer(Trainer):
    def __init__(self, alpha, total_steps, benign_lambda, harmful_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # incase
        self.args.remove_unused_columns = False
        self.current_training_step = 0
        self.alpha = alpha
        self.total_steps = total_steps
        self.benign_lambda = benign_lambda
        self.harmful_lambda = harmful_lambda
    
    def get_training_progress(self):
        # return self.current_training_step / self.total_steps
        return self.current_training_step / 1248/4
    
    def _prepare_inputs(self, inputs):
        """
        Prepares inputs for the model, ensuring both refusal and retain inputs are correctly formatted.
        """
        # Ensure inputs are on the correct device
        device = self.args.device

        # Prepare refusal inputs
        refusal_inputs = {
            'input_ids': inputs.get('input_ids').to(device),
            'attention_mask': inputs.get('attention_mask').to(device),
            'labels': inputs.get('labels').to(device)
        }

        # Prepare retain inputs
        retain_inputs = {
            'input_ids': inputs.get('retain_input_ids').to(device),
            'attention_mask': inputs.get('retain_attention_mask').to(device),
            'labels': inputs.get('retain_labels').to(device)
        }

        # Return a dictionary containing both refusal and retain inputs
        return dict(
            refusal_inputs=refusal_inputs,
            retain_inputs=retain_inputs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for both refusal and retain examples
        """
        self.current_training_step += 1

        refusal_inputs = inputs['refusal_inputs']
        retain_inputs = inputs['retain_inputs']
        
        # Calculate loss for refusal examples
        refusal_outputs = model(**refusal_inputs)
        refusal_loss = refusal_outputs.loss
        
        # Calculate loss for retain examples
        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss

        retain_coeff, refusal_coeff = self.benign_lambda, self.harmful_lambda

        total_loss = retain_coeff * retain_loss + refusal_coeff * refusal_loss
        logger.info(f"total_loss: {total_loss:.4f} || retain_loss/weighted: {retain_loss:.4f} {retain_coeff*retain_loss:.4f} || refusal_loss/weighted: {refusal_loss:.4f} {refusal_coeff*refusal_loss:.4f}")
        if return_outputs:
            return (total_loss, {
                "refusal_outputs": refusal_outputs,
                "retain_outputs": retain_outputs
            })
        return total_loss
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """
        Custom training step with potential safety-specific modifications
        """
        model.train()
        inputs = self._prepare_inputs(inputs)
        loss = self.compute_loss(model, inputs)
        
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
            
        loss.backward()
        
        return loss.detach()
    
    def evaluate(self, eval_dataset=None, ignore_keys=None):
        """
        Custom evaluation method that evaluates both refusal and retain examples separately
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model.eval()
        
        refusal_loss_sum = 0
        retain_loss_sum = 0
        num_batches = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = self._prepare_inputs(batch)
                
                # Get losses for both types of examples
                refusal_outputs = model(**inputs['refusal_inputs'])
                retain_outputs = model(**inputs['retain_inputs'])
                
                refusal_loss_sum += refusal_outputs.loss.item()
                retain_loss_sum += retain_outputs.loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_refusal_loss = refusal_loss_sum / num_batches
        avg_retain_loss = retain_loss_sum / num_batches
        
        # Calculate total loss using the current training progress coefficients
        scheduled_coeff = self.get_training_progress()
        retain_coeff = self.alpha * scheduled_coeff
        refusal_coeff = self.alpha * (1 - scheduled_coeff)
        total_loss = retain_coeff * avg_retain_loss + refusal_coeff * avg_refusal_loss
        
        metrics = {
            "eval_loss": total_loss,
            "eval_refusal_loss": avg_refusal_loss,
            "eval_retain_loss": avg_retain_loss,
            "eval_retain_loss_weighted": retain_coeff * avg_retain_loss,
            "eval_refusal_loss_weighted": refusal_coeff * avg_refusal_loss,
        }
        
        return metrics
    
    # def save_model(self, output_dir: str, _internal_call: bool = False):
    # bug: merge_and_unload modifies loaded_peft_model in-place and returns the base model
    #     merged_model = self.model.merge_and_unload()

    #     merged_model.save_pretrained(output_dir)
    #     self.tokenizer.save_pretrained(output_dir)

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
        if self.tokenizer is not None:
            tokenizer_save_path = os.path.join(output_dir, "tokenizer")
            try:
                logging.info(f"Saving tokenizer to {tokenizer_save_path}...")
                self.tokenizer.save_pretrained(tokenizer_save_path)
                logging.info("Tokenizer saved successfully.")
            except Exception as e:
                logging.error(f"Error saving tokenizer: {e}")
