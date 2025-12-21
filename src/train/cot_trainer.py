import os
import torch
from transformers import Trainer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class RobustCoTTrainer(Trainer):
    def __init__(self, alpha, total_steps, benign_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # incase
        self.args.remove_unused_columns = False
        self.current_training_step = 0
        self.alpha = alpha
        self.total_steps = total_steps
        self.benign_lambda = benign_lambda
    
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
        benign_inputs = {
            'input_ids': inputs.get('input_ids').to(device),
            'attention_mask': inputs.get('attention_mask').to(device),
            'labels': inputs.get('labels').to(device)
        }

        # Prepare retain inputs
        adv_inputs = {
            'input_ids': inputs.get('adv_input_ids').to(device),
            'attention_mask': inputs.get('adv_attention_mask').to(device),
            'labels': inputs.get('adv_labels').to(device)
        }

        # Return a dictionary containing both refusal and retain inputs
        return dict(
            benign_inputs=benign_inputs,
            adv_inputs=adv_inputs
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for both refusal and retain examples
        """
        self.current_training_step += 1

        benign_inputs = inputs['benign_inputs']
        adv_inputs = inputs['adv_inputs']
        
        # Calculate loss for refusal examples
        benign_outputs = model(**benign_inputs)
        benign_loss = benign_outputs.loss
        
        # Calculate loss for retain examples
        adv_outputs = model(**adv_inputs)
        adv_loss = adv_outputs.loss

        benign_coeff = self.benign_lambda
        harmful_coeff = self.harmful_lambda

        total_loss = benign_coeff * benign_loss + harmful_coeff * adv_loss
        logger.info(f"total_loss: {total_loss:.4f} || benign_loss/weighted: {benign_loss:.4f} {benign_coeff*benign_loss:.4f} || adv_loss/weighted: {adv_loss:.4f} {harmful_coeff*adv_loss:.4f}")
        if return_outputs:
            return (total_loss, {
                "benign_outputs": benign_outputs,
                "adv_outputs": adv_outputs
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
        
        benign_loss_sum = 0
        adv_loss_sum = 0
        num_batches = 0
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = self._prepare_inputs(batch)
                
                # Get losses for both types of examples
                benign_outputs = model(**inputs['benign_inputs'])
                adv_outputs = model(**inputs['adv_inputs'])
                
                benign_loss_sum += benign_outputs.loss.item()
                adv_loss_sum += adv_outputs.loss.item()
                num_batches += 1
        
        # Calculate average losses
        avg_benign_loss = benign_loss_sum / num_batches
        avg_adv_loss = adv_loss_sum / num_batches
        
        # Calculate total loss using the current training progress coefficients
        scheduled_coeff = self.get_training_progress()
        benign_coeff = self.alpha * scheduled_coeff
        refusal_coeff = self.alpha * (1 - scheduled_coeff)
        total_loss = retain_coeff * avg_retain_loss + refusal_coeff * avg_refusal_loss
        
        metrics = {
            "eval_loss": total_loss,
            "eval_benign_loss": avg_benign_loss,
            "eval_adv_loss": avg_adv_loss,
            "eval_adv_loss_weighted": harmful_coeff * avg_adv_loss,
            "eval_benign_loss_weighted": benign_coeff * avg_benign_loss,
        }
        
        return metrics
    

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
