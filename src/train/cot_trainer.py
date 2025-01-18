import torch
from transformers import Trainer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class SafetyCoTTrainer(Trainer):
    def __init__(self, alpha, total_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # incase
        self.args.remove_unused_columns = False
        self.current_training_step = 0
        self.alpha = alpha
        self.total_steps = total_steps
    
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

        # coeff
        scheduled_coeff = self.get_training_progress()
        retain_coeff, refusal_coeff = scheduled_coeff+0.2, (1-scheduled_coeff)+0.2
        logger.info(f"retain_coeff: {retain_coeff:.4f} || refusal_coeff: {refusal_coeff:.4f}")
        
        retain_coeff = retain_coeff if retain_coeff > 0.1 else 0.1
        refusal_coeff = refusal_coeff if refusal_coeff > 0.1 else 0.1

        # retain_coeff, refusal_coeff = 0.5, 0.5

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
