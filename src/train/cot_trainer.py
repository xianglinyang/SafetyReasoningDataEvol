import torch
from transformers import Trainer
from tqdm import tqdm

class SafetyCoTTrainer(Trainer):
    def __init__(self, alpha, total_steps, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.current_training_step = 0
        self.alpha = alpha
        self.total_steps = total_steps
    
    def get_training_progress(self):
        return self.current_training_step / self.total_steps
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for both refusal and retain examples
        """
        self.current_training_step += 1

        refusal_inputs = inputs["refusal_inputs"]
        retain_inputs = inputs["retain_inputs"]
        
        # Calculate loss for refusal examples
        refusal_outputs = model(**refusal_inputs)
        refusal_loss = refusal_outputs.loss
        
        # Calculate loss for retain examples
        retain_outputs = model(**retain_inputs)
        retain_loss = retain_outputs.loss

        # coeff
        scheduled_coeff = self.get_training_progress()
        retain_coeff, refusal_coeff = self.alpha * scheduled_coeff, self.alpha * (1-scheduled_coeff)
        total_loss = retain_coeff * retain_loss + refusal_coeff * refusal_loss
        print(f"retain_coeff: {retain_coeff:.4f} || refusal_coeff: {refusal_coeff:.4f}")
        print(f"total_loss: {total_loss:.4f} || retain_loss: {retain_loss:.4f} || refusal_loss: {refusal_loss:.4f}")
        
        if return_outputs:
            return (total_loss, {
                "refusal_outputs": refusal_outputs,
                "retain_outputs": retain_outputs
            })
        return total_loss
    
    def training_step(self, model, inputs):
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
                refusal_outputs = model(**inputs["refusal_inputs"])
                retain_outputs = model(**inputs["retain_inputs"])
                
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
