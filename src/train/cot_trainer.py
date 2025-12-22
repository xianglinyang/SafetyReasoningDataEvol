import os
import torch
from transformers import Trainer
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)

class RobustCoTTrainer(Trainer):
    def __init__(self, total_steps, benign_lambda, harmful_lambda, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # incase
        self.args.remove_unused_columns = False
        self.current_training_step = 0
        self.total_steps = total_steps
        self.benign_lambda = benign_lambda
        self.harmful_lambda = harmful_lambda
    
    def get_training_progress(self):
        return self.current_training_step / self.total_steps
    
    def _prepare_inputs(self, inputs):
        """
        Prepares inputs for the model, ensuring both refusal and retain inputs are correctly formatted.
        """
        # Ensure inputs are on the correct device
        device = self.args.device

        # Prepare normal inputs
        benign_inputs = {
            'input_ids': inputs.get('input_ids').to(device),
            'attention_mask': inputs.get('attention_mask').to(device),
            'labels': inputs.get('labels').to(device)
        }

        # Prepare adversarial inputs
        adv_inputs = {
            'input_ids': inputs.get('adv_input_ids').to(device),
            'attention_mask': inputs.get('adv_attention_mask').to(device),
            'labels': inputs.get('adv_labels').to(device)
        }

        # Get data type and adversarial mask
        data_types = inputs.get('data_type')  # List of 'benign' or 'harmful'
        is_adv = inputs.get('is_adv').to(device)  # Tensor mask for adversarial examples

        # Return a dictionary containing both normal and adv inputs
        return dict(
            benign_inputs=benign_inputs,
            adv_inputs=adv_inputs,
            data_types=data_types,
            is_adv=is_adv
        )
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """
        Custom loss computation for both benign and harmful examples with adversarial masking
        Loss formula: benign_lambda * (all normal losses) + harmful_lambda * (harmful adversarial losses)
        """
        self.current_training_step += 1

        benign_inputs = inputs['benign_inputs']
        adv_inputs = inputs['adv_inputs']
        # is_adv_mask = inputs['is_adv'].float()  # [batch_size], 1 if has adversarial mutation, 0 otherwise
        
        # Calculate logits for normal and adversarial examples
        normal_outputs = model(**benign_inputs)
        adv_outputs = model(**adv_inputs)

        normal_loss = normal_outputs.loss
        adv_loss = adv_outputs.loss

        benign_coeff = self.benign_lambda
        harmful_coeff = self.harmful_lambda

        # Combine losses: benign_lambda * (all normal) + harmful_lambda * (harmful adv only)
        total_loss = benign_coeff * normal_loss + harmful_coeff * adv_loss
    
        logger.info(f"total_loss: {total_loss:.4f} || "
                   f"normal_loss: {normal_loss:.4f} (w:{benign_coeff*normal_loss:.4f}) || "
                   f"adv_loss: {adv_loss:.4f} (w:{harmful_coeff*adv_loss:.4f})")
        
        if return_outputs:
            return (total_loss, {
                "normal_outputs": normal_outputs,
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
        Custom evaluation method that evaluates benign and harmful examples separately
        """
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        model = self.model.eval()
        
        all_normal_losses = []
        all_adv_losses = []
        all_is_adv = []
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
        
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            with torch.no_grad():
                inputs = self._prepare_inputs(batch)
                is_adv_mask = inputs['is_adv']
                
                benign_inputs = inputs['benign_inputs']
                adv_inputs = inputs['adv_inputs']
                batch_size = benign_inputs['input_ids'].shape[0]
                
                # Get logits for both normal and adversarial inputs
                normal_outputs = model(**benign_inputs)
                adv_outputs = model(**adv_inputs)
                
                normal_logits = normal_outputs.logits
                adv_logits = adv_outputs.logits
                
                normal_labels = benign_inputs['labels']
                adv_labels = adv_inputs['labels']
                
                # Compute per-sample losses
                normal_logits_flat = normal_logits.view(-1, normal_logits.size(-1))
                normal_labels_flat = normal_labels.view(-1)
                normal_loss_per_token = loss_fct(normal_logits_flat, normal_labels_flat)
                normal_loss_per_token = normal_loss_per_token.view(batch_size, -1)
                
                adv_logits_flat = adv_logits.view(-1, adv_logits.size(-1))
                adv_labels_flat = adv_labels.view(-1)
                adv_loss_per_token = loss_fct(adv_logits_flat, adv_labels_flat)
                adv_loss_per_token = adv_loss_per_token.view(batch_size, -1)
                
                # Mask out ignored tokens
                normal_mask = (normal_labels != -100).float()
                adv_mask = (adv_labels != -100).float()
                
                # Compute per-sample loss
                normal_loss_per_sample = (normal_loss_per_token * normal_mask).sum(dim=1) / (normal_mask.sum(dim=1) + 1e-10)
                adv_loss_per_sample = (adv_loss_per_token * adv_mask).sum(dim=1) / (adv_mask.sum(dim=1) + 1e-10)
                
                # Collect losses - move to CPU immediately to free GPU memory
                all_normal_losses.append(normal_loss_per_sample.cpu())
                all_adv_losses.append(adv_loss_per_sample.cpu())
                all_is_adv.append(is_adv_mask.cpu())
                
                # Clean up GPU tensors
                del normal_outputs, adv_outputs, normal_logits, adv_logits
                del normal_loss_per_token, adv_loss_per_token, normal_mask, adv_mask
                del normal_loss_per_sample, adv_loss_per_sample
        
        # Concatenate all batches (on CPU)
        all_normal_losses = torch.cat(all_normal_losses)
        all_adv_losses = torch.cat(all_adv_losses)
        all_is_adv = torch.cat(all_is_adv).float()
        
        # Vectorized computation
        avg_normal_loss = all_normal_losses.mean().item()
        
        harmful_adv_count = all_is_adv.sum().item()
        if harmful_adv_count > 0:
            avg_harmful_adv_loss = ((all_adv_losses * all_is_adv).sum() / harmful_adv_count).item()
        else:
            avg_harmful_adv_loss = 0.0
        
        # Calculate total loss using coefficients
        benign_coeff = self.benign_lambda
        harmful_coeff = self.harmful_lambda
        
        total_loss = benign_coeff * avg_normal_loss + harmful_coeff * avg_harmful_adv_loss
        
        # Record total samples before cleanup
        total_samples = len(all_normal_losses)
        
        # Clean up CPU tensors
        del all_normal_losses, all_adv_losses, all_is_adv
        
        metrics = {
            "eval_loss": total_loss,
            "eval_normal_loss": avg_normal_loss,
            "eval_adv_loss": avg_harmful_adv_loss,
            "eval_normal_loss_weighted": benign_coeff * avg_normal_loss,
            "eval_adv_loss_weighted": harmful_coeff * avg_harmful_adv_loss,
            "eval_adv_count": int(harmful_adv_count),
            "eval_total_samples": total_samples,
        }
        
        # Clear CUDA cache after evaluation to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
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
