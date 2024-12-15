from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

# logging the metadata of training with TrainerCallback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            # Safely get values with defaults
            epoch = state.epoch if state.epoch is not None else 0
            loss = logs.get('loss', 0.0)
            grad_norm = logs.get('grad_norm', 0.0)
            learning_rate = logs.get('learning_rate', 0.0)
            
            # For training logs
            if 'loss' in logs:
                logging.info(
                    f"Epoch: {epoch:.2f}, "
                    f"Loss: {loss:.4f}, "
                    f"Grad Norm: {grad_norm:.8f}, "
                    f"Learning Rate: {learning_rate}"
                )
            
            # For evaluation logs
            if 'eval_loss' in logs:
                eval_loss = logs.get('eval_loss', 0.0)
                logging.info(
                    f"Evaluation - "
                    f"Epoch: {epoch:.2f}, "
                    f"Loss: {eval_loss:.4f}"
                )