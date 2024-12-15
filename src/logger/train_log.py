from transformers import TrainerCallback, TrainerState, TrainerControl
import logging

# logging the metadata of training with TrainerCallback
class LoggingCallback(TrainerCallback):
    def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
        if logs is not None:
            # Log the metadata')}")
            logging.info(f"Epoch: {state.epoch:.2f}, Loss: {logs.get('loss'):.4f}, Grad Norm: {logs.get('grad_norm'):.8f}, Learning Rate: {logs.get('learning_rate')}")