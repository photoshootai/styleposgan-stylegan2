import wandb
from pytorch_lightning.callbacks import Callback
 
class LogPredictionsCallback(Callback):
    
    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        """Called when the validation batch ends."""
 
        # `outputs` comes from `LightningModule.validation_step`
        # which corresponds to our model predictions in this case
        
        # Let's log 20 sample image predictions from first batch
        if batch_idx == 0:
            n = 20
            x, y = batch
            # we use `LightningModule.log` method for logging
            pl_module.log('examples', [wandb.Image(x_i, caption=f'Ground Truth: {y_i}\nPrediction: {y_pred}')
                                       for x_i, y_i, y_pred in list(zip(x[:n], y[:n], outputs[:n]))])