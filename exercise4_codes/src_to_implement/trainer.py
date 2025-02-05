import torch as t
import torch.optim as optim
from sklearn.metrics import f1_score
from tqdm.autonotebook import tqdm


class Trainer:
    def __init__(self,
                 model,                        # Model to be trained.
                 crit,                         # Loss function
                 optim=None,                   # Optimizer
                 train_dl=None,                # Training data set
                 val_test_dl=None,             # Validation (or test) data set
                 cuda=True,                    # Whether to use the GPU
                 early_stopping_patience=-1):  # The patience for early stopping
        self._model = model
        self._crit = crit
        self._optim = optim
        self._train_dl = train_dl
        self._val_test_dl = val_test_dl
        self._cuda = cuda

        self._early_stopping_patience = early_stopping_patience

        if cuda:
            self._model = model.cuda()
            self._crit = crit.cuda()
        if self._optim:
            self.scheduler = t.optim.lr_scheduler.ReduceLROnPlateau(
                self._optim, mode='min', factor=0.7, patience=4, verbose=True
            )
        else:
            self.scheduler = None  # Avoids error when exporting
    def save_checkpoint(self, epoch):
        t.save({'state_dict': self._model.state_dict()},
               f'checkpoints/checkpoint_{epoch:03d}.ckp')

    def restore_checkpoint(self, epoch_n):
        ckp = t.load(f'checkpoints/checkpoint_{epoch_n:03d}.ckp',
                     map_location='cuda' if self._cuda else 'cpu')
        self._model.load_state_dict(ckp['state_dict'])

    def save_onnx(self, fn):
        m = self._model.cpu()
        m.eval()
        x = t.randn(1, 3, 300, 300, requires_grad=True)
        _ = m(x)  # Forward pass to shape inference
        t.onnx.export(
            m,                 # model being run
            x,                 # model input
            fn,                # ONNX file to save
            export_params=True,        # store the trained parameter weights
            opset_version=10,          # the ONNX version
            do_constant_folding=True,  # optimize constant folds
            input_names=['input'],  
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
        )

    def train_step(self, x, y):
        """
        - Reset/clear gradients
        - Forward pass
        - Compute loss
        - Backprop (loss.backward)
        - Optimizer step
        - Return the loss (float)
        """
        self._optim.zero_grad()
        outputs = self._model(x)
        loss = self._crit(outputs, y)
        loss.backward()
        self._optim.step()
        return loss.item()

    def val_test_step(self, x, y):
        """
        - Forward pass
        - Compute loss
        - Return (loss (float), outputs (tensor))
        """
        with t.no_grad():
            outputs = self._model(x)
            loss = self._crit(outputs, y)
        return loss.item(), outputs

    def train_epoch(self):
        """
        - Set the model to training mode
        - Iterate over the training loader
          * Move data to GPU if needed
          * Call train_step
        - Return the average training loss over the epoch
        """
        self._model.train()
        total_loss = 0.0
        for x, y in self._train_dl:
            if self._cuda:
                x = x.cuda()
                y = y.cuda()

            loss = self.train_step(x, y)
            total_loss += loss

        avg_loss = total_loss / len(self._train_dl)
        return avg_loss

    def val_test(self,epoch):
        """
        - Set the model to eval mode
        - Disable gradient computation
        - Iterate over the validation (or test) loader
          * Move data to GPU if needed
          * Call val_test_step
        - Compute and return average validation loss
        - (Optional) Compute and print a metric, e.g. F1
        """
        self._model.eval()
        total_loss = 0.0

        all_preds = []
        all_labels = []

        with t.no_grad():
            for x, y in self._val_test_dl:
                if self._cuda:
                    x = x.cuda()
                    y = y.cuda()

                loss, outputs = self.val_test_step(x, y)
                total_loss += loss

                # Collect predictions and labels for metrics
                all_preds.append(outputs.cpu())
                all_labels.append(y.cpu())

        avg_loss = total_loss / len(self._val_test_dl)

        # Example: Compute F1 score for a 2-class (or multi-label) problem
        # If your model outputs 2 values with Sigmoid, treat them as two independent probabilities.
        preds_tensor = t.cat(all_preds, dim=0)
        labels_tensor = t.cat(all_labels, dim=0)

        # Threshold at 0.5 for each of the 2 outputs (multi-label scenario)
        preds_bin = (preds_tensor > 0.5).int()
        labels_bin = labels_tensor.int()

        # Macro-F1 across both outputs
        f1 = f1_score(labels_bin, preds_bin, average='macro')

        print(f"[Epoch {epoch:02d}] [Val] Loss: {avg_loss:.4f}, F1: {f1:.4f}")
        return avg_loss, f1

    def fit(self, epochs=-1):
        """
        - Training loop that alternates between train_epoch and val_test
        - Implements early stopping based on validation loss
        - Either stops after 'epochs' or if patience is exceeded
        - Returns (train_losses, val_losses)
        """
        assert self._early_stopping_patience > 0 or epochs > 0

        train_losses = []
        val_losses = []
        moving_avg_window = 3 
        best_moving_avg = 0.0
        recent_f1_scores = []
        best_val_loss = float('inf')
        best_val_loss = float('inf')
        best_raw_f1 = 0.0
        best_f1 = 0.0  # Track the best F1
        patience_counter = 0
        current_epoch = 0

        while True:
            # 1) Stop if we have reached the specified number of epochs
            if current_epoch == epochs:
                print("Reached maximum number of epochs.")
                break

            # 2) Training for one epoch
            train_loss = self.train_epoch()
            train_losses.append(train_loss)

            # 3) Validation step
            val_loss, val_f1 = self.val_test(current_epoch)
            val_losses.append(val_loss)
            self.scheduler.step(val_loss) 

            if val_f1 > best_raw_f1:
                best_raw_f1 = val_f1
                self.save_checkpoint(current_epoch)
                print(f"Epoch {current_epoch:02d}: New best raw F1: {best_raw_f1:.4f}")

                    # Update our list of recent F1 scores
            recent_f1_scores.append(val_f1)
            if len(recent_f1_scores) > moving_avg_window:
                # Keep only the most recent values
                recent_f1_scores.pop(0)

            '''# 4) Check if we have a new best validation loss
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save the best model so far
                self.save_checkpoint(current_epoch)
            else:
                patience_counter += 1
                # Track best by F1
            if val_f1 > best_f1:
                best_f1 = val_f1
                patience_counter = 0
            # Save the best model so far (based on F1)
                self.save_checkpoint(current_epoch)
            else:
                patience_counter += 1

            # 5) Early stopping criterion
            if patience_counter >= self._early_stopping_patience:
                print(f"Early stopping triggered at epoch {current_epoch}.")
                break'''
            
            if len(recent_f1_scores) == moving_avg_window:
                moving_avg = sum(recent_f1_scores) / moving_avg_window
                print(f"Epoch {current_epoch:02d} - Moving Average F1: {moving_avg:.4f} (Best: {best_moving_avg:.4f})")
                if moving_avg > best_moving_avg:
                    best_moving_avg = moving_avg
                    patience_counter = 0
                    # Save the best model so far
                    self.save_checkpoint(current_epoch)
                else:
                    patience_counter += 1
            else:
            # If not enough epochs for a full window, skip early stopping check
                print(f"Epoch {current_epoch:02d} - Not enough data for moving average, skipping early stopping check.")
                self.save_checkpoint(current_epoch)

        # Check if patience has been exceeded
            if patience_counter >= self._early_stopping_patience:
                print(f"Early stopping triggered at epoch {current_epoch}.")
                break

            current_epoch += 1

        return train_losses, val_losses
