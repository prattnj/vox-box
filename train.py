import os
import time
import torch
from model import Tacotron2
from settings import hparams
from torch.utils.data import DataLoader
from dataset import VocalData, CollateData
from loss import Tacotron2Loss

def load_dataloaders(hparams):
  train_dataset = VocalData(os.path.join(hparams['dataset_dir'], 'train', 'metadata.csv'), hparams)
  val_dataset = VocalData(os.path.join(hparams['dataset_dir'], 'val', 'metadata.csv'), hparams)
  collate_fn = CollateData(hparams['n_frames_per_step'])

  train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True, batch_size=hparams['batch_size'],
                            pin_memory=False, drop_last=True, collate_fn=collate_fn)
  val_loader = DataLoader(val_dataset, num_workers=1, shuffle=False, batch_size=hparams['batch_size'],
                          pin_memory=False, drop_last=False, collate_fn=collate_fn)
  return train_loader, val_loader

def load_model(sd_file=None):
  model = Tacotron2(hparams)
  if sd_file is not None:
    model.load_state_dict(torch.load(sd_file)['state_dict'])
  return model.cuda()

def save_checkpoint(model, filepath):
  # Keep the {'state_dict': ...} format so checkpoints stay compatible with
  # NVIDIA's and with execute.py's loading code
  torch.save({'state_dict': model.state_dict()}, filepath)

def validate(model, val_loader, criterion):
  model.eval()
  with torch.no_grad():
    total_loss, n_batches = 0.0, 0
    for batch in val_loader:
      x, y = model.parse_batch(batch)
      y_pred = model(x)
      total_loss += criterion(y_pred, y).item()
      n_batches += 1
  model.train()
  return total_loss / max(n_batches, 1)

if __name__ == '__main__':

  torch.manual_seed(hparams['seed'])
  torch.cuda.manual_seed(hparams['seed'])
  os.makedirs(hparams['checkpoint_dir'], exist_ok=True)

  print('Loading datasets and loaders...')
  train_loader, val_loader = load_dataloaders(hparams)

  print('Loading model and optimizer...')
  model = load_model(hparams['starting_point'])
  model.train()

  optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
  criterion = Tacotron2Loss()

  best_val_loss = float('inf')
  step = 0

  # Main loop
  print('Beginning main training loop...')
  for epoch in range(hparams['n_epochs']):
    epoch_start = time.time()

    for i, batch in enumerate(train_loader):
      step_start = time.time()

      optimizer.zero_grad()
      x, y = model.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      loss.backward()

      # Clip gradients: attention RNNs are prone to exploding gradients,
      # especially when fine-tuning on a small dataset
      grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), hparams['grad_clip_thresh'])

      optimizer.step()
      step += 1

      print('Epoch {} step {} (global {}): loss {:.6f}, grad norm {:.4f}, {:.2f}s'.format(
        epoch, i, step, loss.item(), grad_norm, time.time() - step_start))

    # End of epoch: validate and checkpoint
    val_loss = validate(model, val_loader, criterion)
    print('Epoch {} done in {:.1f}s: validation loss {:.6f}'.format(
      epoch, time.time() - epoch_start, val_loss))

    save_checkpoint(model, os.path.join(hparams['checkpoint_dir'], 'checkpoint_last.pt'))
    if val_loss < best_val_loss:
      best_val_loss = val_loss
      save_checkpoint(model, os.path.join(hparams['checkpoint_dir'], 'checkpoint_best.pt'))
      print('New best validation loss; saved checkpoint_best.pt')

  print('Training complete. Best validation loss: {:.6f}'.format(best_val_loss))
