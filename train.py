import torch
from model import Tacotron2
from settings import hparams
from torch.utils.data import DataLoader
from dataset import VocalData, CollateData
from loss import Tacotron2Loss

def load_dataloaders(hparams):
  train_dataset = VocalData('tts-dataset/audio-dataset/train/metadata.csv', hparams)
  val_dataset = VocalData('tts-dataset/audio-dataset/val/metadata.csv', hparams)
  collate_fn = CollateData(hparams['n_frames_per_step'])

  train_loader = DataLoader(train_dataset, num_workers=1, shuffle=True, batch_size=hparams['batch_size'],
                            pin_memory=False, drop_last=True, collate_fn=collate_fn)
  return train_loader, val_dataset, collate_fn

def load_model(sd_file=None):
  model = Tacotron2(hparams)
  if sd_file is not None:
    model.load_state_dict(torch.load(sd_file)['state_dict'])
  return model.cuda()

if __name__ == '__main__':

  print('Loading datasets and loaders...')
  train_loader, valset, collate_fn = load_dataloaders(hparams)

  print('Loading model and optimizer...')
  model = load_model(hparams['starting_point'])
  model.train()

  optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'], weight_decay=hparams['weight_decay'])
  criterion = Tacotron2Loss()

  # Main loop
  print('Beginning main training loop...')
  for _ in range(hparams['n_epochs']):
    for i, batch in enumerate(train_loader):

      model.zero_grad()
      x, y = model.parse_batch(batch)
      y_pred = model(x)

      loss = criterion(y_pred, y)
      reduced_loss = loss.item()
      loss.backward()

      break

