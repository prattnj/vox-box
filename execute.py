import torch
import os
from IPython.display import Audio
from scipy.io.wavfile import write
from model import Tacotron2
from settings import hyperparameters
import numpy as np
from util import text_to_sequence

### SETTINGS ###

tacotron_sd_filepath = 'saved-models/tacotron2_noah.pt' # 'None' for default pretrained Tacotron 2
waveglow_sd_filepath = None # 'None' for default pretrained Waveglow
sample_rate = 22050 # Model was likely trained on 22050 Hz
result_dir = 'results'

################

def get_tacotron_model(sd_filepath=None):
  model = None
  if sd_filepath is None:
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
  else:
    model = Tacotron2(hyperparameters)
    model.load_state_dict(torch.load(sd_filepath)['state_dict'])
  model.eval()
  return model.cuda()
  
def get_waveglow_model(sd_filepath=None):
  model = None
  if sd_filepath is None:
    model = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
  else:
    model = torch.load(sd_filepath).cuda()
  model.remove_weightnorm(model)
  model.eval()
  return model.cuda()

tacotron = get_tacotron_model(tacotron_sd_filepath)
waveglow = get_waveglow_model(waveglow_sd_filepath)

# Gather user requests
os.system('clear')
input_text = input("What would you like the AI to say? ")
output_filepath = input("Enter a file name (.wav) to store the output in: ")
if output_filepath[-4:] == '.wav':
  output_filepath = output_filepath[:-4]
output_filepath += '.wav'
print(f'Output will be stored in {result_dir}/{output_filepath}.')
print('Generating...')

sequence = np.array(text_to_sequence(input_text, ['english_cleaners']))[None, :]
sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

mel_outputs, mel_outputs_postnet, _, alignments = tacotron.inference(sequence)

with torch.no_grad():
  audio = waveglow.infer(mel_outputs_postnet)
audio_numpy = audio[0].data.cpu().numpy()

if not os.path.exists(result_dir):
  os.mkdir(result_dir)

write(os.path.join(result_dir, output_filepath), sample_rate, audio_numpy)
print('Done.')