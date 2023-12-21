import torch
import os
from IPython.display import Audio
from scipy.io.wavfile import write
from model import Tacotron2
from settings import hparams
import numpy as np
from util import text_to_sequence

##### SETTINGS #####

# 'saved-models/tacotron2_nvidia.pt'
tacotron_sd_filepath = 'saved-models/test50.pt' # 'None' for default pretrained Tacotron 2
sample_rate = 22050 # Model was likely trained on 22050 Hz
result_dir = 'results'

####################

def get_waveglow():
  waveglow = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_waveglow', model_math='fp16')
  waveglow.remove_weightnorm(waveglow)
  waveglow.cuda().eval()
  return waveglow

def generate_with_local_model(text, sd_file):

  # Load models
  tacotron = Tacotron2(hparams)
  tacotron.load_state_dict(torch.load(sd_file)['state_dict'])
  tacotron.cuda().eval()  
  waveglow = get_waveglow()

  # Prepare input
  sequence = np.array(text_to_sequence(text, ['english_cleaners']))[None, :]
  sequence = torch.autograd.Variable(torch.from_numpy(sequence)).cuda().long()

  # Run chained models
  _, mel_outputs_postnet, _, _ = tacotron.inference(sequence)
  with torch.no_grad():
    audio = waveglow.infer(mel_outputs_postnet)
  audio_numpy = audio[0].data.cpu().numpy()

  return audio_numpy

def generate_with_online_model(text):

  # Load models
  tacotron = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tacotron2', model_math='fp16')
  tacotron.cuda().eval()
  waveglow = get_waveglow()

  # Load utilities
  utils = torch.hub.load('NVIDIA/DeepLearningExamples:torchhub', 'nvidia_tts_utils')
  sequences, lengths = utils.prepare_input_sequence([text])

  # Run chained models
  with torch.no_grad():
    mel, _, _ = tacotron.infer(sequences, lengths)
    audio = waveglow.infer(mel)
  audio_numpy = audio[0].data.cpu().numpy()
  
  return audio_numpy

if __name__ == '__main__':

  # Gather user requests
  input_text = input("What would you like the AI to say? ")
  output_filepath = input("Enter a file name (.wav) to store the output in: ")
  if output_filepath[-4:] == '.wav':
    output_filepath = output_filepath[:-4]
  output_filepath += '.wav'
  print(f'Output will be stored in {result_dir}/{output_filepath}.')
  print('Generating...')

  if tacotron_sd_filepath is None:
    audio_numpy = generate_with_online_model(input_text)
  else:
    audio_numpy = generate_with_local_model(input_text, tacotron_sd_filepath)

  if not os.path.exists(result_dir):
    os.mkdir(result_dir)

  write(os.path.join(result_dir, output_filepath), sample_rate, audio_numpy)
  print('Done.')