from util import symbols

hparams = {
  
  # Model
  'attention_dim': 128,
  'attention_location_kernel_size': 31,
  'attention_location_n_filters': 32,
  'attention_rnn_dim': 1024,
  'decoder_rnn_dim': 1024,
  'encoder_embed_dim': 512,
  'encoder_kernel_size': 5,
  'encoder_n_conv': 3,
  'fp16_run': 0,
  'gate_threshold': 0.5,
  'mask_padding': True,
  'max_decoder_steps': 1000,
  'n_frames_per_step': 1,
  'n_mel_channels': 80,
  'n_symbols': len(symbols),
  'p_attention_dropout': 0.1,
  'p_decoder_dropout': 0.1,
  'postnet_embed_dim': 512,
  'postnet_kernel_size': 5,
  'postnet_n_conv': 5,
  'prenet_dim': 256,
  'symbols_embedding_dim': 512,

  # Dataset
  'text_cleaners': ['english_cleaners'],
  'max_wav_value': 32768.0,
  'sampling_rate': 22050,
  'filter_length': 1024,
  'hop_length': 256,
  'win_length': 1024,
  'mel_fmin': 0.0,
  'mel_fmax': 8000.0,

  # Train
  'batch_size': 12,
  'learning_rate': 1e-3,
  'weight_decay': 1e-6,
  'n_epochs': 1,
  'starting_point': 'saved-models/tacotron2_nvidia.pt', # None for a fresh model
  'dataset_dir': 'tts-dataset/audio-dataset',
}

# hello Molly