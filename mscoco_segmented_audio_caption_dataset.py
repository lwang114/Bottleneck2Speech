import torch
import librosa
import scipy.io as io
from torch.utils.data import Dataset, DataLoader
import json
import numpy as np
from speechcoco.speechcoco import SpeechCoco

EPS = 1e-50
# This function is from DAVEnet (https://github.com/dharwath/DAVEnet-pytorch)
def preemphasis(signal,coeff=0.97):
    """perform preemphasis on the input signal.
    
    :param signal: The signal to filter.
    :param coeff: The preemphasis coefficient. 0 is none, default 0.97.
    :returns: the filtered signal.
    """    
    return np.append(signal[0],signal[1:]-coeff*signal[:-1])

class MSCOCOSegmentCaptionDataset(Dataset):
  def __init__(self, audio_root_path, audio_sequence_file, sequence_info_file, phone2idx_file, feat_configs=None):
    # Inputs:
    # ------
    #     audio_root_path: str containing the root to the audio files
    #     audio_sequence_file: name of the meta info file. Each row contains the string of audio filename and phone sequence of the audio
    #     phone2idx_file: name of the json file containing the mapping between each phone symbol and an integer id
    self.audio_root_path = audio_root_path
    self.max_nframes = feat_configs.get('max_num_frames', 1024)
    self.max_nphones = feat_configs.get('max_num_phones', 100)
    self.sr = feat_configs.get('sample_rate', 16000)
    self.n_mfcc = feat_configs.get('n_mfcc', 40)
    self.skip_ms = feat_configs.get('skip_size', 10)
    self.window_ms = feat_configs.get('window_len', 25)
    self.dct_type = feat_configs.get('dct_type', 3)
    self.coeff = feat_configs.get('coeff', 0.97)

    self.audio_filenames = []
    with open(audio_sequence_file, 'r') as f:
      i = 0
      for line in f:
        # XXX
        # if i > 15:
        #   break
        # i += 1
        audio_info = line.strip().split()
        self.audio_filenames.append(audio_info[0])
    self.speech_api = SpeechCoco(sequence_info_file)

    with open(phone2idx_file, 'r') as f:
      self.phone2idx = json.load(f)

  def __len__(self):
    return len(self.audio_filenames)

  def get_label_segmentation(self, audio_filename):
    img_id = audio_filename.split('_')[0]
    caption_info = self.speech_api.getImgCaptions(int(img_id))
    
    capt = None
    found = 0
    for capt in caption_info:
      if capt.filename == audio_filename:
        found = 1
        break
    if not found:
      print('Warning: caption name not found')

    segmentation = np.zeros((self.max_nframes+1,), dtype=int)
    segmentation[0] = 1
    labels = np.nan * np.ones((self.max_nphones,), dtype=int)
    timecode = capt.timecode.parse()
    nphones = 0
    for tc_word in timecode:
      for tc_syl in tc_word['syllable']:
        for tc_phn in tc_syl['phoneme']: 
          if tc_phn['value'][0] == '_':
            continue
          begin_ms, end_ms = tc_phn['begin'], tc_phn['end']
          begin, end = int(begin_ms / self.skip_ms), int(end_ms / self.skip_ms)  
          if end > self.max_nframes or nphones + 1 > self.max_nphones:
            break
          labels[nphones] = self.phone2idx[tc_phn['value']]
          segmentation[end] = 1
          nphones += 1
    return segmentation, np.asarray(labels)

  def __getitem__(self, idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    audio_filename = self.audio_filenames[idx]
    segmentation, labels = self.get_label_segmentation(audio_filename)
    sr, y = io.wavfile.read(self.audio_root_path + audio_filename)
    y = y.astype(float)
    n_fft = int(self.window_ms * sr / 1000)
    hop_length = int(self.skip_ms * sr / 1000)
    mfcc = librosa.feature.mfcc(y, sr=sr, n_mfcc=self.n_mfcc, dct_type=self.dct_type, n_fft=n_fft, hop_length=hop_length)
    mfcc -= np.mean(mfcc)
    mfcc /= max(np.sqrt(np.var(mfcc)), EPS)
    nframes = min(mfcc.shape[1], self.max_nframes)
    mfcc = self.convert_to_fixed_length(mfcc)
    return torch.FloatTensor(mfcc), torch.IntTensor(labels), torch.IntTensor(segmentation)
  
  def convert_to_fixed_length(self, mfcc):
    T = mfcc.shape[1] 
    pad = abs(self.max_nframes - T)
    if T < self.max_nframes:
      mfcc = np.pad(mfcc, ((0, 0), (0, pad)), 'constant', constant_values=(0))
    elif T > self.max_nframes:
      mfcc = mfcc[:, :-pad]
    return mfcc   
