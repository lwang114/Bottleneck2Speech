import os
import numpy as np
import librosa
from scipy.io import wavfile
from copy import deepcopy
import json

class MSCOCOKaldiPreparer:
  def __init__(self, json_file, g2p_root, split_file=None):
    self.g2p_root = g2p_root
    self.split_file = split_file
    with open(json_file, 'r') as f:
      self.pair_info = json.load(f)

  def generate_synthetic_wavfiles(self, raw_data_root, data_root):
    self.raw_data_root = raw_data_root
    self.data_root = data_root
    if not os.path.isdir(self.data_root):
      os.mkdir(self.data_root)
    if not os.path.isdir(self.data_root + '/wav'):
      os.mkdir(self.data_root + '/wav')
          
    for i, pair_id in enumerate(sorted(self.pair_info, key=lambda x:int(x.split('_')[-1]))):
      # XXX
      # if i > 2:
      #   break
      pair = self.pair_info[pair_id]  
      caption = []
      audio_fns = []
      segments = []
      dur = 0
      for c, data_id in zip(pair['concepts'], pair['data_ids']):
        audio_fn = '_'.join(data_id[1].split('_')[:-1]) 
        begin_ms, end_ms = data_id[2], data_id[3]
        dur += end_ms - begin_ms 
        audio_fns.append(audio_fn)
        
        fs, y = wavfile.read(self.raw_data_root + audio_fn + '.wav')
        print(y.shape)
        begin, end = int(begin_ms * fs / 1000), int(end_ms * fs / 1000)
        seg = y[begin:end]
        segments.append(seg)

      y_concat = np.concatenate(segments) 
      audio_fn_concat = '_'.join(audio_fns)
      wavfile.write(os.path.join(data_root, 'wav', audio_fn_concat + '.wav'), fs, y_concat)

  def prepare(self, data_root, exp_root):
    self.data_root = data_root
    self.exp_root = exp_root
    if self.split_file:
      subdirs = [self.exp_root + 'mscoco/train', self.exp_root + 'mscoco/test']
      with open(self.split_file, 'r') as f:
        test_ids = [i for i, line in enumerate(f.readlines()) if int(line)] 
    else:
      if self.split_file:
        subdirs = [self.exp_root + 'mscoco/train', self.exp_root + 'mscoco/dev', self.exp_root + 'mscoco/eval']
      else:
        subdirs = [self.exp_root + 'mscoco/train']
      test_ids = []

    if not os.path.isdir(self.exp_root + 'mscoco'):
      os.mkdir(self.exp_root + 'mscoco/')
      os.mkdir(self.exp_root + 'mscoco/wav/')
      os.mkdir(self.exp_root + 'mscoco/train')
      os.mkdir(self.exp_root + 'mscoco/dev')
      os.mkdir(self.exp_root + 'mscoco/eval')

    for x in subdirs:
      with open(os.path.join(x, 'text'), 'w') as text_f,\
           open(os.path.join(x, 'wav.scp'), 'w') as wav_scp_f,\
           open(os.path.join(x, 'utt2spk'), 'w') as utt2spk_f:
           # open(os.path.join(x, 'segments'), 'w') as segment_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        # segment_f.truncate()

        for i, pair_id in enumerate(sorted(self.pair_info, key=lambda x:int(x.split('_')[-1]))):
          # XXX
          # if i > 2:
          #   break
          pair = self.pair_info[pair_id]  
          caption = []
          audio_fns = []
          segments = []
          for data_id in pair['data_ids']:
            # Convert caption into phones in IPA 
            audio_fn = '_'.join(data_id[1].split('_')[:-1]) 
            audio_fns.append(audio_fn)
          
          utt_id = 'arr_%06d' % i  
          caption = ' '.join(pair['concepts']) 
          audio_fn_concat = '_'.join(audio_fns)
                    
          if i in test_ids and x.split('/')[-1] == 'test':
            text_f.write('%s %s\n' % (utt_id, caption))
            wav_scp_f.write('%s %s\n' % (utt_id, os.path.join(data_root, 'wav', audio_fn_concat + '.wav')))
            utt2spk_f.write('%s %s\n' % (utt_id, utt_id))
          elif x.split('/')[-1] == 'train':
            text_f.write('%s %s\n' % (utt_id, caption))
            wav_scp_f.write('%s %s\n' % (utt_id, os.path.join(data_root, 'wav', audio_fn_concat + '.wav')))
            utt2spk_f.write('%s %s\n' % (utt_id, utt_id))

if __name__ == '__main__':
  json_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced/mscoco_subset_1300k_concept_info_power_law_1.json' 
  raw_data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014/wav/' 
  data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco_synthetic_imbalanced' # '/home/lwang114/spring2020/SeeSegmentAlign/acoustic_embedders/Bottleneck2Speech/data/mscoco'
  exp_root = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/data/' # 'data/'
  g2p_root = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/g2ps/models/'
  preproc = MSCOCOKaldiPreparer(json_file, g2p_root)
  # preproc.generate_synthetic_wavfiles(raw_data_root, data_root)
  preproc.prepare(data_root, exp_root) 
