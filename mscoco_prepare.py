import os
import numpy as np
import librosa
from scipy.io import wavfile
from copy import deepcopy

class MSCOCOKaldiPreparer:
  def __init__(self, data_root, exp_root, g2p_root, phonetisaurus_root):
    self.data_root = data_root
    self.exp_root = exp_root
    self.g2p_root = g2p_root
    self.phonetisaurus_root = phonetisaurus_root

  def prepare(self, json_file, split_file=None):
    if split_file:
      subdirs = [self.exp_root + 'mscoco/train', self.exp_root + 'mscoco/test']
      with open(split_file, 'r') as f:
        test_ids = [i for i, line in enumerate(f.readlines()) if int(line)] 
    else:
      subdirs = [self.exp_root + 'mscoco/train']
      test_ids = []

    if not os.path.isdir(self.exp_root + 'mscoco'):
      os.mkdir(self.exp_root + 'mscoco/')
      os.mkdir(self.exp_root + 'mscoco/wavs/')
      if self.split_file:
        os.mkdir(self.exp_root + 'mscoco/train')
        os.mkdir(self.exp_root + 'mscoco/test')
      else: 
        os.mkdir(self.exp_root + 'mscoco/train')

    for x in subdirs:
      with open(os.path.join(x, 'text'), 'w') as text_f,\
           open(os.path.join(x, 'wav.scp'), 'w') as wav_scp_f,\
           open(os.path.join(x, 'utt2spk'), 'w') as utt2spk_f,\
           open(os.path.join(x, 'segments'), 'w') as segment_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        segment_f.truncate()

      for i, pair in enumerate(pair_info): 
        # TODO Extract the phone caption in IPA symbols
        caption = []
        audio_fns = []
        segments = []
         
        dur = 0
        for c, data_id in zip(pair['concepts'], pair['data_ids']):
          # Convert caption into phones in IPA
          phones = os.popen('%sphonetisaurus-g2pfst --model=%senglish_4_2_2.fst --word=%s' % (self.phonetisaurus_root, self.g2p_root, c)).read().strip().split()[2:]
          caption.append(phones)
          audio_fn = '_'.join(data_id[1].split('_')[:-1]) 
          begin, end = data_id[2], data_id[3]
          dur += end - begin 
          audio_fns.append(audio_fn)
          fs, y = wavfile.read(self.data_root + audio_fn + '.wav')
          seg = y[begin:end]
          segments.append(seg)
        
        utt_id = 'arr_%06d' % i  
        caption = ' '.join(caption)
        y_concat = np.concatenate(segments) 
        audio_fn_concat = '_'.join(audio_fns)
        wavfile.write(os.path.join(self.exp_root + 'mscoco/wavs', audio_fn_concat + '.wav'), fs, y_concat)
        if i in test_ids and x == 'test':
          text_f.write('%s %s\n' % (utt_id, caption))
          segment_f.write('%s %.4f %.4f\n' % (utt_id, 0, dur))
          wav_scp_f.write('%s %s\n' % (utt_id, os.path.join('mscoco/wavs', audio_fn_concat + '.wav'))
          utt2spk_f.write('%s %s\n' % (utt_id, utt_id))
        elif x == 'train':
          text_f.write('%s %s\n' % (utt_id, caption))
          segment_f.write('%s %.4f %.4f\n' % (utt_id, 0, dur))
          wav_scp_f.write('%s %s\n' % (utt_id, os.path.join('mscoco/wavs', audio_fn_concat + '.wav'))
          utt2spk_f.write('%s %s\n' % (utt_id, utt_id))

if __name__ == '__main__':
  data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/val2014/wav/' 
  exp_root = './'
  g2p_root = '/ws/ifp-53_1/hasegawa/tools/espnet/egs/discophone/ifp_lwang114/g2ps/models/'
  phonetisaurus_root = '/ws/ifp-53_1/hasegawa/tools/kaldi/tools/phonetisaurus-g2p/' 
  preproc = MSCOCOKaldiPreparer(data_root, exp_root, g2p_root, phonetisaurus_root)         
