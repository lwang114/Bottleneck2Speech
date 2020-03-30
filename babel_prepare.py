import os
import shutil
from scipy.io import wavfile 
from scipy.signal import decimate
from nltk.corpus import cmudict
import numpy as np
import librosa

UNK = ['(())']
DEBUG = True

class BabelKaldiPreparer:
  def __init__(self, data_root, sph2pipe, configs):
    self.audio_type = configs.get('audio_type', 'scripted')
    self.is_segment = configs.get('is_segment', False)
    self.fs = 8000
    self.data_root = data_root
    self.sph2pipe = sph2pipe

    if self.audio_type == 'conversational':
      self.transcripts = {'train': os.listdir(data_root+'conversational/training/transcript_roman/'),
                          'test': os.listdir(data_root+'conversational/eval/transcript_roman/'),
                          'dev':  os.listdir(data_root+'conversational/dev/transcript_roman/')} 
    elif self.audio_type == 'scripted':
      self.transcripts = {'train': os.listdir(data_root+'scripted/training/transcript_roman/'),
                        'test': os.listdir(data_root+'scripted/training/transcript_roman/'),
                        'dev':  os.listdir(data_root+'scripted/training/transcript_roman/')} 
    else:
      raise NotImplementedError

  def prepare_tts(self):
    if not os.path.isdir('data'):
      os.mkdir('data')
      os.mkdir('data/train')
      os.mkdir('data/test')
      os.mkdir('data/dev')
   
    '''
    sph_dir = {
      'train': self.data_root + 'conversational/training/',
      'test': self.data_root + 'conversational/eval/',
      'dev':  self.data_root + 'conversational/dev/'
    }
    '''
    
    sph_dir = {
      'train': self.data_root + 'scripted/training/',
      'test': self.data_root + 'scripted/training/',
      'dev':  self.data_root + 'scripted/training/'
    }
      
    for x in ['train', 'dev', 'test']:
      with open(os.path.join('data', x, 'text'), 'w') as text_f, \
           open(os.path.join('data', x, 'wav.scp'), 'w') as wav_scp_f, \
           open(os.path.join('data', x, 'utt2spk'), 'w') as utt2spk_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()

        i = 0
        for transcript_fn in sorted(self.transcripts[x], key=lambda x:x.split('.')[0]):
          # XXX
          # if i > 1:
          #   continue
          i += 1
          utt_id = transcript_fn.split('.')[0]
          if self.is_segment:
            with open(sph_dir[x] + 'transcript_roman/' + transcript_fn, 'r') as transcript_f:
              lines = transcript_f.readlines()
              for start, segment, end in zip(lines[::2], lines[1::2], lines[2::2]):
                start_sec = float(start[1:-1])
                end_sec = float(end[1:-1])
                start = int(self.fs * start_sec)
                end = int(self.fs * end_sec)
                if start >= end:
                  print('Corrupted segment info')
                words = segment.strip().split(' ')
                words = [w for w in words in w not in UNK and (w[0] != '<' or w[-1] != '>')]
                if len(words) == 0:
                  print('Empty segment')
                  continue
                text_f.write(utt_id + ' ' + ' '.join(sent) + '\n')
                wav_scp_f.write(utt_id + ' ' + self.sph2pipe + ' -f wav -p -c 1 ' + \
                os.path.join(sph_dir[x], 'audio/', utt_id + '.sph') + ' |[%d,%d]\n' % (start, end))
                utt2spk_f.write(utt_id + ' ' + '001\n') # XXX dummy speaker id
          else:
            print(x, utt_id)
            sent = []
            with open(sph_dir[x] + 'transcript_roman/' + transcript_fn, 'r') as transcript_f:
              lines = transcript_f.readlines()
              for line in lines[1::2]:
                words = line.strip().split(' ')             
                words = [w for w in words if w not in UNK and (w[0] != '<' or w[-1] != '>')]
                sent += words
              if len(words) == 0:
                print('Empty transcript file')
                continue

            text_f.write(utt_id + ' ' + ' '.join(sent) + '\n')
            wav_scp_f.write(utt_id + ' ' + self.sph2pipe + ' -f wav -p -c 1 ' + \
                os.path.join(sph_dir[x], 'audio/', utt_id + '.sph') + ' |\n')
            utt2spk_f.write(utt_id + ' ' + '001\n') # XXX dummy speaker id
      
if __name__ == '__main__':
  data_root = '/home/lwang114/data/babel/BABEL_OP1_102/'
  sph2pipe = '/home/lwang114/kaldi/tools/sph2pipe_v2.5/sph2pipe'
  configs = {'audio_type': 'conversational', 'is_segment': True}
  kaldi_prep = BabelKaldiPreparer(data_root, sph2pipe, configs)
  kaldi_prep.prepare_tts()
