import os
import numpy as np
import librosa
from scipy.io import wavfile
from copy import deepcopy
import json
import shutil

def prepare_speechcoco(info_file, data_root, exp_root, split_file=None):
    splits = None
    if split_file:
      with open(split_file, 'r') as f:
        splits = f.read().strip().split('\n') 
    
    cur_record_id = ''
    i_record = 0
    texts = []
    utt_ids = []
    with open(info_file, 'r') as in_f,\
         open(os.path.join(exp_root, 'wav.scp'), 'w') as wav_scp_f,\
         open(os.path.join(exp_root, 'text'), 'w') as text_f,\
         open(os.path.join(exp_root, 'utt2spk'), 'w') as utt2spk_f:
        text_f.truncate()
        wav_scp_f.truncate()
        utt2spk_f.truncate()
        for line in in_f:
          parts = line.strip().split()
          record_id = parts[0]
          word = parts[1]
          if record_id != cur_record_id:
            utt_id = 'arr_{:06d}'.format(len(texts))
            if splits is not None:
              if splits[len(texts)] == '1':
                wav_scp_f.write('{} {}/{}.wav\n'.format(utt_id, data_root, record_id))
                utt2spk_f.write('{} {}\n'.format(utt_id, utt_id))
            else:
              wav_scp_f.write('{} {}/{}.wav\n'.format(utt_id, data_root, record_id))
              utt2spk_f.write('{} {}\n'.format(utt_id, utt_id))

            texts.append([word])
            utt_ids.append(utt_id)
            cur_record_id = record_id
            print('Recording {}'.format(len(texts)))
          else:
            texts[-1].append(word)

        for utt_id, text in zip(utt_ids, texts):
          if splits is not None:
            if splits[int(utt_id.split('_')[-1])] == '1': 
              text_f.write('{} {}\n'.format(utt_id, ' '.join(text)))
          else:
            text_f.write('{} {}\n'.format(utt_id, ' '.join(text)))

if __name__ == '__main__':
  root = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/'
  info_file_train = '{}/train2014/mscoco_train_word_segments.txt'.format(root)
  info_file_dev = '{}/val2014/mscoco_val_word_segments.txt'.format(root)
  split_file = '{}/val2014/mscoco_val_split.txt'.format(root)

  exp_root = '/ws/ifp-53_2/hasegawa/lwang114/fall2020/exp/mscoco_kaldi_files_10_4_2020/'
  if not os.path.isdir(exp_root):
    os.mkdir(exp_root)
    os.mkdir('{}/train/'.format(exp_root))
    os.mkdir('{}/dev/'.format(exp_root))
    os.mkdir('{}/eval/'.format(exp_root))

  prepare_speechcoco(info_file_train, '{}/train2014'.format(root), '{}/train'.format(exp_root))
  prepare_speechcoco(info_file_dev, '{}/val2014'.format(root), '{}/dev'.format(exp_root), split_file=split_file)
  shutil.copyfile('{}/dev/wav.scp'.format(exp_root), '{}/eval/wav.scp'.format(exp_root))
  shutil.copyfile('{}/dev/text'.format(exp_root), '{}/eval/text'.format(exp_root))
  shutil.copyfile('{}/dev/utt2spk'.format(exp_root), '{}/eval/utt2spk'.format(exp_root))
