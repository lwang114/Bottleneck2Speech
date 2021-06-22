import json
import os

def prepare(data_root, exp_root):
  if not os.path.isdir(exp_root):
    os.makedirs(exp_root)
  
  for split in ['train', 'test', 'eval']:
    sub_dir = os.path.isdir(os.path.join(exp_root, f'{split}_flickr_audio'))
    if not os.path.isdir(sub_dir):
      os.makedirs(sub_dir)

    sent_json = json.load(os.path.join(data_root, f'{split}_flickr_audio.json'))
    with open(sent_json, 'r') as sent_f,\
         open(os.path.join(sub_dir, 'text'), 'w') as text_f,\
         open(os.path.join(sub_dir, 'wav.scp'), 'w') as wav_scp_f,\
         open(os.path.join(sub_dir, 'utt2spk'), 'w') as utt2spk_f:
      text_f.truncate()
      wav_scp_f.truncate()
      utt2spk_f.truncate()
      
      for line in sent_f:
        sent_info = json.loads(line.rstrip('\n'))
        utt_path = sent_info['utterance_id']
        spk = sent_info['speaker']
        utt_id = utt_path.split('/')[-1]
        text = ' '.join(sent_info['text'])
        text_f.write(f'{utt_id} {text}\n')
        wav_scp_f.write(f'{utt_id} {utt_path}.wav\n')
        utt2spk_f.write(f'{utt_id} {spk}\n')

if __name__ == '__main__':
  data_root = '/ws/ifp-53_2/hasegawa/lwang114/data/zerospeech2021_dataset/phonetic'
  exp_root = 'data/flickr'
  prepare(data_root, exp_root)
     
