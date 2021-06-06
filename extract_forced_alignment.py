import os
import json

def extract_forced_alignment(exp_dir):
  """
  Extract forced alignment from Kaldi based on the tutorial 
  https://www.eleanorchodroff.com/tutorial/kaldi/forced-alignment.html
  
  Args:
    exp_dir: str, path to the forced alignment directory, containing the 
      following files
        data/lang/phone.txt : a file with each line 
          {utterance id}\t{phone in BIES fmt}\t{int phone id}
        data/lang/phone/align_lexicon.txt : a file with each line
          {utterance id}\t{word}\t{phones in BIES fmt} 
    
    out_prefix: str, file storing the forced alignment in the format
      { "utterance_id" : str,
        "words" : a list of dicts with keys 
          "begin" : float,
          "end" : float,
          "text" : str,
          "phonemes" : a list of dicts with keys
            "begin" : float,
            "end" : float,
            "text" : str
      } 
  """
  # Convert time marks and phone ids
  id_to_phone = dict()
  with open('data/lang/phones.txt', 'r') as phn_f,\
       open('merged_alignment.txt', 'r') as merged_f,\
       open('final_ali.txt', 'w') as final_f:
    for line in phn_f:
      phn, idx = line.rstrip('\n').split('\t')
      id_to_phone[idx] = phn

    cur_token = None  
    for line in merged_f:
      utt_id, channel, start, dur, phn_id = line.rstrip('\n').split('\t')
      phn = id_to_phone[phn_id]
      final_f.write(f'{utt_id}\t{channel}\t{start}\t{dur}\t{phn}\n')

  # Convert phone alignment to word alignment
  with open('data/lang/phones/ali_lexicon.txt', 'r') as pron_f,\
       open('final_ali.txt', 'r') as final_f,\
       open('utterance_info.json', 'w') as word_f:
    pron_to_word = dict()
    for line in pron_f:
      parts = line.split('\t')
      pron = tuple(phn.split('_')[0] for phn in parts[1:])
      pron_to_word[pron] = parts[0]

    cur_utt_id = ''
    cur_utt = None
    cur_word = {'begin': None,
                'end': None,
                'phonemes': [],
               }
    start_word = 0
    for line in final_f:
      utt_id, _, begin, dur, phn = line.rstrip('\n').split('\t')
      if utt_id != cur_utt_id:
        if cur_utt_id:
          word_f.write(json.dumps(cur_utt)+'\n')
        cur_utt_id = utt_id
        cur_utt = dict()
        cur_utt['utterance_id'] = utt_id
        cur_utt['words'] = []  

      token, boundary = phn.split('_')
      cur_word['phonemes'].append({'begin': float(begin),
                                   'end': float(begin)+float(dur),
                                   'text': token})
      if boundary in ['E', 'S']:
        pron = tuple(phn['text'] for phn in cur_word['phonemes'])
        cur_word['text'] = pron_to_word[pron] 
        cur_utt['words'].append(cur_word)
        cur_word = dict()
    word_f.write(json.dumps(cur_utt)+'\n')

