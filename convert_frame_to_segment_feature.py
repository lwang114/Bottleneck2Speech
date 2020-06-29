import numpy as np
import json
import argparse

def convert_frame_to_segment_feature(args):
  ffeats_npz = np.load(args.exp_dir + 'embed1_all.npz') 
  if args.dataset_test == 'mscoco20k' or args.dataset_test == 'mscoco2k':
    feat_type = args.feat_type
    skip_ms = 10. # in ms
    audio_sequence_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/%s/%s_phone_info.json' % (args.dataset_test, args.dataset_test)
    with open(audio_sequence_file, 'r') as f:
      audio_seqs = json.load(f) 
  else:
    raise NotImplementedError('Invalid dataset for phone-level feature extraction')

  feat_ids = sorted(ffeats_npz, key=lambda x:int(x.split('_')[-1]))
  # print('len(feat_ids): ', len(feat_ids))

  pfeats = {}
  for feat_id in feat_ids:
    print(feat_id)
    ffeat = ffeats_npz[feat_id]
    if args.audio_model == 'ae':
      ds_rate = 16
      ffeat = ffeat.T
    elif args.audio_model == 'transformer':
      ds_rate = 4
    capt_id = feat_id.split('_')[0] + '_' + str(int(feat_id.split('_')[1])) # Remove the prepended zeros
    audio_seq = audio_seqs[capt_id]
    sfeats = []
    start_phn = 0
    for word in audio_seq['data_ids']:
      print(word)
      for phn in word[2]:
        # Convert each time step from ms to MFCC frames in the synthetic captions
        start_ms, end_ms = phn[1], phn[2]
        start_frame = int(start_ms / skip_ms)
        end_frame = int(end_ms / skip_ms)
        start_frame_local = start_phn
        end_frame_local = end_frame - start_frame + start_phn
        # print(start_frame, end_frame)
        if phn[0] == '#':
          continue
        if start_frame > end_frame:
          print('empty segment: ', phn[0], start_frame, end_frame)

        sfeat = ffeat[int(start_frame_local/ds_rate):int(end_frame_local/ds_rate)+1]
        start_phn += end_frame - start_frame + 1 

        if feat_type == 'mean':
          mean_feat = np.mean(sfeat, axis=0)
          # if feat_id == 'arr_0':
          #   print(phn[0], mean_feat[:10])
          #   print(sfeat[:10])
          #   print(ffeat.shape, start_frame_local, end_frame_local)
          sfeats.append(mean_feat)
        elif feat_type == 'last':
          sfeats.append(sfeat[-1])
        else:
          raise ValueError('Feature type not found')
      
      if feat_type == 'discrete':
        pfeats[feat_id] = sfeats
      else:
        pfeats[feat_id] = np.stack(sfeats, axis=0) 
  np.savez('%s/%s_%s.npz' % (args.exp_dir, args.dataset_test, args.audio_model), **pfeats)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, help='Experiment directory')
  parser.add_argument('--dataset_test', '-d', type=str, help='Dataset used')
  parser.add_argument('--audio_model', '-a', type=str, default='transformer', help='Audio model type')  
  parser.add_argument('--feat_type', '-f', type=str, default='mean', help='Acoustic feature type')
  args = parser.parse_args()
  convert_frame_to_segment_feature(args)
