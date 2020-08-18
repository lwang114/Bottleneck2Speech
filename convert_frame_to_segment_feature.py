import numpy as np
import json
import argparse
import os

def convert_frame_to_segment_feature(args):
  for feat_file in os.listdir(args.exp_dir):
    if feat_file.split('.')[-1] == 'npz' and feat_file.split('_')[1] != 'input' and feat_file.split('.')[0].split('_')[-1] != 'segmented': 
      print('feat_file: {}'.format(feat_file))
      ffeats_npz = np.load(args.exp_dir+feat_file) 
      if args.dataset_test == 'mscoco20k' or args.dataset_test == 'mscoco2k':
        feat_type = args.feat_type
        skip_ms = 10. # in ms
        audio_sequence_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/%s/%s_phone_info.json' % (args.dataset_test, args.dataset_test)
        with open(audio_sequence_file, 'r') as f:
          audio_seqs = json.load(f) 
      else:
        raise NotImplementedError('Invalid dataset for phone-level feature extraction')

      feat_ids = sorted(ffeats_npz, key=lambda x:int(x.split('_')[-1]))
      capt_ids = sorted(audio_seqs, key=lambda x:int(x.split('_')[-1])) 
      # print('len(feat_ids): ', len(feat_ids))

      pfeats = {}
      for feat_id in feat_ids:
        print(feat_id)
        ffeat = ffeats_npz[feat_id]
        print('ffeat.shape', ffeat.shape)
        if len(ffeat.shape) == 3:
          ffeat = ffeat.squeeze(0)
        elif len(ffeat.shape) == 4:
          ffeat = ffeat.squeeze(1)

        if args.audio_model == 'ae':
          ds_rate = 16
          ffeat = ffeat.T
        elif args.audio_model == 'transformer':
          ds_rate = 4
        audio_seq = audio_seqs[capt_ids[int(feat_id.split('_')[-1])]]
        sfeats = []
        start_phn = 0
        for word in audio_seq['data_ids']:
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
            
            if len(ffeat.shape) == 3:
              sfeat = ffeat[:, int(start_frame_local/ds_rate):int(end_frame_local/ds_rate)+1]
              if sfeat.shape[1] == 0:
                print('empty segment: ', phn[0], int(start_frame_local/ds_rate), int(end_frame_local/ds_rate))
                break 
            else:            
              sfeat = ffeat[int(start_frame_local/ds_rate):int(end_frame_local/ds_rate)+1] 
              if sfeat.shape[0] == 0:
                print('empty segment: ', phn[0], int(start_frame_local/ds_rate), int(end_frame_local/ds_rate))
                break
                      
            start_phn += end_frame - start_frame + 1 

            if feat_type == 'mean':
              if len(sfeat.shape) == 3: 
                mean_feat = np.mean(sfeat, axis=1)
              else:
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
      np.savez('%s/%s_segmented.npz' % (args.exp_dir, feat_file.split('.')[0]), **pfeats)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--exp_dir', '-e', type=str, help='Experiment directory')
  parser.add_argument('--dataset_test', '-d', type=str, help='Dataset used')
  parser.add_argument('--audio_model', '-a', type=str, default='transformer', help='Audio model type')  
  parser.add_argument('--feat_type', '-f', type=str, default='mean', help='Acoustic feature type')
  args = parser.parse_args()
  convert_frame_to_segment_feature(args)
