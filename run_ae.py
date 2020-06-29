import argparse
import os
import sys
import time
import torch
import torchvision
from AudioModels import *
import torchvision.transforms as transforms
from traintest_ae import *
from audio_waveform_dataset import *
from mscoco_segmented_audio_caption_dataset import *
import json
import numpy as np
import random

random.seed(1)
np.random.seed(1)

parser = argparse.ArgumentParser(argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.1)
parser.add_argument('--lr_decay', type=int, default=10, help='Divide the learning rate by 10 every lr_decay epochs')
parser.add_argument('--dataset_train', default='TIMIT', choices=['TIMIT', 'mscoco_train'], help='Data set used for training the model')
parser.add_argument('--dataset_test', default='mscoco2k', choices=['mscoco2k', 'mscoco20k'], help='Data set used for training the model')
parser.add_argument('--n_epoch', type=int, default=20)
parser.add_argument('--audio_model', type=str, default='ae', choices=['ae'], help='Acoustic model architecture')
parser.add_argument('--optim', type=str, default='sgd',
        help="training optimizer", choices=["sgd", "adam"])
parser.add_argument('--pretrain_model_file', type=str, default=None, help='Pretrained parameters file (used only in feature extraction)')
parser.add_argument('--save_features', action='store_true', help='Save the hidden activations of the neural networks')
parser.add_argument('--exp_dir', type=str, default=None, help='Experimental directory')
parser.add_argument('--date', type=str, default='', help='Date of the experiment')
parser.add_argument('--feat_type', type=str, default='last', choices=['mean', 'last', 'resample', 'discrete'], help='Method to extract the hidden phoneme representation')
parser.add_argument('--eval_phone_recognition', action='store_true', help='Perform phone recognition during validation')
parser.add_argument('--downsample_rate', type=int, default=1, help='Downsampling rate of the hidden representation')
args = parser.parse_args()

if args.exp_dir is None:
  if len(args.date) > 0:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f_%s' % (args.audio_model, args.dataset_train, args.optim, args.lr, args.date)
  else:
    args.exp_dir = 'exp/%s_%s_%s_lr_%.5f' % (args.audio_model, args.dataset_train, args.optim, args.lr)

if not os.path.isdir('exp'):
  os.mkdir('exp')
if not os.path.isdir(args.exp_dir):
  os.mkdir(args.exp_dir)

# TODO
feat_configs = {}

tasks = [0]
#------------------#
# Network Training #
#------------------#
if 0 in tasks:
  if args.dataset_train == 'TIMIT':
    audio_root_path = '/home/lwang114/data/TIMIT/'
    audio_sequence_file_train = '../data/TIMIT/TIMIT_train_phone_sequence_pytorch.txt'
    audio_sequence_file_test = '../data/TIMIT/TIMIT_test_phone_sequence_pytorch.txt'
    args.class2id_file = '../data/TIMIT/TIMIT_train_phone2ids.json'
  elif args.dataset_train == 'mscoco_train':
    audio_root_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/train2014/wav/'
    audio_sequence_file_train = '../../data/mscoco_phone_sequence_train.txt' 
    audio_sequence_file_test = '../../data/mscoco_phone_sequence_test.txt' 
    sequence_info_file = '/home/lwang114/data/mscoco/audio/train2014/train_2014.sqlite3' # TODO Test on the validation set rather than on the random split
    args.class2id_file = '../../data/mscoco_phone2id.json'
  elif args.dataset_train == 'babel':
    audio_root_path = '/ws/ifp-53_1/hasegawa/data/babel/' 
    audio_sequence_file_train = '../../data/babel_phone_sequence_train.txt'
    audio_sequence_file_test = '../../data/babel_phone_sequence_test.txt'
  # TODO GlobalPhone

  if args.audio_model == 'ae': # XXX
    args.downsample_rate = 2

  trainset = AudioWaveformDataset(audio_root_path, audio_sequence_file_train, feat_configs=feat_configs)
  if args.eval_phone_recognition:
    testset = MSCOCOSegmentCaptionDataset(audio_root_path, audio_sequence_file_test, sequence_info_file, phone2idx_file=args.class2id_file, feat_configs=feat_configs) # TODO Need to make this more general 
  else:
    testset = AudioWaveformDataset(audio_root_path, audio_sequence_file_test, feat_configs=feat_configs) 
    
  train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False) 

  audio_model = ConvAutoEncoder()
  train(audio_model, train_loader, test_loader, args) 

#--------------------------------#
# Frame-Level Feature Extraction #
#--------------------------------#
if 1 in tasks:
  if args.pretrain_model_file is None: # XXX 
    args.pretrain_model_file = 'exp/ae_mscoco_train_sgd_lr_0.00100_june20/audio_model.0.pth' 

  if args.dataset_train == 'mscoco_train':
    audio_root_path = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/wav/' 
    # audio_sequence_file = '../data/mscoco/mscoco20k_phone_info.json'
    audio_sequence_file = '/ws/ifp-53_2/hasegawa/lwang114/data/mscoco/mscoco2k/mscoco2k_wav.scp'    
    feat_configs = {'dataset': 'mscoco2k'} # XXX
    testset = AudioWaveformDataset(audio_root_path, audio_sequence_file, feat_configs)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.batch_size, shuffle=False)
  else:
    raise NotImplementedError
      
  if args.audio_model == 'ae':
    audio_model = ConvAutoEncoder()
    audio_model.load_state_dict(torch.load(args.pretrain_model_file))
  else:
    raise NotImplementedError

  args.save_features = True
  args.eval_phone_recognition = False
  _ = validate(audio_model, test_loader, args)

#-----------------------------------------#
# Frame to Phone Level Feature Conversion #
#-----------------------------------------#
if 2 in tasks:
  ffeats_npz = np.load(args.exp_dir + '/embed1_all.npz') 
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
    audio_seq = audio_seqs[feat_id]
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
  np.savez('%s/phone_features_%s.npz' % (args.exp_dir, args.feat_type), **pfeats) 
