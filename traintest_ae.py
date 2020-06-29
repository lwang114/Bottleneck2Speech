import time
import shutil
import torch
import torch.nn as nn
import numpy as np
import sys
import json
import os
from torch.autograd import Variable
from sklearn.svm import LinearSVC

def train(audio_model, train_loader, test_loader, args, device_id=0): 
  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  # Set up the optimizer
  # XXX
  '''  
  for p in audio_model.parameters():
    if p.requires_grad:
      print(p.size())
  '''
  trainables = [p for p in audio_model.parameters() if p.requires_grad]
  
  exp_dir = args.exp_dir 
  if args.optim == 'sgd':
    optimizer = torch.optim.SGD(trainables, args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay)
  elif args.optim == 'adam':
    optimizer = torch.optim.Adam(trainables, args.lr,
                        weight_decay=args.weight_decay)
  else:
    raise ValueError('Optimizer %s is not supported' % args.optim)

  audio_model.train()

  running_loss = 0.
  best_val_loss = 0.
  best_acc = 0.
  criterion = nn.MSELoss()
  for epoch in range(args.n_epoch):
    running_loss = 0.
    # XXX
    # adjust_learning_rate(args.lr, args.lr_decay, optimizer, epoch)
    begin_time = time.time()
    audio_model.train()
    for i, audio_input in enumerate(train_loader):
      # XXX
      #if i > 3:
      #  break

      inputs, nframes = audio_input 
      inputs = Variable(inputs)
      nframes = nframes.type(dtype=torch.int)
      
      if torch.cuda.is_available():
        inputs = inputs.cuda()
      
      optimizer.zero_grad()
      outputs = audio_model(inputs)
      # print(nphones.data.numpy())

      # MSE loss
      loss = criterion(outputs, inputs) 
      #running_loss += loss.data.cpu().numpy()[0]
      running_loss += loss.data.cpu().numpy()
      loss.backward()
      optimizer.step()
      
      # TODO: Adapt to the size of the dataset
      n_print_step = 200
      if (i + 1) % n_print_step == 0:
        print('Epoch %d takes %.3f s to process %d batches, running loss %.5f' % (epoch, time.time()-begin_time, i, running_loss / n_print_step))
        running_loss = 0.

    print('Epoch %d takes %.3f s to finish' % (epoch, time.time() - begin_time))
    print('Final running loss for epoch %d: %.5f' % (epoch, running_loss / min(len(train_loader), n_print_step)))
    val_loss = validate(audio_model, test_loader, args)
    
    # Save the weights of the model
    if val_loss > best_val_loss:
      best_val_loss = val_loss
      if not os.path.isdir('%s' % exp_dir):
        os.mkdir('%s' % exp_dir)

      torch.save(audio_model.state_dict(),
              '%s/audio_model.%d.pth' % (exp_dir, epoch))  
      with open('%s/validation_loss_%d.txt' % (exp_dir, epoch), 'w') as f:
        f.write('%.5f' % val_loss)

def validate(audio_model, test_loader, args):
  if not isinstance(audio_model, torch.nn.DataParallel):
    audio_model = nn.DataParallel(audio_model)

  if torch.cuda.is_available():
    audio_model = audio_model.cuda()
  
  loss = 0
  total = 0
  begin_time = time.time()
  embed1_all = {}
  segmentations_all = {}
  labels_all = {}
  criterion = nn.MSELoss()
  
  n_print_step = 20
  nframes_all = []
  with torch.no_grad():  
    for i, audio_input in enumerate(test_loader):
      # XXX
      # print(i)
      # if i < 619:
      #   continue
      if args.eval_phone_recognition:
        audios, labels, segmentations = audio_input
      else:
        audios, _ = audio_input
      audios = Variable(audios)
      if torch.cuda.is_available():
        audios = audios.cuda()

      embeds1, outputs = audio_model(audios, save_features=True) 
      loss += len(audios) / float(args.batch_size) * criterion(outputs, audios).data.cpu().numpy() 
      total += len(audios) / float(args.batch_size)       
      if args.save_features or args.eval_phone_recognition:
        for i_b in range(embeds1.size()[0]):
          feat_id = 'arr_'+str(i * args.batch_size + i_b)
          embed1_all[feat_id] = embeds1[i_b].data.cpu().numpy() 
          if args.eval_phone_recognition:
            segmentations_all[feat_id] = segmentations[i_b].data.cpu().numpy()
            labels_all[feat_id] = labels[i_b].data.cpu().numpy()

      if (i + 1) % n_print_step == 0:
        print('Takes %.3f s to process %d batches, MSE loss: %.5f' % (time.time() - begin_time, i, loss / (i + 1)))
    
  if not os.path.isdir('%s' % args.exp_dir):
    os.mkdir('%s' % args.exp_dir)

  np.savez(args.exp_dir+'/embed1_all.npz', **embed1_all) 
  if args.eval_phone_recognition:
    evaluate_phone_recognition(embed1_all, segmentations_all, labels_all, args)
  return  loss / total

def evaluate_phone_recognition(embeds, segmentations, labels, args):
  # Break the sentence features into phone features
  X = []
  y = []
  for feat_id in sorted(embeds, key=lambda x:int(x.split('_')[-1])): 
    embed = embeds[feat_id]
    segmentation = np.nonzero(segmentations[feat_id])[0]
    label = labels[feat_id]
    for i_seg, (begin, end) in enumerate(zip(segmentation[:-1], segmentation[1:])):
      begin, end = int(begin / args.downsample_rate), int(end / args.downsample_rate) + 1
      X.append(np.mean(embed[:, begin:end], axis=1))
      y.append(label[i_seg]) 
  X, y = np.asarray(X), np.asarray(y)
  if not args.class2id_file:
    n_class = np.max(y)
    class2id = {c:c for c in range(n_class)}
  else:
    with open(args.class2id_file, 'r') as f:
      class2id = json.load(f)
    n_class = len(class2id)

  # Train an SVM with the features and report training accuracy
  clf = LinearSVC()
  clf.fit(X, y)
  y_pred = clf.predict(X)
  
  # Print class accuracy
  correct = 0
  total = 0
  class_correct = np.zeros((n_class,))
  class_total = np.zeros((n_class,))
  for i in range(len(y)):
    correct += (y[i] == y_pred[i])
    total += 1
    class_correct[y[i]] += (y[i] == y_pred[i])
    class_total[y[i]] += 1

  accs = class_correct / np.maximum(class_total, 1)
  top_10_classes = sorted(class2id, key=lambda x:accs[class2id[x]], reverse=True)[:10]
  top_10_accs = [accs[class2id[c]] for c in top_10_classes]
  for c, acc in zip(top_10_classes, top_10_accs):
    print('%s: %d %%' % (c, acc * 100))

  with open(args.exp_dir + '/phone_error_rates.txt', 'w') as f:
    for c in class2id:
      f.write('%s: %d %%' % (c, class_correct[class2id[c]] / np.maximum(class_total[class2id[c]], 1) * 100))
