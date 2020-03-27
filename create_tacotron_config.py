import yaml

with open('conf/train_pytorch_tacotron2.yaml', 'r') as f:
  params = yaml.load(f, Loader=yaml.Loader)

params.update({
    "embed-dim": 16,
    "elayers": 1, 
    "eunits": 16,
    "econv-layers": 1,
    "econv-chans": 16,
    "econv-filts": 5,
    "dlayers": 1,
    "dunits": 16,
    "prenet-layers": 1,
    "prenet-units": 16,
    "postnet-layers": 1,
    "postnet-chans": 16,
    "postnet-filts": 5,
    "adim": 16,
    "aconv-chans": 16,
    "aconv-filts": 5,
    "reduction-factor": 5,
    "batch-size": 32,
    "epochs": 5,
    "report-interval-iters": 10,  
})

with open('conf/train_pytorch_tacotron2_mini.yaml', 'w') as f:
  yaml.dump(params, f, Dumper=yaml.Dumper)
