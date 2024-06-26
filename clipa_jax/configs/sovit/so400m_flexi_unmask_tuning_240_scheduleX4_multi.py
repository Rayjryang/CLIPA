# #Copyright @2023 Xianhang Li
#
# # This code is based on materials from the Big Vision [https://github.com/google-research/big_vision].
# # Thanks to Big Vision  for their contributions to the field of computer vision and for their open-source contributions to this project.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: disable=line-too-long

import configs.common as bvcc
import configs.clip_common as common
from ml_collections import ConfigDict


def get_config(arg=None):
  """The base configuration."""
  arg = bvcc.parse_arg(
      arg,  res=240, runlocal=False, batchsize=32768,  token_len=32, txt='bert_base', img='So400m/14',
      init='', img_head=True, load_pretrain=False)
  img_name, img_init = common.inits[arg.img]
  txt_name, txt_init = common.inits[arg.txt]
  config = ConfigDict()

    
 # input section include augmentation
  config.input = {}
  #config.input.data = dict(name='liaon-400m', split='full', data_dir='[your data(laion-400m) location]')
  config.input.data = dict(name='datacomp1b', split='full', data_dir='[your data(laion-400m) location]')
  config.input.cach_raw = True
  config.input.shuffle_buffer_size = 250_000  if not arg.runlocal else 50
  config.init_shapes = [(1, arg.res, arg.res, 3), (1, arg.token_len,)]
  config.init_types = ['float32', 'int32']
  config.input.batch_size = arg.batchsize
  if arg.init:
    vocab_path = arg.init.rsplit('.', 1)[0] + '.txt'
  else:
    vocab_path = f'{txt_init}/vocab.txt'
  tokenizer = lambda inkey: (
      f'bert_tokenize(inkey="{inkey}", max_len={arg.token_len}, '
      f'vocab_path="{vocab_path}")')
  config.input.pp = pp_eval = (
      f'inception_crop(inkey="jpg", size={arg.res}, area_min=40, method="bilinear", antialias=True)|simclr_jitter_gray(jitter_strength=0.4)'
      f'|flatten|{tokenizer("txt")}|keep("image", "labels")'
  )
  config.pp_modules = [
      'ops_general', 'ops_image', 'ops_text', 'bert_ops']

  config.cpu_unit8 = True
  config.mask_ratio = 0.0
 
  # Model section
  config.model_name = 'two_towers'
  config.model_load = {}
  config.model = ConfigDict()
  config.model.image_model = 'flexi_model'
  config.model.text_model = 'text_transformer'
  config.model.image = ConfigDict({
      'variant': 'So400m',
      'pool_type': 'gap',
      'posemb': 'sincos2d',
      'patch_size': (20, 20),
      'posemb_size': (7, 7),
      'remat_policy': 'actcp', #gradient checkpointing
      'head_zeroinit': False,
  })
  config.model.text = ConfigDict({
      'variant': 'H/14',
      'pool_type': 'last',
      'remat_policy': 'actcp',
      'head_zeroinit': False,
  })
  config.model.temperature_init = 1/0.07
  #dim = {'T': 192, 'S': 384, 'B': 512, 'L': 768, 'H': 1024}[arg.img[0]]
  dim = {'T': 192, 'S':384, 'B': 512, 'L': 768, 'H': 1024 , 'So400m': 1024}[arg.img.split('/')[0]]
  config.model.out_dim = (dim if arg.img_head else None, dim)  # (image_out_dim, text_out_dim)

  config.flexi = dict()
  config.flexi.seqhw = dict(
        # The settings to sample from. Corresponding patch-sizes at 240px:
        # 48, 40, 30, 24, 20, 16, 15, 12, 10, 8
        # v=(5, 6, 8, 10, 12, 15, 16, 20, 24, 30),
        # The probabilities/weights of them. Default uniform.
        # p=(1, 1, 1, 1, 1, 1, 1, 1, 1, 1),
        v=(6, 8, 10),
        p=(1, 1, 1),
    )
  
  config.test_seqhw = 8

  # load pre-trained ckpt
  config.masked_init = "[your pre-trained weight lcoation]"

  # optimizer config
  config.optax_name = 'scale_by_adam'
  config.total_steps = int(131072000 * 8 * 1.39  // arg.batchsize)  # seen_samples // batchsize to get the number of steps
  config.lr = 4e-7 * (arg.batchsize // 256)
  config.wd = 0.2
  warmup_steps = int(26214400 // arg.batchsize) # seen_samples // batchsize to get the number of steps
  config.schedule = [
      ('.*', dict(decay_type='cosine', warmup_steps=warmup_steps, min_lr=0, max_lr=4e-7 * (arg.batchsize // 256))),
  ]

  config.optax = dict(mu_dtype='bfloat16',  b1=0.9,  b2=0.95)
  config.loss_use_global_batch = True
  config.local_loss = True


  # log section
  config.log_training_steps = 50
  config.ckpt_steps = 1000
  config.wandb = dict(
      log_wandb=True,
      wandb_offline=False,
      resume=False,
      debug_data=False,
      project='clip_scaling',
      experiment=f'v3-256-flexi_sovit400m14_datacomp1b_32k_{arg.res}_{arg.token_len}_gap_sin2d_H14_2.56b_FLOPs_ft_240_1024m',
      entity='1999ray9999'
  )
  config.save_ckpt = True
  
  config.masked_no_load = {'dont_load': ['.*img/embedding.*','.*img/pos_embedding.*','.*txt/pos_embedding.*','img/cls', 'img/embedding/bias']}
#   config.masked_no_load = {'dont_load': ['.*txt/pos_embedding.*','img/cls', 'img/embedding/bias']}

#   config.masked_no_load = {'dont_load': ['.*txt/pos_embedding.*']}
    

  # Eval section (Both few-shot and zero-shot)
#   config.eval_only = False
  config.eval_only = True
  eval_common = dict(
      type='proj.image_text.contrastive',
      use_global_batch=config.loss_use_global_batch,
      log_steps=1000,
  )

  config.evals = {}
  sub = '[:4]' if arg.runlocal else ''

#   config.evals.disclf = {}
#   config.evals.disclf.dataset_names = ['imagenet2012']
#   config.evals.disclf.split = f'validation{sub}'
#   config.evals.disclf.data_dir = 'gs://celt-tfds-imagenet-eu'
#   config.evals.disclf.pp_img = f'|resize({arg.res}, method="bilinear", antialias=True)|vgg_value_range' # directly resize works better
#   config.evals.disclf.pp_txt = tokenizer('texts')
#   config.evals.disclf.type = 'proj.image_text.discriminative_classifier'
#   config.evals.disclf.prefix = 'z/0shot/'
#   config.evals.disclf.log_steps = eval_common['log_steps']

  for v in [6,8,10]:
    disclf = 'disclf' + f'_seqhw_{v}'

    config.evals[disclf] = {}

    config.evals[disclf]['dataset_names'] = ['imagenet2012']
    config.evals[disclf]['split'] = f'validation{sub}'
    config.evals[disclf]['data_dir'] = 'gs://tfds-imagenet-us-east1'
    config.evals[disclf]['pp_img'] = f'|resize_small({arg.res}, method="bilinear", antialias=True)|central_crop({arg.res})|vgg_value_range'
    config.evals[disclf]['pp_txt'] = tokenizer('texts')
    config.evals[disclf]['type'] = 'proj.image_text.discriminative_classifier'
    config.evals[disclf]['prefix'] = f'z/0shot/seqhw_{v}/'
    config.evals[disclf]['log_steps'] = eval_common['log_steps']
    config.evals[disclf]['test_seqhw'] = v


  config.seed = 0
  config.l = config.m = 0

  return config
