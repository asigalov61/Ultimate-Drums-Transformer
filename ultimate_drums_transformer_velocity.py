# -*- coding: utf-8 -*-
"""Ultimate_Drums_Transformer_Velocity.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/github/asigalov61/Ultimate-Drums-Transformer/blob/main/Ultimate_Drums_Transformer_Velocity.ipynb

# Ultimate Drums Transformer (ver. 4.0)

***

Powered by tegridy-tools: https://github.com/asigalov61/tegridy-tools

***

WARNING: This complete implementation is a functioning model of the Artificial Intelligence. Please excercise great humility, care, and respect. https://www.nscai.gov/

***

#### Project Los Angeles

#### Tegridy Code 2024

***

# (GPU CHECK)
"""

#@title NVIDIA GPU check
!nvidia-smi

"""# (SETUP ENVIRONMENT)"""

#@title Install dependencies
!git clone --depth 1 https://github.com/asigalov61/Ultimate-Drums-Transformer
!pip install huggingface_hub
!pip install einops
!pip install torch-summary
!apt install fluidsynth #Pip does not work for some reason. Only apt works

# Commented out IPython magic to ensure Python compatibility.
#@title Import modules

print('=' * 70)
print('Loading core Ultimate Drums Transformer modules...')

import os
import copy
import pickle
import secrets
import statistics
from time import time
import tqdm

print('=' * 70)
print('Loading main Ultimate Drums Transformer modules...')
import torch

# %cd /content/Ultimate-Drums-Transformer

import TMIDIX

from midi_to_colab_audio import midi_to_colab_audio

from x_transformer_1_23_2 import *

import random

# %cd /content/
print('=' * 70)
print('Loading aux Ultimate Drums Transformer modules...')

import matplotlib.pyplot as plt

from torchsummary import summary
from sklearn import metrics

from IPython.display import Audio, display

from huggingface_hub import hf_hub_download

from google.colab import files

print('=' * 70)
print('Done!')
print('Enjoy! :)')
print('=' * 70)

"""# (LOAD MODEL)"""

#@title Load Ultimate Drums Transformer Pre-Trained Model

#@markdown Model precision option

model_precision = "bfloat16" # @param ["bfloat16", "float16"]

#@markdown bfloat16 == Half precision/faster speed (if supported, otherwise the model will default to float16)

#@markdown float16 == Full precision/fast speed

plot_tokens_embeddings = False # @param {type:"boolean"}

print('=' * 70)
print('Loading Ultimate Drums Transformer Pre-Trained Model...')
print('Please wait...')
print('=' * 70)

full_path_to_models_dir = "/content/Ultimate-Drums-Transformer/Models"

model_checkpoint_file_name = 'Ultimate_Drums_Transformer_Small_Trained_Model_VER4_VEL_4L_14597_steps_0.3894_loss_0.876_acc.pth'
model_path = full_path_to_models_dir+'/Small_V4_VEL/'+model_checkpoint_file_name
if os.path.isfile(model_path):
  print('Model already exists...')

else:
  hf_hub_download(repo_id='asigalov61/Ultimate-Drums-Transformer',
                  filename=model_checkpoint_file_name,
                  local_dir='/content/Ultimate-Drums-Transformer/Models/Small_V4_VEL',
                  local_dir_use_symlinks=False)

print('=' * 70)
print('Instantiating model...')

device_type = 'cuda'

if model_precision == 'bfloat16' and torch.cuda.is_bf16_supported():
  dtype = 'bfloat16'
else:
  dtype = 'float16'

if model_precision == 'float16':
  dtype = 'float16'

ptdtype = {'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
ctx = torch.amp.autocast(device_type=device_type, dtype=ptdtype)

SEQ_LEN = 8192 # Models seq len
PAD_IDX = 393 # Models pad index

# instantiate the model

model = TransformerWrapper(
    num_tokens = PAD_IDX+1,
    max_seq_len = SEQ_LEN,
    attn_layers = Decoder(dim = 1024, depth = 4, heads = 16, attn_flash = True)
    )

model = AutoregressiveWrapper(model, ignore_index = PAD_IDX, pad_value=PAD_IDX)

model.cuda()
print('=' * 70)

print('Loading model checkpoint...')

model.load_state_dict(torch.load(model_path))
print('=' * 70)

model.eval()

print('Done!')
print('=' * 70)

print('Model will use', dtype, 'precision...')
print('=' * 70)

# Model stats
print('Model summary...')
summary(model)

# Plot Token Embeddings
if plot_tokens_embeddings:
  tok_emb = model.net.token_emb.emb.weight.detach().cpu().tolist()

  cos_sim = metrics.pairwise_distances(
    tok_emb, metric='cosine'
  )
  plt.figure(figsize=(7, 7))
  plt.imshow(cos_sim, cmap="inferno", interpolation="nearest")
  im_ratio = cos_sim.shape[0] / cos_sim.shape[1]
  plt.colorbar(fraction=0.046 * im_ratio, pad=0.04)
  plt.xlabel("Position")
  plt.ylabel("Position")
  plt.tight_layout()
  plt.plot()
  plt.savefig("/content/Ultimate-Drums-Transformer-Tokens-Embeddings-Plot.png", bbox_inches="tight")

"""# (GENERATE)

# (IMPROV)
"""

#@title Standard Improv Generator

#@markdown Generation settings

melody_MIDI_patch_number = 0 # @param {type:"slider", min:0, max:127, step:1}
number_of_tokens_tp_generate = 258 # @param {type:"slider", min:30, max:8190, step:3}
number_of_batches_to_generate = 4 #@param {type:"slider", min:1, max:16, step:1}
temperature = 0.9 # @param {type:"slider", min:0.1, max:1, step:0.05}

#@markdown Other settings

render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)
print('Ultimate Drums Transformer Standard Improv Model Generator')
print('=' * 70)

outy = [random.randint(1, 32)]

print('Selected Improv sequence:')
print(outy)
print('=' * 70)

torch.cuda.empty_cache()

inp = [outy] * number_of_batches_to_generate

inp = torch.LongTensor(inp).cuda()

with ctx:
  out = model.generate(inp,
                        number_of_tokens_tp_generate,
                        temperature=temperature,
                        return_prime=True,
                        verbose=True)

out0 = out.tolist()

print('=' * 70)
print('Done!')
print('=' * 70)

torch.cuda.empty_cache()

#======================================================================

print('Rendering results...')

for i in range(number_of_batches_to_generate):

  print('=' * 70)
  print('Batch #', i)
  print('=' * 70)

  out1 = out0[i]

  print('Sample INTs', out1[:12])
  print('=' * 70)

  if len(out1) != 0:

      song = out1
      song_f = []

      time = 0
      dtime = 0
      dur = 128
      vel = 90
      pitch = 0
      channel = 0

      patches = [0] * 16
      patches[0] = melody_MIDI_patch_number

      for ss in song:

          if 0 < ss < 128:

              ptime = time

              time += ss * 32

              dtime = ptime

              song_f.append(['note', ptime, dur, 0, random.choice([60, 62, 64]), vel, melody_MIDI_patch_number])

          if 128 <= ss < 256:

              dtime += (ss-128) * 32

          if 256 <= ss < 384:

              pitch = (ss-256)

          if 384 <= ss < 393:

              vel = (ss-384) * 15

              song_f.append(['note', dtime, dur, 9, pitch, vel, 128])

      data = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                      output_signature = 'Ultimate Drums Transformer',
                                                      output_file_name = '/content/Ultimate-Drums-Transformer-Composition_'+str(i),
                                                      track_name='Project Los Angeles',
                                                      list_of_MIDI_patches=patches
                                                      )


      print('=' * 70)
      print('Displaying resulting composition...')
      print('=' * 70)

      fname = '/content/Ultimate-Drums-Transformer-Composition_'+str(i)

      if render_MIDI_to_audio:
        midi_audio = midi_to_colab_audio(fname + '.mid')
        display(Audio(midi_audio, rate=16000, normalize=False))

      TMIDIX.plot_ms_SONG(song_f, plot_title=fname)

"""# (DRUMS TRACK GENERATION)"""

#@title Load Seed MIDI

#@markdown Press play button to to upload your own seed MIDI or to load one of the provided sample seed MIDIs from the dropdown list below

select_seed_MIDI = "Upload your own custom MIDI" # @param ["Upload your own custom MIDI", "Ultimate-Drums-Transformer-Melody-Seed-1", "Ultimate-Drums-Transformer-Melody-Seed-2", "Ultimate-Drums-Transformer-Melody-Seed-3", "Ultimate-Drums-Transformer-Melody-Seed-4", "Ultimate-Drums-Transformer-Melody-Seed-5", "Ultimate-Drums-Transformer-Melody-Seed-6", "Ultimate-Drums-Transformer-MI-Seed-1", "Ultimate-Drums-Transformer-MI-Seed-2", "Ultimate-Drums-Transformer-MI-Seed-3", "Ultimate-Drums-Transformer-MI-Seed-4"]
render_MIDI_to_audio = False # @param {type:"boolean"}

print('=' * 70)
print('Ultimate Drums Transformer Seed MIDI Loader')
print('=' * 70)

f = ''

if select_seed_MIDI != "Upload your own custom MIDI":
  print('Loading seed MIDI...')
  f = '/content/Ultimate-Drums-Transformer/Seeds/'+select_seed_MIDI+'.mid'

else:
  print('Upload your own custom MIDI...')
  print('=' * 70)
  uploaded_MIDI = files.upload()
  if list(uploaded_MIDI.keys()):
    f = list(uploaded_MIDI.keys())[0]

if f != '':

  print('=' * 70)
  print('File:', f)
  print('=' * 70)

  #=======================================================
  # START PROCESSING

  #===============================================================================
  # Raw single-track ms score

  raw_score = TMIDIX.midi2single_track_ms_score(f)

  #===============================================================================
  # Enhanced score notes

  escore_notes = TMIDIX.advanced_score_processor(raw_score, return_enhanced_score_notes=True)[0]

  #=======================================================
  # PRE-PROCESSING

  #===============================================================================
  # Augmented enhanced score notes

  escore_notes = [e for e in escore_notes if e[3] != 9]

  escore_notes = TMIDIX.augment_enhanced_score_notes(escore_notes)

  patches = TMIDIX.patch_list_from_enhanced_score_notes(escore_notes)

  dscore = TMIDIX.delta_score_notes(escore_notes, compress_timings=True, even_timings=True)

  cscore = TMIDIX.chordify_score([d[1:] for d in dscore])

  #=======================================================

  song_f = escore_notes

  for s in song_f:
    s[1] *= 16
    s[2] *= 16

  time = 0
  dur = 0
  vel = 90
  pitch = 0
  channel = 0


  detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                            output_signature = 'Ultimate Drums Transformer',
                                                            output_file_name = '/content/Ultimate-Drums-Transformer-Seed-Composition',
                                                            track_name='Project Los Angeles',
                                                            list_of_MIDI_patches=patches
                                                            )
  #=======================================================

  print('=' * 70)
  print('Composition stats:')
  print('Composition has', len(cscore), 'chords')
  print('Composition MIDI patches:', sorted(set(patches)))
  print('=' * 70)

  print('Displaying resulting composition...')
  print('=' * 70)

  fname = '/content/Ultimate-Drums-Transformer-Seed-Composition'

  if render_MIDI_to_audio:
    midi_audio = midi_to_colab_audio(fname + '.mid')
    display(Audio(midi_audio, rate=16000, normalize=False))

  TMIDIX.plot_ms_SONG(song_f, plot_title=fname)

else:
  print('=' * 70)

#@title Drums track generation

#@markdown Generation settings
generate_from = "Beginning" # @param ["Beginning", "Last Position"]
number_of_chords_to_generate_drums_for = 64 # @param {type:"slider", min:4, max:8192, step:4}
drums_generation_step_in_chords = 2 # @param {type:"slider", min:1, max:4, step:1}
max_number_of_drums_pitches_per_step = 3 # @param {type:"slider", min:1, max:16, step:1}
number_of_memory_tokens = 4096 # @param {type:"slider", min:32, max:8188, step:16}
temperature = 0.9 # @param {type:"slider", min:0.1, max:1, step:0.05}

#@markdown Other settings
render_MIDI_to_audio = True # @param {type:"boolean"}

print('=' * 70)
print('Ultimate Drums Transformer Drums Track Generator')
print('=' * 70)

#===============================================================================

def generate_drums(input_seq,
                   max_drums_limit = 3,
                   num_memory_tokens = 4096,
                   temperature=0.9):

    x = torch.tensor([input_seq] * 1, dtype=torch.long, device='cuda')

    o = 128

    ncount = 0

    time = 0

    ntime = input_seq[-1]

    while o > 127 and ncount < max_drums_limit and time < ntime:
      with ctx:
        out = model.generate(x[-num_memory_tokens:],
                            1,
                            temperature=temperature,
                            return_prime=False,
                            verbose=False)

      o = out.tolist()[0][0]

      if 128 <= o < 256:
        ncount = 0
        time += (o-128)

      if 384 <= o < 393:
        ncount += 1

      if o > 127 and time < ntime:
        x = torch.cat((x, out), 1)

    return x.tolist()[0][len(input_seq):]

#===============================================================================

sum_by_step = lambda lst, step: [sum(lst[i:i+step]) for i in range(0, len(lst), step)]

#===============================================================================

comp_times = [t[1] for t in dscore if t[1] != 0]

comp_times = list(sum_by_step(comp_times, drums_generation_step_in_chords))

if generate_from == 'Beginning':

  print('Generating drums track...')
  print('=' * 70)

  output = []

  torch.cuda.empty_cache()

  for c in tqdm.tqdm(comp_times[:number_of_chords_to_generate_drums_for]):

    try:

      output.append(c)

      out = generate_drums(output,
                          temperature=temperature,
                          max_drums_limit=max_number_of_drums_pitches_per_step,
                          num_memory_tokens=number_of_memory_tokens
                          )


      output.extend(out)

    except KeyboardInterrupt:
      print('Stopping generation...')
      break

    except:
      break

  torch.cuda.empty_cache()

else:

  pidx = sum([1 for o in output if o < 128])

  if pidx > 0 and pidx < len(comp_times[:number_of_chords_to_generate_drums_for]):

    #===============================================================================

    print('Continuing generating drums track...')
    print('=' * 70)

    torch.cuda.empty_cache()

    for c in tqdm.tqdm(comp_times[pidx:number_of_chords_to_generate_drums_for]):

      try:
        output.append(c)

        out = generate_drums(output,
                            temperature=temperature,
                            max_drums_limit=max_number_of_drums_pitches_per_step,
                            num_memory_tokens=number_of_memory_tokens
                            )
        output.extend(out)

      except KeyboardInterrupt:
        print('=' * 70)
        print('Stopping generation...')
        break

      except:
        break

    torch.cuda.empty_cache()

  else:
    print('Nothing to continue!')
    print('Please start from the begining...')

#===============================================================================

print('=' * 70)
print('Done!')
print('=' * 70)

#===============================================================================

print('Rendering results...')

print('=' * 70)
print('Sample INTs', output[:12])
print('=' * 70)

if len(output) != 0:

    song = output
    song_f = []

    time = 0
    dtime = 0
    dur = 32
    vel = 90
    pitch = 0
    channel = 0

    for ss in song:

        if 0 < ss < 128:

            ptime = time

            time += ss * 32

            dtime = ptime

        if 128 <= ss < 256:

            dtime += (ss-128) * 32

        if 256 <= ss < 384:

            pitch = (ss-256)

        if 384 <= ss < 393:

            vel = (ss-384) * 15

            song_f.append(['note', dtime, dur, 9, pitch, vel, 128])

#===============================================================================

original_score = []
time = 0

pidx = sum([1 for o in output if o < 128])

for c in cscore[:((pidx+1) * drums_generation_step_in_chords)]:
  for cc in c:
    time += cc[0] * 32
    dur = cc[1] * 32
    original_note = ['note'] + copy.deepcopy(cc)
    original_note[1] = time
    original_note[2] = dur
    original_score.append(original_note)

song_f = sorted(original_score + song_f, key=lambda x: x[1])

#===============================================================================

detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                          output_signature = 'Ultimate Drums Transformer',
                                                          output_file_name = '/content/Ultimate-Drums-Transformer-Composition',
                                                          track_name='Project Los Angeles',
                                                          list_of_MIDI_patches=patches
                                                          )

#=========================================================================

print('=' * 70)
print('Displaying resulting composition...')
print('=' * 70)

fname = '/content/Ultimate-Drums-Transformer-Composition'

if render_MIDI_to_audio:
  midi_audio = midi_to_colab_audio(fname + '.mid')
  display(Audio(midi_audio, rate=16000, normalize=False))

TMIDIX.plot_ms_SONG(song_f, plot_title=fname)

"""# Congrats! You did it! :)"""