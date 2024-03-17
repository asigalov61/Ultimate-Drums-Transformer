import argparse
import glob
import json
import os.path

import time as reqtime
import datetime
from pytz import timezone

import torch

import gradio as gr

from x_transformer_1_23_2 import *
import random
import tqdm

from midi_to_colab_audio import midi_to_colab_audio
import TMIDIX

import matplotlib.pyplot as plt

in_space = os.getenv("SYSTEM") == "spaces"

# =================================================================================================

def generate_drums(notes_times,
                   max_drums_limit = 8,
                   num_memory_tokens = 4096,
                   temperature=0.9):

    x = torch.tensor([notes_times] * 1, dtype=torch.long, device=DEVICE)

    o = 128

    ncount = 0

    while o > 127 and ncount < max_drums_limit:
      with ctx:
        out = model.generate(x[-num_memory_tokens:],
                            1,
                            temperature=temperature,
                            return_prime=False,
                            verbose=False)

      o = out.tolist()[0][0]

      if 256 <= o < 384:
        ncount += 1

      if o > 127:
        x = torch.cat((x, out), 1)

    return x.tolist()[0][len(notes_times):]

# =================================================================================================

@torch.no_grad()
def GenerateDrums(input_midi, input_num_tokens):
    print('=' * 70)
    print('Req start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    start_time = reqtime.time()

    fn = os.path.basename(input_midi.name)
    fn1 = fn.split('.')[0]

    input_num_tokens = int(input_num_tokens)

    print('-' * 70)
    print('Input file name:', fn)
    print('Req num toks:', input_num_tokens)
    print('-' * 70)

    #===============================================================================
    # Raw single-track ms score
    
    raw_score = TMIDIX.midi2single_track_ms_score(input_midi.name)
    
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
    
    cscore_melody = [c[0] for c in cscore]
    
    comp_times = [0] + [t[1] for t in dscore if t[1] != 0]

    #===============================================================================

    print('=' * 70)
    
    print('Sample output events', escore_notes[:5])
    print('=' * 70)
    print('Generating...')

    output = []

    for c in comp_times[:input_num_tokens]:
      output.append(c)
    
      out = generate_drums(output,
                           temperature=0.9,
                           max_drums_limit=8,
                           num_memory_tokens=4096
                           )
      output.extend(out)

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
        ntime = 0
        dur = 32
        vel = 90
        vels = [100, 120]
        pitch = 0
        channel = 0
    
        idx = 0
    
        for ss in song:
    
            if 0 <= ss < 128:

                time += cscore[idx][0][0] * 32
    
                for c in cscore[idx]:
                  song_f.append(['note', time, c[1] * 32, c[2], c[3], c[4], c[5]])
                    
                dtime = time
                
                idx += 1
    
            if 128 <= ss < 256:
    
                dtime += (ss-128) * 32
    
            if 256 <= ss < 384:
    
                pitch = (ss-256)
    
            if 384 <= ss < 393:
            
                vel = ((ss-384)+1) * 15
            
                if dtime == time:
                    song_f.append(['note', time, dur, 9, pitch, vel, 128])
                else:
                    song_f.append(['note', dtime, dur, 9, pitch, vel, 128])
    
    detailed_stats = TMIDIX.Tegridy_ms_SONG_to_MIDI_Converter(song_f,
                                                              output_signature = 'Ultimate Drums Transformer',
                                                              output_file_name = fn1,
                                                              track_name='Project Los Angeles',
                                                              list_of_MIDI_patches=patches
                                                              )
    
    new_fn = fn1+'.mid'
            
    
    audio = midi_to_colab_audio(new_fn, 
                        soundfont_path=soundfont,
                        sample_rate=16000,
                        volume_scale=10,
                        output_for_gradio=True
                        )
    
    print('Done!')
    print('=' * 70)

    #========================================================

    output_midi_title = str(fn1)
    output_midi_summary = str(song_f[:3])
    output_midi = str(new_fn)
    output_audio = (16000, audio)
    
    output_plot = TMIDIX.plot_ms_SONG(song_f, plot_title=output_midi, return_plt=True)

    print('Output MIDI file name:', output_midi)
    print('Output MIDI title:', output_midi_title)
    print('Output MIDI summary:', '')
    print('=' * 70) 
    

    #========================================================
    
    print('-' * 70)
    print('Req end time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('-' * 70)
    print('Req execution time:', (reqtime.time() - start_time), 'sec')
    
    yield output_midi_title, output_midi_summary, output_midi, output_audio, output_plot

# =================================================================================================

if __name__ == "__main__":
    
    PDT = timezone('US/Pacific')
    
    print('=' * 70)
    print('App start time: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now(PDT)))
    print('=' * 70)
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", default=False, help="share gradio app")
    parser.add_argument("--port", type=int, default=7860, help="gradio server port")
    opt = parser.parse_args()

    soundfont = "SGM-v2.01-YamahaGrand-Guit-Bass-v2.7.sf2"
    
    print('Loading model...')

    SEQ_LEN = 8192 # Models seq len
    PAD_IDX = 393 # Models pad index
    DEVICE = 'cpu' # 'cuda'

    # instantiate the model

    model = TransformerWrapper(
        num_tokens = PAD_IDX+1,
        max_seq_len = SEQ_LEN,
        attn_layers = Decoder(dim = 1024, depth = 4, heads = 8, attn_flash = True)
        )
    
    model = AutoregressiveWrapper(model, ignore_index = PAD_IDX)

    model.to(DEVICE)
    print('=' * 70)

    print('Loading model checkpoint...')

    model.load_state_dict(
        torch.load('Ultimate_Drums_Transformer_Small_Trained_Model_VER3_VEL_11222_steps_0.5749_loss_0.8085_acc.pth',
                   map_location=DEVICE))
    print('=' * 70)

    model.eval()

    if DEVICE == 'cpu':
        dtype = torch.bfloat16
    else:
        dtype = torch.float16

    ctx = torch.amp.autocast(device_type=DEVICE, dtype=dtype)

    print('Done!')
    print('=' * 70)

    app = gr.Blocks()
    with app:
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Ultimate Drums Transformer</h1>")
        gr.Markdown("<h1 style='text-align: center; margin-bottom: 1rem'>Generate unique drums track for any MIDI</h1>")
        gr.Markdown(
            "![Visitors](https://api.visitorbadge.io/api/visitors?path=asigalov61.Ultimate-Drums-Transformer&style=flat)\n\n"
            "SOTA pure drums transformer which is capable of drums track generation for any source composition\n\n"
            "Check out [Ultimate Drums Transformer](https://github.com/asigalov61/Ultimate-Drums-Transformer) on GitHub!\n\n"
            "[Open In Colab]"
            "(https://colab.research.google.com/github/asigalov61/Ultimate-Drums-Transformer/blob/main/Ultimate_Drums_Transformer.ipynb)"
            " for faster execution and endless generation"
        )
        gr.Markdown("## Upload your MIDI or select a sample example MIDI")
        
        input_midi = gr.File(label="Input MIDI", file_types=[".midi", ".mid", ".kar"])
        input_num_tokens = gr.Slider(16, 512, value=256, label="Number of Tokens", info="Number of tokens to generate")
        
        run_btn = gr.Button("generate", variant="primary")

        gr.Markdown("## Generation results")

        output_midi_title = gr.Textbox(label="Output MIDI title")
        output_midi_summary = gr.Textbox(label="Output MIDI summary")
        output_audio = gr.Audio(label="Output MIDI audio", format="wav", elem_id="midi_audio")
        output_plot = gr.Plot(label="Output MIDI score plot")
        output_midi = gr.File(label="Output MIDI file", file_types=[".mid"])


        run_event = run_btn.click(GenerateDrums, [input_midi, input_num_tokens],
                                  [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot])

        gr.Examples(
            [["Ultimate-Drums-Transformer-Melody-Seed-1.mid", 256], 
             ["Ultimate-Drums-Transformer-Melody-Seed-2.mid", 256], 
             ["Ultimate-Drums-Transformer-Melody-Seed-3.mid", 256],
             ["Ultimate-Drums-Transformer-Melody-Seed-4.mid", 256],
             ["Ultimate-Drums-Transformer-Melody-Seed-5.mid", 256],
             ["Ultimate-Drums-Transformer-Melody-Seed-6.mid", 256],
             ["Ultimate-Drums-Transformer-MI-Seed-1.mid", 256],
             ["Ultimate-Drums-Transformer-MI-Seed-2.mid", 256],
             ["Ultimate-Drums-Transformer-MI-Seed-3.mid", 256],
             ["Ultimate-Drums-Transformer-MI-Seed-4.mid", 256]],
            [input_midi, input_num_tokens],
            [output_midi_title, output_midi_summary, output_midi, output_audio, output_plot],
            GenerateDrums,
            cache_examples=False,
        )
        
        app.queue(concurrency_count=1).launch(server_port=opt.port, share=opt.share, inbrowser=True)