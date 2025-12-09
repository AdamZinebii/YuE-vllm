"""
YuE Vocal-Only Generation with Instrumental Conditioning

This script generates vocals conditioned on a provided instrumental track.
It uses teacher-forcing to fix the instrumental tokens while generating
matching vocal tokens.

Usage:
    python infer_vocal_only.py \
        --instrumental_path your_instrumental.mp3 \
        --genre_txt ../prompt_egs/genre.txt \
        --lyrics_txt ../prompt_egs/lyrics.txt \
        --output_dir ../output
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer'))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'xcodec_mini_infer', 'descriptaudiocodec'))
import re
import random
import uuid
import copy
from tqdm import tqdm
from collections import Counter
import argparse
import numpy as np
import torch
import torchaudio
from torchaudio.transforms import Resample
import soundfile as sf
from einops import rearrange
from transformers import AutoModelForCausalLM, LogitsProcessor, LogitsProcessorList
from omegaconf import OmegaConf
from codecmanipulator import CodecManipulator
from mmtokenizer import _MMSentencePieceTokenizer
from models.soundstream_hubert_new import SoundStream
from vocoder import build_codec_model, process_audio
from post_process_audio import replace_low_freq_with_energy_matched


parser = argparse.ArgumentParser()
# Model Configuration
parser.add_argument("--stage1_model", type=str, default="../models/YuE-s1-7B-anneal-en-cot", help="Stage 1 model checkpoint")
parser.add_argument("--stage2_model", type=str, default="../models/YuE-s2-1B-general", help="Stage 2 model checkpoint")
parser.add_argument("--max_new_tokens", type=int, default=3000, help="Max tokens per segment")
parser.add_argument("--repetition_penalty", type=float, default=1.1, help="Repetition penalty (1.0-2.0)")
parser.add_argument("--run_n_segments", type=int, default=2, help="Number of segments to generate")
parser.add_argument("--stage2_batch_size", type=int, default=4, help="Stage 2 batch size")

# Instrumental Conditioning (NEW)
parser.add_argument("--instrumental_path", type=str, required=True, help="Path to instrumental MP3/WAV to condition on")
parser.add_argument("--instrumental_start_time", type=float, default=0.0, help="Start time in seconds")
parser.add_argument("--instrumental_end_time", type=float, default=None, help="End time in seconds (None = full length)")

# Prompt
parser.add_argument("--genre_txt", type=str, required=True, help="Genre tags file")
parser.add_argument("--lyrics_txt", type=str, required=True, help="Lyrics file")

# Output
parser.add_argument("--output_dir", type=str, default="./output", help="Output directory")
parser.add_argument("--keep_intermediate", action="store_true", help="Keep intermediate files")
parser.add_argument("--disable_offload_model", action="store_true", help="Don't offload Stage 1 model")
parser.add_argument("--cuda_idx", type=int, default=0)
parser.add_argument("--seed", type=int, default=42, help="Random seed")

# Codec config
parser.add_argument('--basic_model_config', default='./xcodec_mini_infer/final_ckpt/config.yaml')
parser.add_argument('--resume_path', default='./xcodec_mini_infer/final_ckpt/ckpt_00360000.pth')
parser.add_argument('--config_path', type=str, default='./xcodec_mini_infer/decoders/config.yaml')
parser.add_argument('--vocal_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_131000.pth')
parser.add_argument('--inst_decoder_path', type=str, default='./xcodec_mini_infer/decoders/decoder_151000.pth')
parser.add_argument('-r', '--rescale', action='store_true', help='Rescale output to avoid clipping')


args = parser.parse_args()

if not os.path.exists(args.instrumental_path):
    raise FileNotFoundError(f"Instrumental file not found: {args.instrumental_path}")

stage1_model = args.stage1_model
stage2_model = args.stage2_model
cuda_idx = args.cuda_idx
max_new_tokens = args.max_new_tokens
stage1_output_dir = os.path.join(args.output_dir, "stage1")
stage2_output_dir = os.path.join(args.output_dir, "stage2")
os.makedirs(stage1_output_dir, exist_ok=True)
os.makedirs(stage2_output_dir, exist_ok=True)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(args.seed)

# Load tokenizer and model
device = torch.device(f"cuda:{cuda_idx}" if torch.cuda.is_available() else "cpu")
mmtokenizer = _MMSentencePieceTokenizer("./mm_tokenizer_v0.2_hf/tokenizer.model")

print("Loading Stage 1 model...")
model = AutoModelForCausalLM.from_pretrained(
    stage1_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model.to(device)
model.eval()

if torch.__version__ >= "2.0.0":
    model = torch.compile(model)

# Load codec tools
codectool = CodecManipulator("xcodec", 0, 1)
codectool_stage2 = CodecManipulator("xcodec", 0, 8)
model_config = OmegaConf.load(args.basic_model_config)
codec_model = eval(model_config.generator.name)(**model_config.generator.config).to(device)
parameter_dict = torch.load(args.resume_path, map_location='cpu', weights_only=False)
codec_model.load_state_dict(parameter_dict['codec_model'])
codec_model.to(device)
codec_model.eval()


class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores


def load_audio_mono(filepath, sampling_rate=16000):
    audio, sr = torchaudio.load(filepath)
    audio = torch.mean(audio, dim=0, keepdim=True)
    if sr != sampling_rate:
        resampler = Resample(orig_freq=sr, new_freq=sampling_rate)
        audio = resampler(audio)
    return audio


def encode_audio(codec_model, audio_prompt, device, target_bw=0.5):
    if len(audio_prompt.shape) < 3:
        audio_prompt.unsqueeze_(0)
    with torch.no_grad():
        raw_codes = codec_model.encode(audio_prompt.to(device), target_bw=target_bw)
    raw_codes = raw_codes.transpose(0, 1)
    raw_codes = raw_codes.cpu().numpy().astype(np.int16)
    return raw_codes


def split_lyrics(lyrics):
    pattern = r"\[(\w+)\](.*?)(?=\[|\Z)"
    segments = re.findall(pattern, lyrics, re.DOTALL)
    structured_lyrics = [f"[{seg[0]}]\n{seg[1].strip()}\n\n" for seg in segments]
    return structured_lyrics


# ============================================================================
# LOAD AND ENCODE INSTRUMENTAL
# ============================================================================
print(f"Loading instrumental from: {args.instrumental_path}")
instrumental_audio = load_audio_mono(args.instrumental_path)

# Trim to specified time range
sample_rate = 16000
start_sample = int(args.instrumental_start_time * sample_rate)
if args.instrumental_end_time is not None:
    end_sample = int(args.instrumental_end_time * sample_rate)
else:
    end_sample = instrumental_audio.shape[-1]

instrumental_audio = instrumental_audio[:, start_sample:end_sample]
instrumental_duration = instrumental_audio.shape[-1] / sample_rate
print(f"Instrumental duration: {instrumental_duration:.2f} seconds")

# Encode to codec tokens
print("Encoding instrumental to codec tokens...")
instrumental_codes = encode_audio(codec_model, instrumental_audio, device, target_bw=0.5)
instrumental_ids = codectool.npy2ids(instrumental_codes[0])
print(f"Instrumental tokens: {len(instrumental_ids)}")


# ============================================================================
# LOAD PROMPTS
# ============================================================================
with open(args.genre_txt) as f:
    genres = f.read().strip()
with open(args.lyrics_txt) as f:
    lyrics = split_lyrics(f.read())

full_lyrics = "\n".join(lyrics)
prompt_texts = [f"Generate music from the given lyrics segment by segment.\n[Genre] {genres}\n{full_lyrics}"]
prompt_texts += lyrics

random_id = uuid.uuid4()
stage1_output_set = []

# Sampling params
top_p = 0.93
temperature = 1.0
repetition_penalty = args.repetition_penalty

# Special tokens
start_of_segment = mmtokenizer.tokenize('[start_of_segment]')
end_of_segment = mmtokenizer.tokenize('[end_of_segment]')


# ============================================================================
# STAGE 1: VOCAL GENERATION WITH INSTRUMENTAL TEACHER-FORCING
# ============================================================================
print("\n" + "="*60)
print("STAGE 1: Generating vocals conditioned on instrumental...")
print("="*60)

run_n_segments = min(args.run_n_segments + 1, len(lyrics))
raw_output = None
instrumental_idx = 0  # Track position in instrumental

for i, p in enumerate(tqdm(prompt_texts[:run_n_segments], desc="Stage1 inference...")):
    section_text = p.replace('[start_of_segment]', '').replace('[end_of_segment]', '')
    guidance_scale = 1.5 if i <= 1 else 1.2
    
    if i == 0:
        continue
    
    if i == 1:
        head_id = mmtokenizer.tokenize(prompt_texts[0])
        prompt_ids = head_id + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids
    else:
        prompt_ids = end_of_segment + start_of_segment + mmtokenizer.tokenize(section_text) + [mmtokenizer.soa] + codectool.sep_ids

    prompt_ids = torch.as_tensor(prompt_ids).unsqueeze(0).to(device)
    input_ids = torch.cat([raw_output, prompt_ids], dim=1) if i > 1 else prompt_ids
    
    # Use window slicing if needed
    max_context = 16384 - max_new_tokens - 1
    if input_ids.shape[-1] > max_context:
        print(f'Section {i}: using last {max_context} tokens')
        input_ids = input_ids[:, -(max_context):]
    
    # Calculate how many tokens we can generate for this segment
    # Each frame = 2 tokens (1 vocal + 1 instrumental)
    remaining_instrumental = len(instrumental_ids) - instrumental_idx
    segment_max_tokens = min(max_new_tokens, remaining_instrumental * 2)
    
    if segment_max_tokens <= 0:
        print(f"Warning: No more instrumental tokens available at segment {i}")
        break
    
    # Generate with teacher-forcing for instrumental
    # We generate token by token: predict vocal, then force instrumental
    generated_tokens = []
    current_input = input_ids.clone()
    
    logits_processor = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 32002), 
        BlockTokenRangeProcessor(32016, 32016)
    ])
    
    tokens_generated = 0
    max_segment_frames = segment_max_tokens // 2  # Each frame = vocal + instrumental
    
    with torch.no_grad():
        for frame in tqdm(range(max_segment_frames), desc=f"Section {i} frames", leave=False):
            if instrumental_idx >= len(instrumental_ids):
                break
            
            # Generate ONE vocal token
            output = model.generate(
                input_ids=current_input,
                max_new_tokens=1,
                min_new_tokens=1,
                do_sample=True,
                top_p=top_p,
                temperature=temperature,
                repetition_penalty=repetition_penalty,
                logits_processor=logits_processor,
                guidance_scale=guidance_scale,
                pad_token_id=mmtokenizer.eoa,
            )
            
            vocal_token = output[:, -1:]
            
            # Check if we hit end of audio
            if vocal_token[0, 0].item() == mmtokenizer.eoa:
                generated_tokens.append(vocal_token[0, 0].item())
                break
            
            # Force the instrumental token from our conditioning track
            inst_token = torch.tensor([[instrumental_ids[instrumental_idx]]], device=device)
            instrumental_idx += 1
            
            # Append both tokens
            current_input = torch.cat([current_input, vocal_token, inst_token], dim=1)
            generated_tokens.extend([vocal_token[0, 0].item(), inst_token[0, 0].item()])
            tokens_generated += 2
    
    # Add EOA if not already there
    if len(generated_tokens) > 0 and generated_tokens[-1] != mmtokenizer.eoa:
        generated_tokens.append(mmtokenizer.eoa)
    
    generated_tensor = torch.tensor([generated_tokens], device=device)
    
    if i > 1:
        raw_output = torch.cat([raw_output, prompt_ids, generated_tensor], dim=1)
    else:
        raw_output = torch.cat([prompt_ids, generated_tensor], dim=1)

print(f"\nGenerated {raw_output.shape[-1]} total tokens")
print(f"Used {instrumental_idx} of {len(instrumental_ids)} instrumental tokens")


# ============================================================================
# EXTRACT AND SAVE VOCALS/INSTRUMENTALS
# ============================================================================
ids = raw_output[0].cpu().numpy()
soa_idx = np.where(ids == mmtokenizer.soa)[0].tolist()
eoa_idx = np.where(ids == mmtokenizer.eoa)[0].tolist()

if len(soa_idx) != len(eoa_idx):
    raise ValueError(f'Invalid soa/eoa pairs: {len(soa_idx)} soa, {len(eoa_idx)} eoa')

vocals = []
instrumentals = []

for idx in range(len(soa_idx)):
    codec_ids = ids[soa_idx[idx]+1:eoa_idx[idx]]
    if len(codec_ids) > 0 and codec_ids[0] == 32016:
        codec_ids = codec_ids[1:]
    codec_ids = codec_ids[:2 * (codec_ids.shape[0] // 2)]
    
    if len(codec_ids) == 0:
        continue
    
    vocals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[0])
    vocals.append(vocals_ids)
    instrumentals_ids = codectool.ids2npy(rearrange(codec_ids, "(n b) -> b n", b=2)[1])
    instrumentals.append(instrumentals_ids)

if len(vocals) == 0:
    raise ValueError("No vocals generated!")

vocals = np.concatenate(vocals, axis=1)
instrumentals = np.concatenate(instrumentals, axis=1)

vocal_save_path = os.path.join(stage1_output_dir, f"vocal_only_{random_id}_vtrack.npy")
inst_save_path = os.path.join(stage1_output_dir, f"vocal_only_{random_id}_itrack.npy")
np.save(vocal_save_path, vocals)
np.save(inst_save_path, instrumentals)
stage1_output_set.append(vocal_save_path)
stage1_output_set.append(inst_save_path)

print(f"\nSaved vocals: {vocal_save_path}")
print(f"Saved instrumentals: {inst_save_path}")


# ============================================================================
# STAGE 2: REFINE CODECS
# ============================================================================
if not args.disable_offload_model:
    model.cpu()
    del model
    torch.cuda.empty_cache()

print("\n" + "="*60)
print("STAGE 2: Refining codecs...")
print("="*60)

model_stage2 = AutoModelForCausalLM.from_pretrained(
    stage2_model,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)
model_stage2.to(device)
model_stage2.eval()

if torch.__version__ >= "2.0.0":
    model_stage2 = torch.compile(model_stage2)


def stage2_generate(model, prompt, batch_size=16):
    codec_ids = codectool.unflatten(prompt, n_quantizer=1)
    codec_ids = codectool.offset_tok_ids(
        codec_ids,
        global_offset=codectool.global_offset,
        codebook_size=codectool.codebook_size,
        num_codebooks=codectool.num_codebooks,
    ).astype(np.int32)
    
    if batch_size > 1:
        codec_list = []
        for b in range(batch_size):
            idx_begin = b * 300
            idx_end = (b + 1) * 300
            codec_list.append(codec_ids[:, idx_begin:idx_end])
        codec_ids = np.concatenate(codec_list, axis=0)
        prompt_ids = np.concatenate([
            np.tile([mmtokenizer.soa, mmtokenizer.stage_1], (batch_size, 1)),
            codec_ids,
            np.tile([mmtokenizer.stage_2], (batch_size, 1)),
        ], axis=1)
    else:
        prompt_ids = np.concatenate([
            np.array([mmtokenizer.soa, mmtokenizer.stage_1]),
            codec_ids.flatten(),
            np.array([mmtokenizer.stage_2])
        ]).astype(np.int32)
        prompt_ids = prompt_ids[np.newaxis, ...]

    codec_ids = torch.as_tensor(codec_ids).to(device)
    prompt_ids = torch.as_tensor(prompt_ids).to(device)
    len_prompt = prompt_ids.shape[-1]
    
    block_list = LogitsProcessorList([
        BlockTokenRangeProcessor(0, 46358),
        BlockTokenRangeProcessor(53526, mmtokenizer.vocab_size)
    ])

    for frames_idx in range(codec_ids.shape[1]):
        cb0 = codec_ids[:, frames_idx:frames_idx+1]
        prompt_ids = torch.cat([prompt_ids, cb0], dim=1)
        
        with torch.no_grad():
            stage2_output = model.generate(
                input_ids=prompt_ids,
                min_new_tokens=7,
                max_new_tokens=7,
                eos_token_id=mmtokenizer.eoa,
                pad_token_id=mmtokenizer.eoa,
                logits_processor=block_list,
            )
        prompt_ids = stage2_output

    if batch_size > 1:
        output = prompt_ids.cpu().numpy()[:, len_prompt:]
        output_list = [output[b] for b in range(batch_size)]
        output = np.concatenate(output_list, axis=0)
    else:
        output = prompt_ids[0].cpu().numpy()[len_prompt:]

    return output


def stage2_inference(model, stage1_output_set, stage2_output_dir, batch_size=4):
    stage2_result = []
    for idx in tqdm(range(len(stage1_output_set))):
        output_filename = os.path.join(stage2_output_dir, os.path.basename(stage1_output_set[idx]))
        
        if os.path.exists(output_filename):
            print(f'{output_filename} already done.')
            stage2_result.append(output_filename)
            continue
        
        prompt = np.load(stage1_output_set[idx]).astype(np.int32)
        output_duration = prompt.shape[-1] // 50 // 6 * 6
        num_batch = output_duration // 6
        
        if num_batch <= batch_size:
            output = stage2_generate(model, prompt[:, :output_duration*50], batch_size=num_batch)
        else:
            segments = []
            num_segments = (num_batch // batch_size) + (1 if num_batch % batch_size != 0 else 0)
            for seg in range(num_segments):
                start_idx = seg * batch_size * 300
                end_idx = min((seg + 1) * batch_size * 300, output_duration*50)
                current_batch_size = batch_size if seg != num_segments-1 or num_batch % batch_size == 0 else num_batch % batch_size
                segment = stage2_generate(model, prompt[:, start_idx:end_idx], batch_size=current_batch_size)
                segments.append(segment)
            output = np.concatenate(segments, axis=0)
        
        if output_duration*50 != prompt.shape[-1]:
            ending = stage2_generate(model, prompt[:, output_duration*50:], batch_size=1)
            output = np.concatenate([output, ending], axis=0)
        
        output = codectool_stage2.ids2npy(output)
        
        # Fix invalid codes
        fixed_output = copy.deepcopy(output)
        for line_idx, line in enumerate(output):
            for j, element in enumerate(line):
                if element < 0 or element > 1023:
                    counter = Counter(line)
                    most_frequent = sorted(counter.items(), key=lambda x: x[1], reverse=True)[0][0]
                    fixed_output[line_idx, j] = most_frequent
        
        np.save(output_filename, fixed_output)
        stage2_result.append(output_filename)
    
    return stage2_result


stage2_result = stage2_inference(model_stage2, stage1_output_set, stage2_output_dir, batch_size=args.stage2_batch_size)
print('Stage 2 DONE.\n')


# ============================================================================
# AUDIO RECONSTRUCTION
# ============================================================================
def save_audio(wav: torch.Tensor, path, sample_rate: int, rescale: bool = False):
    folder_path = os.path.dirname(path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    limit = 0.99
    max_val = wav.abs().max()
    wav = wav * min(limit / max_val, 1) if rescale else wav.clamp(-limit, limit)
    torchaudio.save(str(path), wav, sample_rate=sample_rate, encoding='PCM_S', bits_per_sample=16)


recons_output_dir = os.path.join(args.output_dir, "recons")
recons_mix_dir = os.path.join(recons_output_dir, 'mix')
os.makedirs(recons_mix_dir, exist_ok=True)

tracks = []
for npy in stage2_result:
    codec_result = np.load(npy)
    with torch.no_grad():
        decoded_waveform = codec_model.decode(
            torch.as_tensor(codec_result.astype(np.int16), dtype=torch.long)
            .unsqueeze(0).permute(1, 0, 2).to(device)
        )
    decoded_waveform = decoded_waveform.cpu().squeeze(0)
    save_path = os.path.join(recons_output_dir, os.path.splitext(os.path.basename(npy))[0] + ".mp3")
    tracks.append(save_path)
    save_audio(decoded_waveform, save_path, 16000)

# Mix tracks
for inst_path in tracks:
    try:
        if '_itrack' in inst_path:
            vocal_path = inst_path.replace('_itrack', '_vtrack')
            if not os.path.exists(vocal_path):
                continue
            recons_mix = os.path.join(recons_mix_dir, os.path.basename(inst_path).replace('_itrack', '_mixed'))
            vocal_stem, sr = sf.read(inst_path)
            instrumental_stem, _ = sf.read(vocal_path)
            mix_stem = (vocal_stem + instrumental_stem)
            sf.write(recons_mix, mix_stem, sr)
    except Exception as e:
        print(f"Mix error: {e}")

# Vocoder
vocal_decoder, inst_decoder = build_codec_model(args.config_path, args.vocal_decoder_path, args.inst_decoder_path)
vocoder_output_dir = os.path.join(args.output_dir, 'vocoder')
vocoder_stems_dir = os.path.join(vocoder_output_dir, 'stems')
vocoder_mix_dir = os.path.join(vocoder_output_dir, 'mix')
os.makedirs(vocoder_mix_dir, exist_ok=True)
os.makedirs(vocoder_stems_dir, exist_ok=True)

for npy in stage2_result:
    if '_itrack' in npy:
        instrumental_output = process_audio(npy, os.path.join(vocoder_stems_dir, 'itrack.mp3'), args.rescale, args, inst_decoder, codec_model)
    else:
        vocal_output = process_audio(npy, os.path.join(vocoder_stems_dir, 'vtrack.mp3'), args.rescale, args, vocal_decoder, codec_model)

try:
    mix_output = instrumental_output + vocal_output
    vocoder_mix = os.path.join(vocoder_mix_dir, os.path.basename(recons_mix))
    save_audio(mix_output, vocoder_mix, 44100, args.rescale)
    print(f"Created mix: {vocoder_mix}")
except Exception as e:
    print(f"Final mix error: {e}")

# Post-process
try:
    replace_low_freq_with_energy_matched(
        a_file=recons_mix,
        b_file=vocoder_mix,
        c_file=os.path.join(args.output_dir, os.path.basename(recons_mix)),
        cutoff_freq=5500.0
    )
except Exception as e:
    print(f"Post-process error: {e}")

print("\n" + "="*60)
print("VOCAL-ONLY GENERATION COMPLETE!")
print("="*60)
print(f"Output directory: {args.output_dir}")
print(f"Generated vocals conditioned on: {args.instrumental_path}")
