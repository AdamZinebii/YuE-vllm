# YuE Inference Scripts

This directory contains inference scripts for the YuE music generation model.

## Available Scripts

| Script | Backend | Description |
|--------|---------|-------------|
| `infer.py` | HuggingFace Transformers | Original inference script with Classifier-Free Guidance support |
| `infer_vllm.py` | vLLM | High-throughput inference using vLLM engine |

---

## Comparison: `infer.py` vs `infer_vllm.py`

### Key Differences

| Feature | `infer.py` (Transformers) | `infer_vllm.py` (vLLM) |
|---------|---------------------------|------------------------|
| **Model Loading** | `AutoModelForCausalLM.from_pretrained()` | `vllm.LLM()` |
| **Generation API** | `model.generate()` with kwargs | `model.generate()` with `SamplingParams` |
| **Guidance Scale (CFG)** | ✅ Supported (1.5 for first segment, 1.2 for rest) | ❌ Not supported |
| **Logits Processor** | `LogitsProcessor` class inheritance | Per-request callable function |
| **Torch Compile** | `torch.compile(model)` for PyTorch 2.0+ | Built-in optimizations |
| **Memory Management** | Manual `.to(device)` and `.cpu()` | Automatic via `gpu_memory_utilization` |
| **Inference Speed** | Standard | Faster due to PagedAttention & continuous batching |

### When to Use Each

#### Use `infer.py` (Transformers) when:
- You need **Classifier-Free Guidance** for better prompt following
- You want the original, tested behavior
- You're using a GPU with limited memory and need FlashAttention 2

#### Use `infer_vllm.py` (vLLM) when:
- You prioritize **inference speed** over CFG support
- You're running batch inference workloads
- You want to leverage vLLM's optimizations (PagedAttention, continuous batching)

---

## Setup (Required)

Before running either inference script, you **must** clone the xcodec model from HuggingFace:

```bash
cd YuE/inference/
git lfs install
git clone https://huggingface.co/m-a-p/xcodec_mini_infer
```

> **Note:** Make sure you have `git-lfs` installed. If not:
> ```bash
> # Ubuntu/Debian
> apt install git-lfs
> 
> # Or without root (see https://github.com/git-lfs/git-lfs/issues/4134#issuecomment-1635204943)
> ```

---

## Usage

### Original (Transformers)

```bash
python infer.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

### vLLM Version

```bash
python infer_vllm.py \
    --cuda_idx 0 \
    --stage1_model m-a-p/YuE-s1-7B-anneal-en-cot \
    --stage2_model m-a-p/YuE-s2-1B-general \
    --genre_txt ../prompt_egs/genre.txt \
    --lyrics_txt ../prompt_egs/lyrics.txt \
    --run_n_segments 2 \
    --stage2_batch_size 4 \
    --output_dir ../output \
    --max_new_tokens 3000 \
    --repetition_penalty 1.1
```

---

## Dependencies

### For `infer.py`
```bash
pip install torch transformers flash-attn --no-build-isolation
pip install omegaconf torchaudio einops sentencepiece soundfile
```

### For `infer_vllm.py`
```bash
pip install vllm
pip install omegaconf torchaudio einops sentencepiece soundfile
```

---

## Technical Details

### Logits Processor Implementation

Both scripts block certain token ranges during generation to ensure valid audio tokens are produced.

**Transformers version (`infer.py`):**
```python
class BlockTokenRangeProcessor(LogitsProcessor):
    def __init__(self, start_id, end_id):
        self.blocked_token_ids = list(range(start_id, end_id))

    def __call__(self, input_ids, scores):
        scores[:, self.blocked_token_ids] = -float("inf")
        return scores
```

**vLLM version (`infer_vllm.py`):**
```python
def block_tokens_processor(token_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
    logits[0:32002] = float("-inf")
    logits[32016] = float("-inf")
    return logits
```

### Guidance Scale (CFG)

The original `infer.py` uses Classifier-Free Guidance with:
- `guidance_scale=1.5` for the first segment
- `guidance_scale=1.2` for subsequent segments

This helps the model follow the genre/lyrics prompts more closely. The vLLM version does not support this feature, which may result in:
- More diverse but potentially less prompt-adherent outputs
- Slightly different musical characteristics

---

## Output Structure

Both scripts produce the same output structure:

```
output/
├── stage1/           # Stage 1 intermediate .npy files
├── stage2/           # Stage 2 refined .npy files  
├── recons/           # Reconstructed audio at 16kHz
│   └── mix/          # Mixed vocal + instrumental
└── vocoder/          # Upsampled audio at 44.1kHz
    ├── stems/        # Separate vocal and instrumental
    └── mix/          # Final mixed output
```
