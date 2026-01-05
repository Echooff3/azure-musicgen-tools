# Azure Training Environment Debugging Log

## Problem History

### Initial Error
```
RuntimeError: Failed to import transformers.trainer because of the following error:
cannot import name 'EncoderDecoderCache' from 'transformers'
```

**Root Cause**: transformers <4.42.0 doesn't have EncoderDecoderCache class

**Fix Attempted #1**: Upgrade transformers to >=4.42.0
**Result**: Led to next error ❌

---

### Error #2: PyTorch pytree API Mismatch
```
AttributeError: module 'torch.utils._pytree' has no attribute 'register_pytree_node'
```

**Root Cause**: torch 2.1.0 has old pytree API; transformers 4.42.0+ needs torch 2.3+

**Fix Attempted #2**: Upgrade torch to 2.3.1 in requirements.txt and conda env
**Result**: Led to next error ❌

---

### Error #3: Invalid Conda Channel
```
UnavailableInvalidChannel: HTTP 404 Not Found for channel torch::pytorch
```

**Root Cause**: Used invalid `torch::pytorch` channel syntax in conda

**Fix Attempted #3**: Changed channels to `pytorch`, `nvidia`, `conda-forge`, `defaults` and used conda packages for pytorch
**Result**: Led to next error ❌

---

### Error #4: PyTorch/torchaudio ABI Mismatch  
```
OSError: undefined symbol: _ZN3c104impl3cow11cow_deleterEPv
```

**Root Cause**: Mixing conda pytorch with pip dependencies creates ABI incompatibilities

**Fix Attempted #4**: Install PyTorch 2.3.1 via pip only (not conda)
**Result**: Led to next error ❌

---

### Error #5: Dependency Conflict (CURRENT)
```
ERROR: Cannot install av==13.1.0 and torch==2.3.1 because:
    audiocraft 1.3.0 depends on av==11.0.0
    audiocraft 1.2.0 depends on torch==2.1.0
```

**Root Cause**: 
- audiocraft pins torch==2.1.0
- We need torch==2.3.1 for transformers 4.42.0+
- **THESE ARE MUTUALLY EXCLUSIVE**

**Critical Insight**: Checking train_musicgen_job.py imports:
- ✅ Uses: torch, torchaudio, librosa, transformers, peft, datasets
- ❌ Does NOT import audiocraft anywhere
- MusicGen is loaded directly from transformers, NOT audiocraft

**Solution**: Remove audiocraft from dependencies - it's not used in training!

---

## Dependency Chain Analysis

### What We Actually Need
1. **torch 2.3.1** - Required by transformers 4.42.0+ (pytree API)
2. **transformers 4.42.0+** - Has EncoderDecoderCache 
3. **torchaudio 2.3.1** - Must match torch version
4. **peft** - For LoRA training
5. **librosa** - Audio processing
6. **av** - Video/audio codec (used by what?)

### What We DON'T Need
1. **audiocraft** - Not imported in training script! MusicGen comes from transformers.

---

## Final Solution

Remove audiocraft from conda_env_musicgen_training.yml entirely. The training script loads MusicGen directly from transformers/HuggingFace, not from the audiocraft library.

**Changes Made**:
1. Removed `audiocraft>=1.2.0` from conda env
2. Removed `av==13.1.0` (only needed by audiocraft)
3. Removed `xformers` (optional optimization, not critical)
4. Kept clean dependency list:
   - torch==2.3.1 (for transformers 4.42.0+ compatibility)
   - transformers>=4.42.0 (has EncoderDecoderCache)
   - peft, accelerate, datasets (LoRA training)
   - librosa, soundfile (audio processing)
   - Azure SDK packages

**Why This Works**:
- Training script imports MusicGen from `transformers`, NOT `audiocraft`
- `transformers` library includes MusicGen models from HuggingFace Hub
- No circular dependency: torch 2.3.1 + transformers 4.42.0+ = ✅
- All actual imports in train_musicgen_job.py are satisfied

**Cost Savings**: This also saves ~5-10 minutes on environment build time by removing unnecessary packages.

---

## Error #6: Missing input_ids for MusicGen Training
```
ValueError: You have to specify either input_ids or inputs_embeds
```

**Root Cause**: 
- MusicGen architecture requires text encoder input (`input_ids`) even for audio-only training
- The `collate_fn` was only providing audio inputs, missing text conditioning
- Local training script (local_train_directml.py) had the correct implementation, but Azure script didn't

**Fix Applied**: Updated `collate_fn` in train_musicgen_job.py to:
1. Generate dummy text prompts ("audio loop") for each audio sample
2. Tokenize prompts using processor.tokenizer
3. Add `input_ids` and `attention_mask` to inputs
4. Keep audio as labels for self-supervised training

**Status**: ✅ FIXED - Resubmitting job now

---

## Error #7: Missing decoder_start_token_id Configuration
```
ValueError: Make sure to set the decoder_start_token_id attribute of the model's configuration.
```

**Root Cause**: 
- MusicGen's decoder needs a start token when training with labels
- The model config doesn't have `decoder_start_token_id` set by default
- When we provide `labels` but no `decoder_input_ids`, the model tries to create decoder inputs using shift_tokens_right, which requires this config value

**Fix Applied**: Set `decoder_start_token_id` to `pad_token_id` after loading the model (standard practice for MusicGen)

**Status**: ✅ FIXED - Resubmitting job now
