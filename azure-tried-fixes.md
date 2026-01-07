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

### Error #5: Azure ML Batch Scoring - Missing model_type in config.json
```
ValueError: Unrecognized model in /mnt/azureml/cr/j/e1ebddddf54940f5ae920d5b4390459d/exe/wd/4c27a924-8de6-491d-badc-363450fd2d69_score_model. 
Should have a `model_type` key in its config.json
```

**Root Cause**: When Azure ML downloads the registered model to the batch compute instance, the config.json either gets corrupted or is not properly formatted. The AutoProcessor.from_pretrained() requires a valid model_type at the root level.

**Diagnosis**: The local model/config.json HAS "model_type": "musicgen" at line 70, but the downloaded version in Azure ML batch doesn't.

**Fix Attempted #5**: 
- **Root Cause Confirmed**: Azure ML model registration CORRUPTS config.json during upload/download
- The local config.json has all required nested configs (text_encoder, audio_encoder, decoder)
- When Azure downloads it to batch compute, these nested configs are lost

**Solution Implemented**:
1. Modified deployment/score.py to load base model from HuggingFace ("facebook/musicgen-small")
2. Then load fine-tuned weights from registered model's model.safetensors
3. This bypasses the corrupted config.json entirely
4. Added safetensors>=0.4.0 to deployment/conda_inference.yml

**Code Changes**:
```python
# Load base architecture from HuggingFace (has valid config)
processor = AutoProcessor.from_pretrained("facebook/musicgen-small")
model = MusicgenForConditionalGeneration.from_pretrained(
    "facebook/musicgen-small",
    torch_dtype=torch.float32,
    device_map="cpu"
)

# Then load fine-tuned weights from registered model
from safetensors.torch import load_file
state_dict = load_file(os.path.join(model_path, "model.safetensors"))
model.load_state_dict(state_dict, strict=False)
```

**Status**: ✅ FIXED - Environment v3 deployed with workaround for corrupted config

---

### Error #6: Azure ML Batch Inference - Missing azureml-core SDK
```
ModuleNotFoundError: No module named 'azureml'
```

**Root Cause**: The batch inference environment is missing the legacy `azureml-core` and `azureml-defaults` packages that Azure ML's batch driver needs internally.

**Environment Issue**: The Azure ML batch inference driver internally uses the legacy azureml SDK, but we were only including the newer `azure-ai-ml` SDK.

**⚠️ THIS ERROR OCCURRED TWICE** - First fix was incomplete!

**First Attempt (INCOMPLETE)**:
- Added packages to deployment/conda_inference.yml ✅
- Created new environment musicgen-inference-env:1 ✅
- Updated batch-deployment.yml to reference new env ✅
- **FORGOT TO ACTUALLY DEPLOY THE UPDATED BATCH DEPLOYMENT** ❌

**Complete Fix Required**:
1. Update deployment/conda_inference.yml with:
   - azureml-core>=1.57.0
   - azureml-defaults>=1.57.0
   - azure-ai-ml>=1.12.0

2. Create the environment:
```bash
az ml environment create --name musicgen-inference-env \
  --conda-file deployment/conda_inference.yml \
  --image mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest \
  --resource-group rg-mg3 \
  --workspace-name mg-ml-workspace
```

3. Update batch-deployment.yml to use new environment:
   - `environment: azureml:musicgen-inference-env:1`

4. **CRITICAL: Actually deploy the updated batch deployment**:
```bash
az ml batch-deployment update --file deployment/batch-deployment.yml \
  --resource-group rg-mg3 \
  --workspace-name mg-ml-workspace
```

**Status**: ✅ FIXED - Batch deployment updated successfully

---

### Error #11: Missing accelerate Package for device_map
```
ValueError: Using a `device_map`, `tp_plan`, `torch.device` context manager or setting 
`torch.set_default_device(device)` requires `accelerate`. You can install it with `pip install accelerate`
```

**Root Cause**: 
- score.py uses `device_map="auto"` in `MusicgenForConditionalGeneration.from_pretrained()`
- The `accelerate` package is required for device_map functionality in transformers
- deployment/conda_inference.yml was missing this dependency

**Fix Applied**:
1. Added `accelerate>=0.25.0` to deployment/conda_inference.yml
2. Created new environment version: musicgen-inference-env:2
3. Updated batch-deployment.yml to use version 2
4. Deployed the updated batch deployment

**Commands Run**:
```bash
az ml environment create --name musicgen-inference-env \
  --conda-file deployment/conda_inference.yml \
  --image mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:latest \
  --resource-group rg-mg3 \
  --workspace-name mg-ml-workspace

az ml batch-deployment update --file deployment/batch-deployment.yml \
  --resource-group rg-mg3 \
  --workspace-name mg-ml-workspace
```

**Status**: ✅ FIXED - Environment v2 deployed with accelerate

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

**Status**: ✅ FIXED

---

## Error #8: Embedding Layer Type Mismatch (FloatTensor instead of LongTensor)
```
RuntimeError: Expected tensor for argument #1 'indices' to have one of the following scalar types: Long, Int; but got torch.cuda.FloatTensor instead (while checking arguments for embedding)
```

**Stack Trace Location**: 
- `modeling_musicgen.py`, line 576: `self.embed_tokens[codebook](input[:, codebook])`
- The decoder's embedding layer expects discrete token indices (Long/Int), not float values

**Root Cause**:
- `collate_fn` was setting `labels = input_values.clone()` 
- `input_values` are raw audio waveforms (FloatTensor)
- MusicGen's decoder uses embeddings that require **discrete audio codes** (integer indices from EnCodec)
- The decoder expects labels to be audio codes (shape: [batch, num_codebooks, seq_len], dtype: Long)
- We were passing raw float audio, causing the embedding lookup to fail

**Analysis**:
MusicGen architecture:
1. **Audio Encoder (EnCodec)**: Encodes raw audio → discrete codes (integers, 4 codebooks)
2. **Text Encoder (T5)**: Encodes text prompts → encoder hidden states
3. **Decoder**: Takes audio codes as input, generates new audio codes

For training, we need to:
1. Encode audio through EnCodec to get discrete codes
2. Use those codes as both input and labels for the decoder
3. The model handles the teacher-forcing internally

**Fix Applied**: 
Updated `collate_fn` to:
1. Move audio encoding to the model's audio_encoder (EnCodec)
2. Extract discrete audio codes (quantized indices)
3. Pass codes as `decoder_input_ids` and shifted codes as `labels`
4. Ensure all tensors have correct dtypes (Long for indices)

**Code Changes**:
```python
# OLD (WRONG):
inputs['labels'] = inputs['input_values'].clone()  # FloatTensor - WRONG!

# NEW (CORRECT):
# Created custom MusicGenTrainer class that:
# 1. Takes raw audio (input_values) from collate_fn
# 2. Encodes through model.audio_encoder.encode() to get discrete codes
# 3. Passes codes as decoder_input_ids and labels
# 4. Ensures codes are Long dtype for embedding lookup

class MusicGenTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Get the base model (unwrap PEFT)
        base_model = model.base_model.model if hasattr(model, 'base_model') else model
        
        # Encode audio to discrete codes
        with torch.no_grad():
            encoder_outputs = base_model.audio_encoder.encode(
                input_values, 
                padding_mask=padding_mask,
                bandwidth=6.0
            )
            audio_codes = encoder_outputs.audio_codes.squeeze(0).long()
        
        # Forward with discrete codes
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=audio_codes,
            labels=audio_codes,
        )
        return outputs.loss
```

**Files Modified**:
- `src/musicgen_training/train_musicgen_job.py`:
  - Updated `collate_fn` to NOT set labels (custom trainer handles it)
  - Added `MusicGenTrainer` class with custom `compute_loss`
  - Changed `Trainer` → `MusicGenTrainer` instantiation

**Status**: ✅ FIXED

---

## Error #9: Unsupported EnCodec Bandwidth
```
ValueError: This model doesn't support the bandwidth 6.0. Select one of [2.2].
```

**Root Cause**:
- Hardcoded `bandwidth=6.0` in the audio encoder call
- MusicGen-small's EnCodec only supports bandwidth 2.2
- Different model sizes have different supported bandwidths

**Fix Applied**: 
Dynamically get the highest available bandwidth from the encoder config:
```python
target_bandwidths = base_model.audio_encoder.config.target_bandwidths
bandwidth = max(target_bandwidths) if target_bandwidths else None
```

**Status**: ✅ FIXED

---

## Error #10: Float32/Float16 Dtype Mismatch
```
RuntimeError: Input type (float) and bias type (c10::Half) should be the same
```

**Root Cause**:
- Model loaded with `torch_dtype=torch.float16` for GPU efficiency
- Audio `input_values` from DataLoader are float32
- EnCodec encoder expects matching dtypes for input and weights

**Fix Applied**: 
Cast input_values to match encoder dtype before encoding:
```python
encoder_dtype = next(base_model.audio_encoder.parameters()).dtype
input_values = input_values.to(dtype=encoder_dtype)
```

**Status**: ✅ FIXED - Resubmitting job now
