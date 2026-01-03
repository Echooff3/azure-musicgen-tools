# GPU SKU Reference for MusicGen Training

> **Generated**: January 3, 2026  
> **Region**: East US  
> **For**: MusicGen model training with LoRA

This guide shows all available GPU SKUs in your region and which ones to use for MusicGen training.

## 🎯 Quick Recommendation

**For your config** ([azuredeploy.parameters.json](azuredeploy.parameters.json)):

```json
"gpuComputeVmSize": {
  "value": "Standard_NC4as_T4_v3"  // ⭐ RECOMMENDED: $0.526/hr
}
```

**Why**: Budget-friendly T4 GPU, perfectly adequate for MusicGen-small with LoRA training.

---

## 📊 Available GPU SKUs in East US

| SKU | vCPUs | RAM (GB) | GPUs | GPU Type | Cost/hr | Training Time* | Total Cost* |
|-----|-------|----------|------|----------|---------|----------------|-------------|
| **Standard_NC4as_T4_v3** ⭐ | 4 | 28 | 1 | T4 | **$0.526** | 8-10 hrs | **$4-5** |
| **Standard_NC6s_v3** | 6 | 112 | 1 | V100 | $3.06 | 2-3 hrs | $6-9 |
| **Standard_NC8as_T4_v3** | 8 | 56 | 1 | T4 | $0.702 | 8-10 hrs | $5-7 |
| **Standard_NC12s_v3** | 12 | 224 | 2 | V100 | $6.12 | 1.5-2 hrs | $9-12 |
| **Standard_NC16as_T4_v3** | 16 | 110 | 1 | T4 | $1.404 | 8-10 hrs | $11-14 |
| **Standard_NC24s_v3** | 24 | 448 | 4 | V100 | $12.24 | 1-1.5 hrs | $12-18 |
| **Standard_NC24rs_v3** | 24 | 448 | 4 | V100 | $13.46 | 1-1.5 hrs | $13-20 |
| **Standard_NC24ads_A100_v4** | 24 | 220 | 1 | A100 | $3.67 | 1.5-2 hrs | $5-7 |
| **Standard_NC48ads_A100_v4** | 48 | 440 | 2 | A100 | $7.34 | 1-1.5 hrs | $10-15 |
| **Standard_NC64as_T4_v3** | 64 | 440 | 4 | T4 | $3.507 | 2-3 hrs** | $7-10 |
| **Standard_NC96ads_A100_v4** | 96 | 880 | 4 | A100 | $14.69 | 0.5-1 hr** | $7-15 |
| **Standard_NC40ads_H100_v5** | 40 | 320 | 1 | H100 | ~$4.50 | 1-1.5 hrs | $6-9 |
| **Standard_NC80adis_H100_v5** | 80 | 640 | 2 | H100 | ~$9.00 | 0.5-1 hr** | $4-9 |

\* Based on your config: 10 epochs, batch size 4, MusicGen-small model  
\*\* Requires distributed training code to use multiple GPUs efficiently

---

## ✅ Recommended Options by Use Case

### 1️⃣ Budget / Learning / Testing
**Use**: `Standard_NC4as_T4_v3`

```json
"gpuComputeVmSize": {"value": "Standard_NC4as_T4_v3"},
"gpuComputeMaxNodes": {"value": 2}
```

- **Cost**: $0.526/hr = ~$4-5 per training run
- **Quota needed**: "Standard NCasT4_v3 Family vCPUs" = 4
- **Training time**: 8-10 hours
- **Best for**: First-time training, experimentation, limited budget

### 2️⃣ Balanced Performance
**Use**: `Standard_NC6s_v3`

```json
"gpuComputeVmSize": {"value": "Standard_NC6s_v3"},
"gpuComputeMaxNodes": {"value": 2}
```

- **Cost**: $3.06/hr = ~$6-9 per training run
- **Quota needed**: "Standard NCSv3 Family vCPUs" = 6
- **Training time**: 2-3 hours
- **Best for**: Production training, regular iteration

### 3️⃣ Speed Matters
**Use**: `Standard_NC12s_v3` (2x V100 GPUs)

```json
"gpuComputeVmSize": {"value": "Standard_NC12s_v3"},
"gpuComputeMaxNodes": {"value": 2}
```

- **Cost**: $6.12/hr = ~$9-12 per training run
- **Quota needed**: "Standard NCSv3 Family vCPUs" = 12
- **Training time**: 1.5-2 hours
- **Best for**: Rapid prototyping, multiple training runs per day

### 4️⃣ Maximum Performance
**Use**: `Standard_NC24ads_A100_v4` (A100 GPU)

```json
"gpuComputeVmSize": {"value": "Standard_NC24ads_A100_v4"},
"gpuComputeMaxNodes": {"value": 1}
```

- **Cost**: $3.67/hr = ~$5-7 per training run
- **Quota needed**: "Standard NCadsA100v4 Family vCPUs" = 24
- **Training time**: 1.5-2 hours
- **Best for**: Large-scale production, enterprise use

---

## ❌ Not Recommended (Wasteful for MusicGen)

| SKU | Why Not Use |
|-----|-------------|
| Standard_NC16as_T4_v3 | Same T4 GPU as NC4as, but 3x the cost - just paying for extra CPUs |
| Standard_NC24s_v3 | 4 GPUs require distributed training code - complex setup |
| Standard_NC64as_T4_v3 | 4 T4s need multi-GPU code - better to use 1 V100 |
| Standard_NC96ads_A100_v4 | $14.69/hr is extreme overkill for MusicGen-small |
| Standard_NC80adis_H100_v5 | H100s are for massive models (100B+ params), not 300M |

---

## 💰 Cost Comparison Table

**Full training run (10 epochs, 4 batch size, MusicGen-small):**

| SKU | Hourly Rate | Training Hours | Total Cost | Cost vs T4 |
|-----|-------------|----------------|------------|------------|
| NC4as_T4_v3 ⭐ | $0.53 | 8-10 | **$4-5** | Baseline |
| NC8as_T4_v3 | $0.70 | 8-10 | $5-7 | +40% cost, same speed |
| NC6s_v3 | $3.06 | 2-3 | $6-9 | +50% cost, 3-4x faster |
| NC12s_v3 | $6.12 | 1.5-2 | $9-12 | +2x cost, 5-6x faster |
| NC24ads_A100_v4 | $3.67 | 1.5-2 | $5-7 | +25% cost, 4-5x faster |

**Winner for value**: `Standard_NC4as_T4_v3` (lowest total cost)  
**Winner for speed**: `Standard_NC12s_v3` or A100 SKUs

---

## 🔄 How to Change GPU SKU

### Step 1: Edit parameters file

Edit [azuredeploy.parameters.json](azuredeploy.parameters.json) line 24:

```json
"gpuComputeVmSize": {
  "value": "Standard_NC4as_T4_v3"  // Change this value
}
```

### Step 2: Request appropriate quota

```bash
# Check current quota in East US
az vm list-usage --location eastus --query "[?contains(name.localizedValue, 'NC')]" -o table
```

**Quota families by SKU:**
- NC4/8/16/64as_T4_v3 → "Standard NCasT4_v3 Family vCPUs"
- NC6/12/24s_v3 → "Standard NCSv3 Family vCPUs"
- NC24/48/96ads_A100_v4 → "Standard NCadsA100v4 Family vCPUs"
- NC40/80ads_H100_v5 → "Standard NCadsH100v5 Family vCPUs"

### Step 3: Deploy (after quota approval)

```bash
cd arm-templates

# Update deployGpuCompute to true in parameters file
# Then deploy:
./deploy.sh  # Linux/Mac
# or
deploy.bat   # Windows
```

---

## 🎓 GPU Architecture Guide

### T4 (Turing - 2018)
- **Best for**: Budget training, inference
- **Memory**: 16 GB
- **Compute**: 8.1 TFLOPS (FP32), 65 TFLOPS (FP16)
- **When to use**: Small models, limited budget, learning

### V100 (Volta - 2017)
- **Best for**: Production training, balanced performance
- **Memory**: 16 GB (32 GB on some SKUs)
- **Compute**: 15.7 TFLOPS (FP32), 125 TFLOPS (FP16)
- **When to use**: Regular training runs, production workloads

### A100 (Ampere - 2020)
- **Best for**: Large-scale training, multi-GPU
- **Memory**: 40 GB (80 GB on some SKUs)
- **Compute**: 19.5 TFLOPS (FP32), 312 TFLOPS (FP16)
- **When to use**: Large models, high-throughput training

### H100 (Hopper - 2022)
- **Best for**: Cutting-edge research, massive models
- **Memory**: 80 GB
- **Compute**: 60 TFLOPS (FP32), 1000 TFLOPS (FP16 with sparsity)
- **When to use**: Frontier research, 100B+ parameter models

**For MusicGen-small (300M params)**: T4 or V100 is perfectly adequate.

---

## 💡 Pro Tips

### Use Spot Instances for 90% Discount
After deploying regular compute, create a spot cluster:

```bash
az ml compute create \
    --name gpu-cluster-spot \
    --type AmlCompute \
    --size Standard_NC4as_T4_v3 \
    --tier spot \
    --min-instances 0 \
    --max-instances 2 \
    --resource-group rg-mg3 \
    --workspace-name musicgen-ml-workspace
```

**Spot pricing:**
- NC4as_T4_v3: $0.053/hr (instead of $0.526/hr)
- NC6s_v3: $0.306/hr (instead of $3.06/hr)

**Trade-off**: Can be evicted when Azure needs capacity.

### Auto-Scale to Zero
Always keep `minNodes: 0` in your config:

```json
"gpuComputeMinNodes": {"value": 0},
"gpuComputeMaxNodes": {"value": 2}
```

This ensures:
- Cluster scales to 0 when idle (no cost)
- Auto-starts when job submitted
- Scales down after 120 seconds idle

### Monitor Costs
Set budget alerts:

```bash
# Create budget (example: $100/month)
az consumption budget create \
    --budget-name "musicgen-monthly" \
    --amount 100 \
    --time-grain Monthly \
    --start-date 2026-01-01 \
    --end-date 2027-01-01 \
    --resource-group rg-mg3
```

---

## 📝 Summary

**My recommendation for your setup:**

1. **Start with**: `Standard_NC4as_T4_v3` 
   - Cheapest option ($4-5 per training run)
   - Easiest quota to get approved
   - Good enough for MusicGen-small

2. **If speed matters**: Upgrade to `Standard_NC6s_v3`
   - 3-4x faster training
   - Still reasonable cost ($6-9 per run)

3. **Don't bother with**: Multi-GPU SKUs unless you modify training code for distributed training

4. **Enable**: Auto-scale to 0 (already in your config ✅)

5. **Consider**: Spot instances for 90% savings (with eviction risk)

---

## 🔗 Related Documentation

- [ARM Templates README](README.md) - Deployment guide
- [GPU Setup Guide](README.md#gpu-compute-configuration) - Detailed quota request instructions
- [Main README](../README.md) - Full project documentation
- [Drum Training Guide](../DRUM_TRAINING_GUIDE.md) - MusicGen training specifics

---

**Questions?** Check the main README troubleshooting section.
