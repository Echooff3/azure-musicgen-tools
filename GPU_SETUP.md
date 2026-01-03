# GPU Setup Guide

This guide helps you set up GPU compute for MusicGen model training.

## Why is GPU Optional?

**GPU compute is disabled by default** in the ARM template deployment because:
- Most Azure subscriptions start with **zero GPU quota**
- GPU SKUs may not be available in all regions  
- This prevents deployment failures for new users
- You can add GPU compute later when you're ready to train models

## Do I Need GPU?

| Task | Requires GPU? |
|------|--------------|
| Loop Extraction | ❌ No - runs on CPU |
| Model Training | ✅ Yes - requires GPU |
| Model Inference | ❌ No - can run on CPU |

**You can deploy and use loop extraction immediately**, then add GPU when you're ready for training.

## How to Enable GPU Compute

### Step 1: Check Current GPU Quota

```bash
# Check your current GPU quota in your region
az vm list-usage --location eastus --output table | grep -i "Standard NC"
```

Look for entries like:
- `Standard NCSv3 Family vCPUs`
- `Standard NC Family vCPUs`

If the "Current Value" is 0 and "Limit" is 0, you need to request quota.

### Step 2: Request GPU Quota Increase

**Via Azure Portal:**

1. Go to [Azure Portal](https://portal.azure.com)
2. Navigate to **Subscriptions** → Select your subscription
3. Click **Usage + quotas** in the left menu
4. In the search box, type: `Standard NCSv3`
5. Click on `Standard NCSv3 Family vCPUs`
6. Click **Request increase**
7. Enter new quota limit:
   - Minimum: **6 vCPUs** (for 1 Standard_NC6s_v3 VM)
   - Recommended: **12-24 vCPUs** (for 2-4 VMs)
8. Submit the request

**Via Azure CLI:**

> **Note**: Replace `Your Name` and `your@email.com` with your actual information before running this command.

```bash
# Create a support ticket for quota increase
az support tickets create \
    --ticket-name "GPU-Quota-Request" \
    --title "Request GPU Quota Increase" \
    --description "Requesting quota increase for Standard NCSv3 Family vCPUs to 12" \
    --severity "minimal" \
    --contact-first-name "Your" \
    --contact-last-name "Name" \
    --contact-method "email" \
    --contact-email "your@email.com" \
    --contact-country "US" \
    --contact-language "en-us"
```

**Processing Time:**
- Typically approved within **1-2 business days**
- You'll receive email notification when approved

### Step 3: Verify GPU Availability in Your Region

```bash
# List available GPU SKUs in your region
az vm list-skus --location eastus --size Standard_NC --output table

# Check specific SKU
az vm list-skus --location eastus --query "[?name=='Standard_NC6s_v3']"
```

**If you see `NotAvailableForSubscription`:**
- Your quota request may still be pending
- The SKU might not be available in that region
- Try a different region (westus2, northeurope, etc.)

### Step 4: Add GPU Compute to Your Workspace

Once your quota is approved, you have two options:

#### Option A: Redeploy with GPU Enabled

1. Edit `arm-templates/azuredeploy.parameters.json`:
   ```json
   {
     "deployGpuCompute": {
       "value": true
     }
   }
   ```

2. Redeploy the template:
   ```bash
   ./arm-templates/deploy.sh
   ```

#### Option B: Add GPU Compute Manually (Recommended)

```bash
az ml compute create \
    --name gpu-cluster \
    --type AmlCompute \
    --size Standard_NC6s_v3 \
    --min-instances 0 \
    --max-instances 2 \
    --resource-group musicgen-rg \
    --workspace-name musicgen-ml-workspace
```

**Verify creation:**
```bash
az ml compute show \
    --name gpu-cluster \
    --resource-group musicgen-rg \
    --workspace-name musicgen-ml-workspace
```

### Step 5: Update Your Environment

If you added GPU manually, update your `.env` file:

```bash
GPU_COMPUTE_CLUSTER=gpu-cluster
```

## Alternative GPU Options

If you can't get quota for Standard_NC6s_v3, try these alternatives:

### Other GPU SKUs (in order of cost)

1. **Standard_NC6s_v3** - $3.06/hr (6 vCPUs, 112 GB RAM, 1 V100 GPU) - **Recommended**
2. **Standard_NC4as_T4_v3** - $0.526/hr (4 vCPUs, 28 GB RAM, 1 T4 GPU) - **Budget option**
3. **Standard_NC6** - $0.90/hr (6 vCPUs, 56 GB RAM, 1 K80 GPU) - **Legacy, slower**
4. **Standard_NC12s_v3** - $6.12/hr (12 vCPUs, 224 GB RAM, 2 V100 GPUs) - **Faster training**

### Use Spot Instances (Up to 90% Discount)

```bash
az ml compute create \
    --name gpu-cluster-spot \
    --type AmlCompute \
    --size Standard_NC6s_v3 \
    --tier spot \
    --min-instances 0 \
    --max-instances 2 \
    --resource-group musicgen-rg \
    --workspace-name musicgen-ml-workspace
```

⚠️ **Note**: Spot instances can be evicted when Azure needs capacity. Use for development/testing.

## Testing Your GPU Setup

Once GPU compute is available, test it:

```python
# Test script: test_gpu.py
from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential

ml_client = MLClient(
    DefaultAzureCredential(),
    subscription_id="your-subscription-id",
    resource_group_name="musicgen-rg",
    workspace_name="musicgen-ml-workspace"
)

# List all computes
for compute in ml_client.compute.list():
    print(f"{compute.name}: {compute.type} - {compute.size}")

# Check GPU cluster specifically
gpu_cluster = ml_client.compute.get("gpu-cluster")
print(f"\nGPU Cluster: {gpu_cluster.name}")
print(f"  Size: {gpu_cluster.size}")
print(f"  State: {gpu_cluster.provisioning_state}")
print(f"  Min nodes: {gpu_cluster.scale_settings.min_node_count}")
print(f"  Max nodes: {gpu_cluster.scale_settings.max_node_count}")
```

## Common Issues

### Issue: "OperationNotAllowed" or "QuotaExceeded"

**Solution**: Your quota request hasn't been approved yet. Wait for approval email or check status in Azure Portal → Support tickets.

### Issue: "SKUNotAvailable" in your region

**Solutions**:
1. Try a different region (eastus, westus2, northeurope)
2. Try a different GPU SKU (NC4as_T4_v3, NC6)
3. Contact Azure support for regional availability

### Issue: Deployment succeeds but cluster shows "Failed" state

**Solutions**:
1. Check Activity Log in Azure Portal for detailed error
2. Verify quota in the specific region
3. Try deleting and recreating the compute cluster
4. Ensure no network restrictions (NSG, firewall)

### Issue: Cost concerns

**Solutions**:
1. Keep `min-instances: 0` for auto-scale down when idle
2. Use Spot instances for ~90% discount (with eviction risk)
3. Use smaller GPU SKU for testing (NC4as_T4_v3)
4. Delete compute when not actively training models

## Cost Optimization

### Cost Comparison (per hour)

| SKU | Regular | Spot | Savings |
|-----|---------|------|---------|
| NC4as_T4_v3 | $0.526 | ~$0.053 | 90% |
| NC6s_v3 | $3.06 | ~$0.306 | 90% |
| NC12s_v3 | $6.12 | ~$0.612 | 90% |

### Best Practices

1. **Auto-scale to zero**: Set `min-instances: 0`
2. **Use spot for dev/test**: Regular for production
3. **Monitor usage**: Set budget alerts in Azure Portal
4. **Delete when done**: Remove cluster if not training for weeks
5. **Batch training jobs**: Train multiple models in one session

## Next Steps

Once GPU is set up:

1. ✅ Upload audio files to blob storage
2. ✅ Run loop extraction (CPU-based)
3. ✅ Submit MusicGen training job (GPU-based):
   ```bash
   python config/submit_musicgen_training_job.py
   ```
4. ✅ Monitor training in Azure ML Studio
5. ✅ Deploy trained model for inference

## Need Help?

- **GPU Quota Issues**: [ARM Template Troubleshooting](arm-templates/README.md#gpu-compute-creation-failed)
- **Training Issues**: [Main README Troubleshooting](README.md#troubleshooting)
- **Azure ML Docs**: https://docs.microsoft.com/en-us/azure/machine-learning/
- **GPU Pricing**: https://azure.microsoft.com/en-us/pricing/details/machine-learning/

---

**Remember**: You can start using the toolkit immediately with CPU compute for loop extraction. Add GPU later when you're ready for model training! 🚀
