# ARM Template Deployment Guide

This directory contains Azure Resource Manager (ARM) templates to deploy all necessary Azure resources for the MusicGen Tools project.

## 📦 Resources Deployed

The ARM template creates the following resources:

### Core Resources
- **Storage Account** (Standard_LRS by default - most cost-effective)
  - Blob container: `audio-input` - for source audio files
  - Blob container: `audio-loops` - for extracted loops
  - Blob container: `musicgen-models` - for trained models
  
- **Azure Machine Learning Workspace**
  - CPU Compute Cluster - for loop extraction jobs
  - GPU Compute Cluster - for model training jobs
  
- **Container Registry** (Basic tier)
  - For custom Docker images
  
- **Key Vault** (Standard tier)
  - For secrets and credentials
  
- **Application Insights**
  - For monitoring and logging

## 🚀 Quick Deployment

### Option 1: Using Azure Portal (Deploy to Azure Button)

1. Click the button below to deploy directly from the Azure Portal:

[![Deploy to Azure](https://aka.ms/deploytoazurebutton)](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2FEchooff3%2Fazure-musicgen-tools%2Fmain%2Farm-templates%2Fazuredeploy.json)

2. Fill in the required parameters:
   - **Subscription**: Select your Azure subscription
   - **Resource Group**: Create new or use existing
   - **Region**: Choose your preferred region (e.g., East US)
   - **Project Name**: Prefix for all resources (max 10 characters)
   - Adjust compute sizes and node counts as needed

3. Click "Review + Create" then "Create"

**If you encounter an error** about downloading the template, use **Option 1b** below.

### Option 1b: Manual Portal Deployment (If Button Fails)

If the Deploy to Azure button fails with an error message like "There was an error downloading the template from URI", follow these steps:

1. **Copy the template content**:
   - Open [azuredeploy.json](https://raw.githubusercontent.com/Echooff3/azure-musicgen-tools/main/arm-templates/azuredeploy.json) in your browser
   - Select all (Ctrl+A / Cmd+A) and copy the entire JSON content

2. **Create custom deployment in Azure Portal**:
   - Go to https://portal.azure.com
   - Search for "Deploy a custom template" or go to https://portal.azure.com/#create/Microsoft.Template
   - Click "Build your own template in the editor"
   - Delete the example template
   - Paste the copied JSON content
   - Click "Save"

3. **Fill in parameters** as described in Option 1 above

4. **Deploy**: Click "Review + Create" then "Create"

> **Note**: This error can occur because GitHub serves raw files with `Content-Type: text/plain` instead of `application/json`. The manual method works around this limitation.

### Option 2: Using Azure CLI (Bash/Linux/Mac) - Recommended

```bash
# Make the script executable
chmod +x arm-templates/deploy.sh

# Run the deployment script
./arm-templates/deploy.sh
```

The script will:
- Check Azure CLI installation
- Verify login status
- Create resource group
- Deploy all resources
- Generate `.env` file with connection details
- Display next steps

### Option 3: Using Azure CLI (Windows)

```cmd
# Run the deployment script
arm-templates\deploy.bat
```

### Option 4: Manual Azure CLI Deployment

```bash
# Set variables
RESOURCE_GROUP="musicgen-rg"
LOCATION="eastus"
PROJECT_NAME="musicgen"

# Create resource group
az group create \
    --name $RESOURCE_GROUP \
    --location $LOCATION

# Deploy template
az deployment group create \
    --name musicgen-deployment \
    --resource-group $RESOURCE_GROUP \
    --template-file arm-templates/azuredeploy.json \
    --parameters arm-templates/azuredeploy.parameters.json \
    --parameters projectName=$PROJECT_NAME location=$LOCATION
```

## ✅ Template Validation

Before deploying, you can validate the ARM template:

```bash
# Validate template syntax and structure
python3 arm-templates/validate-template.py
```

This will check:
- ✅ Valid JSON syntax
- ✅ Correct ARM template structure
- ✅ Resource definitions
- ✅ API versions
- ✅ File integrity

## ⚙️ Configuration Parameters

Edit `azuredeploy.parameters.json` to customize:

| Parameter | Default | Description | Cost Impact |
|-----------|---------|-------------|-------------|
| `projectName` | musicgen | Prefix for resource names (max 10 chars) | None |
| `location` | eastus | Azure region | Varies by region |
| `storageAccountType` | Standard_LRS | Storage redundancy | LRS is cheapest |
| `cpuComputeVmSize` | Standard_DS3_v2 | CPU cluster VM size | ~$0.19/hr per node |
| `cpuComputeMinNodes` | 0 | Min CPU nodes (0 = auto-scale down) | 0 = no idle cost |
| `cpuComputeMaxNodes` | 4 | Max CPU nodes | Caps maximum cost |
| `gpuComputeVmSize` | Standard_NC6s_v3 | GPU cluster VM size | ~$3.06/hr per node |
| `gpuComputeMinNodes` | 0 | Min GPU nodes (0 = auto-scale down) | 0 = no idle cost |
| `gpuComputeMaxNodes` | 2 | Max GPU nodes | Caps maximum cost |

### Resource Naming Convention

The template generates unique names for Azure resources:

- **Storage Account**: `{projectName}st{uniqueId}` 
  - Example: `musicgenstc24d4d3b5bb`
  - Must be 3-24 characters, lowercase letters and numbers only
  - The `uniqueId` is 11 characters derived from the resource group ID
  
- **Container Registry**: `{projectName}acr{uniqueId}`
  - Example: `musicgenacrc24d4d3b5bb`
  - Must be 5-50 characters, alphanumeric only
  - Uses the same 11-character `uniqueId` for consistency

- **Key Vault**: `{projectName}-kv-{uniqueId}`
- **ML Workspace**: `{projectName}-ml-workspace`

**Important**: Keep `projectName` to 10 characters or less to ensure all generated names comply with Azure naming restrictions.

### Cost Optimization Tips

1. **Keep minimum nodes at 0**: Clusters auto-scale down when idle
2. **Use smaller VM sizes for testing**: 
   - CPU: Start with `Standard_DS2_v2` ($0.126/hr)
   - GPU: Start with `Standard_NC6s_v3` ($3.06/hr)
3. **Choose appropriate storage tier**:
   - `Standard_LRS`: Lowest cost, locally redundant
   - `Standard_GRS`: Higher cost, geo-redundant
4. **Set idle time before scale down**: Default is 120 seconds

## 🌍 Recommended Regions

Choose regions based on:
- **Proximity**: Closest to your location for lower latency
- **GPU Availability**: Not all regions have GPU VMs
- **Cost**: Pricing varies by region

Recommended regions with GPU availability:
- `eastus` - East US (often cheapest)
- `westus2` - West US 2
- `northeurope` - North Europe
- `westeurope` - West Europe
- `southeastasia` - Southeast Asia

Check GPU availability: https://azure.microsoft.com/en-us/pricing/details/virtual-machines/linux/

## 📋 After Deployment

### 1. Verify Resources

```bash
# List all resources in the resource group
az resource list \
    --resource-group musicgen-rg \
    --output table
```

### 2. Get Storage Connection String

The deployment script automatically creates a `.env` file. If you need to retrieve it manually:

```bash
az storage account show-connection-string \
    --name <storage-account-name> \
    --resource-group musicgen-rg \
    --output tsv
```

### 3. Access Azure ML Workspace

```bash
# Open workspace in browser
az ml workspace show \
    --name <workspace-name> \
    --resource-group musicgen-rg \
    --query workspaceUrl -o tsv
```

Or visit: https://ml.azure.com

## 🔐 Security Considerations

The template includes:
- ✅ HTTPS-only storage access
- ✅ Disabled public blob access
- ✅ Key Vault for secrets
- ✅ System-assigned managed identity for ML Workspace
- ✅ Soft delete enabled (7-day retention)

### Additional Security Steps

1. **Limit network access**:
```bash
az storage account update \
    --name <storage-account-name> \
    --resource-group musicgen-rg \
    --default-action Deny
```

2. **Add your IP to firewall**:
```bash
az storage account network-rule add \
    --account-name <storage-account-name> \
    --resource-group musicgen-rg \
    --ip-address <your-ip>
```

3. **Use managed identities**: The ML workspace already uses managed identity for storage access

## 🧹 Cleanup

To delete all resources:

```bash
# Delete the entire resource group
az group delete \
    --name musicgen-rg \
    --yes \
    --no-wait
```

⚠️ **Warning**: This permanently deletes ALL resources in the group including:
- All audio files in storage
- All trained models
- All compute clusters
- All logs and monitoring data

## 📊 Cost Estimation

**Monthly costs with default parameters** (assuming 8 hours of use per month):

| Resource | Configuration | Monthly Cost |
|----------|---------------|--------------|
| Storage Account | 100GB, Standard_LRS | ~$2 |
| ML Workspace | Base | Free |
| Container Registry | Basic | ~$5 |
| Key Vault | 1000 operations | ~$0.03 |
| Application Insights | 5GB data | Free tier |
| CPU Compute | 8 hrs × Standard_DS3_v2 | ~$1.50 |
| GPU Compute | 8 hrs × Standard_NC6s_v3 | ~$24.50 |
| **Total** | | **~$33/month** |

**Cost for idle resources** (min nodes = 0): **~$7/month** (storage + registry + vault only)

💡 **Tip**: For development, use smaller VMs and increase for production workloads.

## 🐛 Troubleshooting

### Deploy to Azure Button - Template Download Error

```
Error: There was an error downloading the template from URI 'https://raw.githubusercontent.com/...'
Ensure that the template is publicly accessible and that the publisher has enabled CORS policy...
```

**Root Cause**: This occurs because GitHub serves raw files with `Content-Type: text/plain` instead of `application/json`, which can cause Azure Portal to reject the template.

**Solutions** (in order of recommendation):

1. **Use the CLI deployment** (Option 2 above) - Most reliable:
   ```bash
   ./arm-templates/deploy.sh
   ```

2. **Use manual portal deployment** (Option 1b above):
   - Copy the [template JSON](https://raw.githubusercontent.com/Echooff3/azure-musicgen-tools/main/arm-templates/azuredeploy.json)
   - Go to Azure Portal → "Deploy a custom template" → "Build your own template"
   - Paste the JSON and deploy

3. **Try the button again** - Sometimes it works intermittently

### Deployment Fails - Quota Exceeded

```
Error: Operation could not be completed as it results in exceeding approved quota
```

**Solution**: Request quota increase or use smaller VM sizes:
```bash
az vm list-usage \
    --location eastus \
    --output table
```

### Deployment Fails - Name Already Exists

```
Error: Storage account name 'xxx' is already taken
```

**Solution**: Change `projectName` parameter to generate unique names.

### Can't Access Storage Account

**Solution**: Check firewall rules and network access:
```bash
az storage account show \
    --name <storage-account-name> \
    --resource-group musicgen-rg \
    --query networkRuleSet
```

## 📚 Additional Resources

- [Azure ML Pricing](https://azure.microsoft.com/en-us/pricing/details/machine-learning/)
- [Storage Pricing](https://azure.microsoft.com/en-us/pricing/details/storage/blobs/)
- [ARM Template Reference](https://docs.microsoft.com/en-us/azure/templates/)
- [Azure ML Documentation](https://docs.microsoft.com/en-us/azure/machine-learning/)

## 🔄 Updates and Modifications

To modify existing deployment:

1. Edit `azuredeploy.parameters.json`
2. Re-run deployment with same resource group name
3. Azure will update changed resources (incremental deployment)

Example - Scale up GPU cluster:
```json
{
  "gpuComputeMaxNodes": {
    "value": 4
  }
}
```

Then re-deploy:
```bash
./arm-templates/deploy.sh
```
