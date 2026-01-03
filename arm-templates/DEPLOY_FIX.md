# Quick Fix: Deploy to Azure Button Error

If you're seeing this error:

```
There was an error downloading the template from URI 'https://raw.githubusercontent.com/...'
Ensure that the template is publicly accessible and that the publisher has enabled CORS policy...
```

## Why This Happens

GitHub serves raw files with `Content-Type: text/plain` instead of `application/json`. Azure Portal sometimes rejects templates because of this.

## Solution 1: Use the Deployment Script (Easiest)

**Linux/Mac:**
```bash
git clone https://github.com/Echooff3/azure-musicgen-tools.git
cd azure-musicgen-tools
chmod +x arm-templates/deploy.sh
./arm-templates/deploy.sh
```

**Windows:**
```cmd
git clone https://github.com/Echooff3/azure-musicgen-tools.git
cd azure-musicgen-tools
arm-templates\deploy.bat
```

This completely bypasses the Portal button issue!

## Solution 2: Manual Portal Deployment

1. **Get the template**:
   - Go to https://raw.githubusercontent.com/Echooff3/azure-musicgen-tools/main/arm-templates/azuredeploy.json
   - Copy all the content (Ctrl+A, Ctrl+C)

2. **Deploy in Portal**:
   - Go to https://portal.azure.com
   - Search for "Deploy a custom template"
   - Click "Build your own template in the editor"
   - Delete the example content
   - Paste the template you copied
   - Click "Save"

3. **Fill in parameters**:
   - Choose your subscription
   - Create or select resource group
   - Choose region (e.g., East US)
   - Set project name (max 10 characters)

4. **Deploy**:
   - Click "Review + Create"
   - Review the settings
   - Click "Create"

## Solution 3: Azure CLI (For Advanced Users)

```bash
# Login to Azure
az login

# Set variables
RESOURCE_GROUP="musicgen-rg"
LOCATION="eastus"
PROJECT_NAME="musicgen"

# Create resource group
az group create --name $RESOURCE_GROUP --location $LOCATION

# Deploy
az deployment group create \
  --name musicgen-deployment \
  --resource-group $RESOURCE_GROUP \
  --template-file arm-templates/azuredeploy.json \
  --parameters projectName=$PROJECT_NAME
```

## Need More Help?

See the [full deployment guide](README.md) for detailed instructions, cost optimization tips, and troubleshooting.
