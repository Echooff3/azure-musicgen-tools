# Grant Azure ML compute managed identity access to storage account
# This script assigns Storage Blob Data Contributor role to the ML workspace

# Load environment variables from .env file
$envContent = Get-Content -Path ".env"
$envVars = @{}
foreach ($line in $envContent) {
    if ($line -match '^([^#][^=]+)=(.*)$') {
        $envVars[$matches[1]] = $matches[2]
    }
}

$SUBSCRIPTION_ID = $envVars['AZURE_SUBSCRIPTION_ID']
$RESOURCE_GROUP = $envVars['AZURE_RESOURCE_GROUP']
$WORKSPACE_NAME = $envVars['AZURE_WORKSPACE_NAME']
$STORAGE_ACCOUNT = $envVars['AZURE_STORAGE_ACCOUNT_NAME']

Write-Host "Granting storage access to Azure ML workspace..." -ForegroundColor Cyan
Write-Host "Subscription: $SUBSCRIPTION_ID"
Write-Host "Resource Group: $RESOURCE_GROUP"
Write-Host "Workspace: $WORKSPACE_NAME"
Write-Host "Storage Account: $STORAGE_ACCOUNT"

# Get the workspace managed identity principal ID
Write-Host "`nGetting workspace managed identity..." -ForegroundColor Yellow
$workspaceIdentity = az ml workspace show `
    --name $WORKSPACE_NAME `
    --resource-group $RESOURCE_GROUP `
    --subscription $SUBSCRIPTION_ID `
    --query identity.principal_id `
    -o tsv

if (-not $workspaceIdentity) {
    Write-Host "ERROR: Could not get workspace identity. Make sure workspace has system-assigned managed identity enabled." -ForegroundColor Red
    exit 1
}

Write-Host "Workspace Identity Principal ID: $workspaceIdentity" -ForegroundColor Green

# Get storage account resource ID
$storageId = az storage account show `
    --name $STORAGE_ACCOUNT `
    --resource-group $RESOURCE_GROUP `
    --subscription $SUBSCRIPTION_ID `
    --query id `
    -o tsv

Write-Host "Storage Account ID: $storageId" -ForegroundColor Green

# Assign Storage Blob Data Contributor role
Write-Host "`nAssigning 'Storage Blob Data Contributor' role..." -ForegroundColor Yellow
az role assignment create `
    --assignee $workspaceIdentity `
    --role "Storage Blob Data Contributor" `
    --scope $storageId `
    --subscription $SUBSCRIPTION_ID

if ($LASTEXITCODE -eq 0) {
    Write-Host "`nSUCCESS! Azure ML workspace now has access to storage account." -ForegroundColor Green
    Write-Host "You can now submit jobs that use managed identity." -ForegroundColor Green
} else {
    Write-Host "`nERROR: Failed to assign role. Check your permissions." -ForegroundColor Red
    exit 1
}
