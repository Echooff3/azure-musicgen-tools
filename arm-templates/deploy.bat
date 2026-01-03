REM Azure MusicGen Tools - ARM Template Deployment

@echo off
setlocal enabledelayedexpansion

echo ==========================================
echo Azure MusicGen Tools - Resource Deployment
echo ==========================================
echo.

REM Configuration
set RESOURCE_GROUP_NAME=musicgen-rg
set LOCATION=eastus
set PROJECT_NAME=musicgen
set TEMPLATE_FILE=arm-templates\azuredeploy.json
set PARAMETERS_FILE=arm-templates\azuredeploy.parameters.json

REM Check if Azure CLI is installed
where az >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Azure CLI is not installed. Please install it from https://docs.microsoft.com/en-us/cli/azure/install-azure-cli
    exit /b 1
)

echo ✅ Azure CLI found
echo.

REM Login check
echo Checking Azure login status...
az account show >nul 2>nul
if %ERRORLEVEL% neq 0 (
    echo ❌ Not logged in to Azure. Running 'az login'...
    az login
) else (
    echo ✅ Already logged in to Azure
)
echo.

REM Show current subscription
for /f "tokens=*" %%i in ('az account show --query name -o tsv') do set SUBSCRIPTION_NAME=%%i
for /f "tokens=*" %%i in ('az account show --query id -o tsv') do set SUBSCRIPTION_ID=%%i
echo 📋 Current subscription: %SUBSCRIPTION_NAME%
echo    Subscription ID: %SUBSCRIPTION_ID%
echo.

set /p CONFIRM="Do you want to use this subscription? (y/n) "
if /i not "%CONFIRM%"=="y" (
    echo Please run 'az account set --subscription <subscription-id>' to change subscription
    exit /b 1
)

REM Create resource group
echo Creating resource group: %RESOURCE_GROUP_NAME% in %LOCATION%...
az group create ^
    --name "%RESOURCE_GROUP_NAME%" ^
    --location "%LOCATION%" ^
    --output table

echo.
echo ✅ Resource group created
echo.

REM Deploy ARM template
echo Deploying Azure resources...
echo This may take 10-15 minutes...
echo.

for /f "tokens=2-4 delims=/ " %%a in ('date /t') do set DATE=%%c%%a%%b
for /f "tokens=1-2 delims=/: " %%a in ('time /t') do set TIME=%%a%%b
set DEPLOYMENT_NAME=musicgen-deployment-%DATE%-%TIME%

az deployment group create ^
    --name "%DEPLOYMENT_NAME%" ^
    --resource-group "%RESOURCE_GROUP_NAME%" ^
    --template-file "%TEMPLATE_FILE%" ^
    --parameters "%PARAMETERS_FILE%" ^
    --parameters projectName="%PROJECT_NAME%" location="%LOCATION%" ^
    --output table

echo.
echo ✅ Deployment complete!
echo.

REM Get outputs
echo Retrieving deployment outputs...
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.outputs.storageAccountName.value -o tsv') do set STORAGE_ACCOUNT=%%i
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.outputs.storageConnectionString.value -o tsv') do set STORAGE_CONNECTION_STRING=%%i
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.outputs.workspaceName.value -o tsv') do set WORKSPACE_NAME=%%i
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.outputs.cpuComputeCluster.value -o tsv') do set CPU_CLUSTER=%%i
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.outputs.gpuComputeCluster.value -o tsv') do set GPU_CLUSTER=%%i
for /f "tokens=*" %%i in ('az deployment group show --name "%DEPLOYMENT_NAME%" --resource-group "%RESOURCE_GROUP_NAME%" --query properties.parameters.deployGpuCompute.value -o tsv') do set GPU_DEPLOYED=%%i

echo.
echo ==========================================
echo Deployment Summary
echo ==========================================
echo Resource Group:        %RESOURCE_GROUP_NAME%
echo Location:              %LOCATION%
echo Storage Account:       %STORAGE_ACCOUNT%
echo ML Workspace:          %WORKSPACE_NAME%
echo CPU Compute Cluster:   %CPU_CLUSTER%
if "%GPU_DEPLOYED%"=="true" (
    echo GPU Compute Cluster:   %GPU_CLUSTER% (deployed)
) else (
    echo GPU Compute Cluster:   Not deployed (see GPU_SETUP.md to add later)
)
echo.
echo Blob Containers created:
echo   - audio-input        (upload your audio files here)
echo   - audio-loops        (extracted loops)
echo   - musicgen-models    (trained models)
echo.
echo ==========================================
echo.

REM Create .env file
set ENV_FILE=.env
echo Creating %ENV_FILE% file...

(
echo # Azure Storage Configuration
echo AZURE_STORAGE_CONNECTION_STRING=%STORAGE_CONNECTION_STRING%
echo AZURE_STORAGE_ACCOUNT_NAME=%STORAGE_ACCOUNT%
echo INPUT_CONTAINER_NAME=audio-input
echo OUTPUT_CONTAINER_NAME=audio-loops
echo MODEL_CONTAINER_NAME=musicgen-models
echo.
echo # Azure ML Configuration
echo AZURE_SUBSCRIPTION_ID=%SUBSCRIPTION_ID%
echo AZURE_RESOURCE_GROUP=%RESOURCE_GROUP_NAME%
echo AZURE_WORKSPACE_NAME=%WORKSPACE_NAME%
echo.
echo # Compute Clusters
echo CPU_COMPUTE_CLUSTER=%CPU_CLUSTER%
echo GPU_COMPUTE_CLUSTER=%GPU_CLUSTER%
echo.
echo # Audio Processing Configuration
echo BARS_PER_LOOP=4
echo DEFAULT_BPM=120
echo DEFAULT_TIME_SIGNATURE=4/4
echo.
echo # Training Configuration
echo MODEL_NAME=facebook/musicgen-small
echo LEARNING_RATE=1e-4
echo BATCH_SIZE=4
echo NUM_EPOCHS=10
echo LORA_RANK=8
echo LORA_ALPHA=16
) > %ENV_FILE%

echo ✅ Configuration saved to %ENV_FILE%
echo.

REM Print next steps
echo ==========================================
echo Next Steps
echo ==========================================
echo.
echo 1. Upload audio files to the 'audio-input' container:
echo    az storage blob upload-batch ^
echo      --account-name %STORAGE_ACCOUNT% ^
echo      --destination audio-input ^
echo      --source /path/to/your/audio/files
echo.
echo 2. Extract audio loops:
echo    python config\submit_loop_extraction_job.py
echo.
if "%GPU_DEPLOYED%"=="true" (
    echo 3. Train MusicGen model:
    echo    python config\submit_musicgen_training_job.py
) else (
    echo 3. Set up GPU compute (required for model training):
    echo    See GPU_SETUP.md for instructions
    echo    Then: python config\submit_musicgen_training_job.py
)
echo.
echo 4. Deploy model for inference:
echo    python config\deploy_to_azureml.py ^
echo      --model-path .\model ^
echo      --endpoint-name musicgen-endpoint
echo.
echo For more information, see README.md
echo ==========================================

endlocal
