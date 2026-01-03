"""
Azure Blob Storage utilities for audio processing.
"""
import os
from typing import List, Optional
from azure.storage.blob import BlobServiceClient, ContainerClient, BlobClient
from azure.core.exceptions import ResourceNotFoundError
import logging

logger = logging.getLogger(__name__)


class AzureBlobManager:
    """Manager for Azure Blob Storage operations."""
    
    def __init__(self, connection_string: Optional[str] = None):
        """
        Initialize Azure Blob Manager.
        
        Args:
            connection_string: Azure Storage connection string. 
                             If None, reads from environment variable.
        """
        self.connection_string = connection_string or os.getenv('AZURE_STORAGE_CONNECTION_STRING')
        if not self.connection_string:
            raise ValueError("Azure Storage connection string not provided")
        
        self.blob_service_client = BlobServiceClient.from_connection_string(
            self.connection_string
        )
    
    def get_container_client(self, container_name: str) -> ContainerClient:
        """Get a container client, creating the container if it doesn't exist."""
        container_client = self.blob_service_client.get_container_client(container_name)
        try:
            container_client.get_container_properties()
        except ResourceNotFoundError:
            logger.info(f"Creating container: {container_name}")
            container_client.create_container()
        return container_client
    
    def list_blobs(self, container_name: str, prefix: Optional[str] = None) -> List[str]:
        """
        List all blobs in a container with optional prefix filter.
        
        Args:
            container_name: Name of the container
            prefix: Optional prefix to filter blobs
            
        Returns:
            List of blob names
        """
        container_client = self.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        return [blob.name for blob in blobs]
    
    def download_blob(self, container_name: str, blob_name: str, 
                     local_path: str) -> None:
        """
        Download a blob to a local file.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            local_path: Local path to save the file
        """
        os.makedirs(os.path.dirname(local_path), exist_ok=True)
        
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name, 
            blob=blob_name
        )
        
        with open(local_path, "wb") as f:
            blob_data = blob_client.download_blob()
            f.write(blob_data.readall())
        
        logger.info(f"Downloaded {blob_name} to {local_path}")
    
    def upload_blob(self, container_name: str, blob_name: str, 
                   local_path: str, overwrite: bool = True) -> None:
        """
        Upload a local file to a blob.
        
        Args:
            container_name: Name of the container
            blob_name: Name of the blob
            local_path: Local path of the file to upload
            overwrite: Whether to overwrite existing blob
        """
        blob_client = self.blob_service_client.get_blob_client(
            container=container_name,
            blob=blob_name
        )
        
        with open(local_path, "rb") as f:
            blob_client.upload_blob(f, overwrite=overwrite)
        
        logger.info(f"Uploaded {local_path} to {blob_name}")
    
    def list_subfolders(self, container_name: str, prefix: str = "") -> List[str]:
        """
        List all unique subfolder paths in a container.
        
        Args:
            container_name: Name of the container
            prefix: Optional prefix to filter
            
        Returns:
            List of subfolder paths
        """
        container_client = self.get_container_client(container_name)
        blobs = container_client.list_blobs(name_starts_with=prefix)
        
        folders = set()
        for blob in blobs:
            # Extract folder path from blob name
            parts = blob.name.split('/')
            if len(parts) > 1:
                folder = '/'.join(parts[:-1])
                folders.add(folder)
        
        return sorted(list(folders))
