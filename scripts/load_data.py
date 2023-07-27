import os
import io
from PIL import Image
from azure.storage.blob import BlobServiceClient

def load_data():
    # Connect to the azure blob storage
    connection_string = os.getenv('AZURE_ART_CONNECTION_STRING')
    container_name = 'human-ai-art-images'
    #print("CONNECTION STRING: ", connection_string)
    
    blob_service_client = BlobServiceClient.from_connection_string(connection_string)
    container_client = blob_service_client.get_container_client(container_name)
    
    # download the blobs in the container
    blobs = container_client.list_blobs()
    human_images = []
    ai_images = []
    
    for i, blob in enumerate(blobs):
        if blob.name.startswith('AI'):
            image_bytes = container_client.download_blob(blob.name)
            image = Image.open(io.BytesIO(image_bytes.readall()))
            ai_images.append(image)
        else:
            image_bytes = container_client.download_blob(blob.name)
            image = Image.open(io.BytesIO(image_bytes.readall()))
            human_images.append(image)
    
    # return a list for each dataset
    print("Number of human images: ", len(human_images))
    print("Number of AI images: ", len(ai_images))
    return human_images, ai_images
    
#if __name__ == '__main__':
#    load_data()