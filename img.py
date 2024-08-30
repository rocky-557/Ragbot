import fitz  # PyMuPDF for PDF processing
from PIL import Image
import io
from transformers import CLIPProcessor, CLIPModel
from torch.utils.data import DataLoader
from datasets import Dataset, Image as DImage
from tqdm import tqdm
import numpy as np
import torch
import PIL
from typing import List, Union
import os
from sklearn.metrics.pairwise import cosine_similarity
import pickle

with open('img_embeddings.pkl','rb') as f:
    vector_embedding=pickle.load(f)

# Initialize CLIP model and processor
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
preprocess = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

def extract_images_from_pdf(pdf_path: str, output_folder: str) -> List[Image.Image]:
    """
    Extracts images from a PDF file, saves them to a folder,
    and returns them as a list of PIL Images.
    """
    doc = fitz.open(pdf_path)
    images = []
    for page_num in range(doc.page_count):
        page = doc[page_num]
        image_list = page.get_images()
        for img_index, img in enumerate(image_list):
            xref = img[0]
            base_image = doc.extract_image(xref)
            image_bytes = base_image["image"]
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

            # Save the image
            image_path = os.path.join(output_folder, f"image_{page_num}_{img_index}.png")
            image.save(image_path)

            images.append(image)

    doc.close()
    return images


def encode_images(images: Union[List[str], List[PIL.Image.Image]], batch_size: int):
    """
    Encodes a list of images (file paths or PIL Images) into embeddings using CLIP.
    """
    def transform_fn(el):
        # Handle images depending on their type
        if isinstance(el['image'][0], PIL.Image.Image):  # Handle PIL Images directly
            imgs = el['image']
        else:  # Decode images if they are file paths
            imgs = [DImage().decode_example(_) for _ in el['image']]
        return preprocess(images=imgs, return_tensors='pt').to(device)

    # Prepare the dataset based on input type (file paths or PIL images)
    dataset = Dataset.from_dict({'image': images})
    if isinstance(images[0], str):
        dataset = dataset.cast_column('image', DImage(decode=True))
    else:
        dataset = dataset.cast_column('image', DImage())  # Directly use PIL Images

    dataset.set_format('torch')
    dataset.set_transform(transform_fn)

    # Initialize DataLoader for batching
    dataloader = DataLoader(dataset, batch_size=batch_size)
    image_embeddings = []

    # Encode images in batches
    pbar = tqdm(total=(len(images) + batch_size - 1) // batch_size, position=0)
    with torch.no_grad():
        for batch in dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            image_embeddings.extend(model.get_image_features(**batch).detach().cpu().numpy())
            pbar.update(1)
        pbar.close()
    return np.stack(image_embeddings)

'''# Extract images from the PDF
pdf_path = 'vol_3.pdf'  # Replace with your PDF path
extracted_images = extract_images_from_pdf(pdf_path,'./')'''

'''# Encode the extracted images and store the resulting embeddings
vector_embedding = np.array(encode_images(extracted_images, batch_size=32))'''

with open('img_embeddings.pkl','rb') as f:
    ve= pickle.load(f)

def find_similar_images(query_embedding, all_embeddings, top_k=5):
    """
    Find the top K similar images to the query embedding.

    :param query_embedding: The embedding of the query image.
    :param all_embeddings: A numpy array containing all the stored embeddings.
    :param top_k: The number of top similar images to return.
    :return: Indices of the top_k most similar images.
    """
    # Compute cosine similarity between the query embedding and all embeddings
    similarities = cosine_similarity([query_embedding], all_embeddings)[0]

    # Get the indices of the top K similar images
    top_k_indices = similarities.argsort()[-top_k:][::-1]

    return top_k_indices, similarities[top_k_indices]

def encode_text_query(query: str):
    """
    Encodes a text query into an embedding using CLIP.
    :param query: The query string (text).
    :return: The embedding of the text query.
    """
    inputs = preprocess(text=[query], return_tensors="pt", padding=True, truncation=True)
    text_features = model.get_text_features(**inputs)
    return text_features.detach().numpy().flatten()


extracted = [file for file in os.listdir('./extracted_images/') if file.endswith(('.png', '.jpg', '.jpeg', '.tiff', '.bmp', '.webp'))]
def get_imgs(query):
  # Encode the query string to get its embedding
  query_string = query  # Example query
  query_embedding = encode_text_query(query_string)

  # Assuming 'vector_embedding' contains the embeddings of images
  similar_image_indices, similarities = find_similar_images(query_embedding, vector_embedding, top_k=5)
  imgs=[]
  for i in similar_image_indices:
        imgs.append(extracted[i])
  return imgs

