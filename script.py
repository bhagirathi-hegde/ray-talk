import glob
import time
import warnings

import ray
import torch
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

DATA_FOLDER = "data/"

def predict(image_batch):
    warnings.filterwarnings('ignore')
    torch.set_num_threads(1)
    label_list = []
    for image in image_batch:
        try:
            img = Image.open(image)

            processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
            model = ViTForImageClassification.from_pretrained("google/vit-base-patch16-224")

            inputs = processor(images=img, return_tensors="pt")

            outputs = model(**inputs)
            logits = outputs.logits
            predicted_class_idx = logits.argmax(-1).item()
            label_list.append(model.config.id2label[predicted_class_idx])
        except ValueError as e:
            if "Unsupported number of image dimensions" in str(e):
                print(f"Error: Unsupported image format for {image}. Skipping.")
    return label_list


if __name__ == "__main__":
    images = glob.glob(f"{DATA_FOLDER}*.JPEG")[:300]
    # Split the list into chunks of 10 images each
    image_batches = [images[i:i + 15] for i in range(0, len(images), 15)]

    st = time.perf_counter()

    # local single processor 
    results = [predict(image_batch) for image_batch in image_batches]
  
    en = time.perf_counter()
    print("Inference Throughput (images/sec): ", len(images) / (en - st))
