from transformers import CLIPModel, CLIPProcessor, LlavaForConditionalGeneration, LlavaProcessor
from abc import abstractmethod
import numpy as np
import numpy.typing as npt
import torch
from PIL import Image


#abstract class
class EmbeddingModel:
    """Abstract class for embedding models."""
    @abstractmethod
    def embed_image(self, image_path):
        """Embed a single image."""
        pass


class ClipEmbeddingModel(EmbeddingModel):
    """Handles image and dataset embedding operations."""

    def __init__(self, use_text: bool=False):
        # change this to small backbone for base models
        self.embedding_model = CLIPModel.from_pretrained(
            "openai/clip-vit-large-patch14"
        ).to("cuda")
        self.embedding_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-large-patch14"
        )
        self.use_text = use_text
        # Initialize the vlm only if text embedding is needed
        if self.use_text:
            self.llava_model = LlavaForConditionalGeneration.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            ).to("cuda")
            self.llava_processor = LlavaProcessor.from_pretrained(
                "llava-hf/llava-1.5-7b-hf"
            )
            self.prompt = "Describe this image in detail. In your description, specifically mention ALL VISIBLE parts of each object in the image."
            self.conversation = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": self.prompt}]}]

    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise

        if self.embedding_processor is None or self.embedding_model is None:
            print("Error: CLIP model or processor is not initialized.")
            raise
        
        inputs = self.embedding_processor(images=image, return_tensors="pt").to("cuda")

        # Generate image embeddings
        with torch.no_grad():
            image_embeddings = (
                self.embedding_model.get_image_features(**inputs).detach().cpu().numpy()
            )
        return image_embeddings

    def generate_caption(self, image_path) -> str:
        """Generate a caption for a single image."""
        try:
            image = Image.open(image_path).convert("RGB")
        except FileNotFoundError:
            print(f"Error: Image file '{image_path}' not found.")
            raise
        except Exception as e:
            print(f"Error opening image '{image_path}': {e}")
            raise
        
        prompt_formatted = self.llava_processor.apply_chat_template(self.conversation, tokenize=False, add_generation_prompt=True)
        inputs = self.llava_processor(images=image, text=prompt_formatted, return_tensors="pt").to("cuda")
        print(f"type and shape of inputs is : {type(inputs)}")

        with torch.no_grad():
            # 77 is the max number of tokens that clip text encoder can handle
            outputs = (
                self.llava_model.generate(**inputs, max_new_tokens=77,
                pad_token_id = self.llava_processor.tokenizer.pad_token_id)
            )
        input_token_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_token_length:]
        generated_text = self.llava_processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        #generated_text = generated_text.replace(self.prompt, "").strip()
        print(f"Generated caption for image {image_path}: {generated_text}")

        return generated_text

    def embed_text(self, text) -> npt.NDArray:
        """Embed a single text."""
        inputs = self.embedding_processor(text=text, return_tensors="pt").to("cuda")
        
        with torch.no_grad():
            text_embeddings = (
                self.embedding_model.get_text_features(**inputs).detach().cpu().numpy()
            )

        return text_embeddings


class EmbeddingManager:
    """Handles image and dataset embedding operations."""
    
    def __init__(self, use_text: bool=False, image_weight: float=0.5, text_weight: float=0.5):
        self.embedding_model = ClipEmbeddingModel(use_text=use_text)
        self.use_text = use_text
        self.image_weight = image_weight
        self.text_weight = text_weight
    
    def embed_image(self, image_path) -> npt.NDArray:
        """Embed a single image."""
        return self.embedding_model.embed_image(image_path)

    def embed_text_for_image(self, image_path) -> npt.NDArray:
        """Generate caption for an image and embed the caption."""
        caption = self.embedding_model.generate_caption(image_path)

        return self.embedding_model.embed_text(caption)
    
    def embed_dataset(self, dataset_path, debug=False) -> npt.NDArray:
        """Embed all images in a dataset."""
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path '{dataset_path}' not found.")

        print(f"Embedding dataset from '{dataset_path}' ...")
        dataset_embeddings = []
        dataset_text_embeddings = []
        dataset_mixed_embeddings = []

        # IDD has both png and jpg images in train set
        image_files = list(dataset_path.rglob("*.png")) + list(dataset_path.rglob("*.jpg"))
    
        if not image_files:
            print(f"Warning: No images found in dataset path '{dataset_path}'.")
            return []

        from tqdm import tqdm
        for img in tqdm(image_files, desc="Embedding images"):
            embedding = self.embed_image(img)
            print(f"Shape and type of image embedding is: {embedding.shape}, {type(embedding)}")
            if embedding is not None:
                dataset_embeddings.append(embedding)
            else:
                raise ValueError(f"Error embedding image '{img}'.")
            
            if self.use_text:
                cache_path = dataset_path / "caption_cache"
                print(f"Cache path is: {cache_path}")
                caption_embedding = self.embed_text_for_image(img)
                print(f"Shape and type of text embedding is : {caption_embedding.shape}, {type(caption_embedding)}")
                if caption_embedding is not None:
                    dataset_text_embeddings.append(caption_embedding)
                else:
                    raise ValueError(f"Error embedding text for image '{img}'.")
                
                # calculate mixed embeddings
                mixed_embedding = self.image_weight * embedding + self.text_weight * caption_embedding
                dataset_mixed_embeddings.append(mixed_embedding)

        print("Finished embedding dataset.")

        return dataset_embeddings, dataset_text_embeddings, dataset_mixed_embeddings
        
    def calculate_statistics(self, domain_name, domain_path, train_path):

        """
        Calculate or load domain statistics.
        Args:
            domain_name (str): The name of the domain.
            domain_path (Path): The path to the domain database where the statistics will be saved.
            train_path (Path): The path to the train set.
        Returns:
            dict: A dictionary containing the statistics.
        """
        suffix = "_statistics.npz"
        statistics_path = domain_path / f"{domain_name}{suffix}"
        stats_dict = {}

        print(f"Statistics file: {statistics_path}")
        if statistics_path.exists():  # Load the data if it exists
            try:
                print(f"Loading statistics from {domain_name}{suffix} ...")
                stats = np.load(statistics_path)
                stats_dict.update({
                    "train_average_embedding": stats["train_average_embedding"],
                })
                print(f"Statistics loaded from {domain_name}{suffix}")
                return stats_dict
            except Exception as e:
                print(f"Error loading statistics file '{statistics_path}': {e}")
                return None

        print(f"Statistics file {statistics_path} does not exist, calculating statistics for domain '{domain_name}' ...")
        train_dataset_embeddings, train_dataset_text_embeddings, train_dataset_mixed_embeddings = self.embed_dataset(train_path)

        if not train_dataset_embeddings:
            raise ValueError("No embeddings were generated for dataset.")

        try:
            # if using text, calculate mixed embeddings
            if self.use_text:
                train_average_embedding = np.mean(train_dataset_mixed_embeddings, axis=0)
            else:
                train_average_embedding = np.mean(train_dataset_embeddings, axis=0)
        except Exception as e:
            print(f"Error computing mean embedding: {e}")
            raise

        stats_dict.update({
            "train_average_embedding": train_average_embedding
        })

        try:
            np.savez(
                statistics_path,
                train_average_embedding=train_average_embedding,
            )
            print(f"Statistics saved to {domain_name}{suffix}")
        except Exception as e:
            print(f"Error saving statistics file '{statistics_path}': {e}")
            raise
        return stats_dict
