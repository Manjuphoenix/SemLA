from pathlib import Path
import os
import yaml
import argparse
import ast
from domain_orchestrator.utils import get_domain_args

DETECTRON2_DATASET_PATH = os.getenv("DETECTRON2_DATASETS")

if __name__ == "__main__":
    # Argparse
    parser = argparse.ArgumentParser()
    # Path to the yaml file that contains the paths to the domains training data
    parser.add_argument("--source_domains_file", type=str, required=True)
    # Path to the lora library where the statistics will be stored
    parser.add_argument("--lora_library_path", type=str, required=True)
    # Flag to enable text embedding generation
    parser.add_argument("--use_text", action="store_true")
    # Weights for combining image and text embeddings, only when use_text is enabled
    parser.add_argument("--image_weight", type=float, default=0.5, help="Weight for image embeddings (default: 0.5)")
    parser.add_argument("--text_weight", type=float, default=0.5, help="Weight for text embeddings (default: 0.5)")
    # Multiple weight combinations for efficient processing
    parser.add_argument("--weight_combinations", type=str, help="List of weight tuples as string, e.g., '[(0.9, 0.1), (0.5, 0.5), (0.2, 0.8)]'")
    # Flag to force regeneration of embeddings even if they already exist
    parser.add_argument("--force_embedding", action="store_true", help="Force regeneration of embeddings even if they already exist")

    # Parse arguments
    args = parser.parse_args()
    source_domains_file = Path(args.source_domains_file)
    lora_library_path = Path(args.lora_library_path)
    use_text = args.use_text
    image_weight = args.image_weight
    text_weight = args.text_weight
    force_embedding = args.force_embedding

    # Parse weight combinations if provided
    weight_combinations = None
    if args.weight_combinations:
        try:
            weight_combinations = ast.literal_eval(args.weight_combinations)
            if not isinstance(weight_combinations, list):
                raise ValueError("weight_combinations must be a list")
            for combo in weight_combinations:
                if not isinstance(combo, tuple) or len(combo) != 2:
                    raise ValueError("Each weight combination must be a tuple of 2 values")
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing weight_combinations: {e}")
            print("Expected format: '[(0.9, 0.1), (0.5, 0.5), (0.2, 0.8)]'")
            exit(1)

    with open(source_domains_file, "r") as f:
        source_domains = yaml.safe_load(f)

    embedding_manager = None

    print("Generating embeddings for all source domains ...")
    if use_text:
        if weight_combinations:
            print(f"Generating combined embeddings for {weight_combinations} weight combinations:")
        else:
            print(f"Generating combined embeddings with weights - Image: {image_weight}, Text: {text_weight}")
    else:
        print("Generating image embeddings only")
    
    if force_embedding:
        print("Force embedding mode enabled - will regenerate embeddings even if they exist")
    
    # Process each domain
    for domain_name in source_domains:

        args = get_domain_args(domain_name, "train", get_cofing_only=True)
        train_dataset_path = Path(args.train_dataset_path)
        print(train_dataset_path)

        assert train_dataset_path.exists(), f"Path to training dataset {train_dataset_path} does not exist!"

        if embedding_manager is None:
            from domain_orchestrator import embedding
            # Initialize with default weights (will be overridden for each combination)
            embedding_manager = embedding.EmbeddingManager(
                use_text=use_text,
                image_weight=image_weight,
                text_weight=text_weight
            )

        domain_path = lora_library_path / Path(domain_name)

        if weight_combinations and use_text:
            # Process multiple weight combinations efficiently
            embedding_manager.calculate_statistics_multiple_weights(
                domain_name=domain_name,
                domain_path=domain_path,
                train_path=train_dataset_path,
                weight_combinations=weight_combinations,
                force_embedding=force_embedding
            )
        else:
            # Process just for single weight combination
            embedding_manager.calculate_statistics(
                domain_name=domain_name,
                domain_path=domain_path,
                train_path=train_dataset_path,
                image_weight=image_weight,
                text_weight=text_weight,
                force_embedding=force_embedding
            )

    print("Finished generating embeddings for all domains")