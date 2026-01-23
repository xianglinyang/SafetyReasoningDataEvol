#!/usr/bin/env python3
"""
Merge a single LoRA checkpoint into a standalone model.

Usage:
    python -m src.train.merge_checkpoint \
        --base_model_path meta-llama/Meta-Llama-3-8B-Instruct \
        --checkpoint_path ./outputs/checkpoint-epoch-0 \
        --merged_output_path ./outputs/merged-epoch-0 \
        --device cpu
"""

import argparse
import logging
import sys
import os

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.utils.train_utils import merge_single_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(
        description="Merge a LoRA checkpoint into a standalone model"
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
        required=True,
        help="Path to the base model (HuggingFace name or local path)"
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True,
        help="Path to the checkpoint directory containing adapter_model/"
    )
    parser.add_argument(
        "--merged_output_path",
        type=str,
        required=True,
        help="Path where the merged model will be saved"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to use for merging (default: cpu)"
    )
    
    args = parser.parse_args()
    
    logger.info("=" * 60)
    logger.info("Merging LoRA Checkpoint")
    logger.info("=" * 60)
    logger.info(f"Base model:      {args.base_model_path}")
    logger.info(f"Checkpoint:      {args.checkpoint_path}")
    logger.info(f"Output:          {args.merged_output_path}")
    logger.info(f"Device:          {args.device}")
    logger.info("=" * 60)
    
    try:
        merge_single_checkpoint(
            base_model_path=args.base_model_path,
            checkpoint_path=args.checkpoint_path,
            merged_output_path=args.merged_output_path,
            device=args.device
        )
        logger.info("=" * 60)
        logger.info("✓ Merge completed successfully!")
        logger.info("=" * 60)
        return 0
    except Exception as e:
        logger.error(f"Error during merge: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
