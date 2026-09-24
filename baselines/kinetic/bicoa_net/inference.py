#!/usr/bin/env python3
"""
BiCoA-Net Inference Script
Predicts protein-ligand binding kinetics (pKoff) from protein sequences and ligand SMILES.

Usage:
    python inference.py --checkpoint model.pt --input data.csv
"""

import os
import sys
import argparse
from pathlib import Path


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description='Predict protein-ligand binding kinetics using BiCoA-Net',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python inference.py --checkpoint model.pt --input data.csv
    
    # Specify output location and use CPU
    python inference.py --checkpoint model.pt --input data.csv --output-dir results --device cpu
    
    # Process multiple files
    python inference.py --checkpoint model.pt --input file1.csv file2.csv file3.csv
        """
    )
    
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to the trained model checkpoint (.pt file)'
    )
    
    parser.add_argument(
        '--input',
        type=str,
        nargs='+',
        required=True,
        help='Input CSV file(s) containing FASTA sequences and SMILES strings'
    )
    
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./predictions',
        help='Directory to save prediction results (default: ./predictions)'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Device to use for inference (default: cuda)'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size for inference (default: 32)'
    )
    
    return parser.parse_args()


def validate_inputs(args):
    """Validate input arguments and files."""
    errors = []
    
    # Check checkpoint file
    if not os.path.exists(args.checkpoint):
        errors.append(f"Checkpoint file not found: {args.checkpoint}")
    
    # Check input files
    missing_files = []
    wrong_format = []
    for file in args.input:
        if not os.path.exists(file):
            missing_files.append(file)
        elif not file.lower().endswith('.csv'):
            wrong_format.append(file)
    
    if missing_files:
        errors.append("The following input files were not found:")
        for file in missing_files:
            errors.append(f"  - {file}")
    
    if wrong_format:
        errors.append("The following files are not CSV format:")
        for file in wrong_format:
            errors.append(f"  - {file}")
        errors.append("Please use CSV files (.csv extension)")
    
    # Check device availability
    if args.device == 'cuda':
        try:
            import torch
            if not torch.cuda.is_available():
                errors.append("CUDA device requested but not available. Use --device cpu instead.")
        except ImportError:
            errors.append("PyTorch not installed. Please install it first.")
    
    return errors


def print_header():
    """Print script header."""
    print("\n" + "="*80)
    print(" "*25 + "BiCoA-Net Inference")
    print("="*80)


def print_summary(args):
    """Print configuration summary."""
    print("\nConfiguration:")
    print(f"  Model checkpoint:  {args.checkpoint}")
    print(f"  Input files:       {len(args.input)} file(s)")
    for i, file in enumerate(args.input, 1):
        print(f"                     {i}. {file}")
    print(f"  Output directory:  {args.output_dir}")
    print(f"  Device:            {args.device}")
    print(f"  Batch size:        {args.batch_size}")
    print()


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_args()
    
    # Print header
    print_header()
    
    # Validate inputs
    errors = validate_inputs(args)
    if errors:
        print("❌ Validation Error(s):")
        for error in errors:
            print(f"   {error}")
        print()
        sys.exit(1)
    
    # Print configuration
    print_summary(args)
    
    # Import prediction function
    try:
        from predict_casestudy import predict_casestudy
    except ImportError as e:
        print("❌ Error: Could not import prediction function.")
        print("   Please ensure 'predict_casestudy.py' is in the same directory.")
        print(f"   Error details: {e}")
        sys.exit(1)
    
    # Run predictions
    try:
        print("Running predictions...")
        print("-" * 80)
        
        predictions = predict_casestudy(
            checkpoint_path=args.checkpoint,
            data_files=args.input,
            output_dir=args.output_dir,
            device=args.device
        )
        
        # Print success message
        print("-" * 80)
        print("\n✅ SUCCESS! All predictions completed")
        print("="*80)
        print(f"\nResults saved to: {args.output_dir}/")
        print("\nOutput files (CSV format):")
        for file in args.input:
            base_name = Path(file).stem
            print(f"  • predictions_{base_name}.csv")
        print()
        
    except Exception as e:
        print("\n" + "="*80)
        print("❌ ERROR during prediction")
        print("="*80)
        print(f"\n{str(e)}\n")
        
        # Print detailed traceback for debugging
        import traceback
        print("Detailed error information:")
        print("-" * 80)
        traceback.print_exc()
        print()
        sys.exit(1)


if __name__ == '__main__':
    main()
