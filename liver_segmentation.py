#!/usr/bin/env python3
"""
Multi-Organelle Liver Segmentation using nnU-Net

Description:
    Automated segmentation of liver organelles in electron microscopy images 
    using nnU-Net v2 framework. Ground truth annotations were created using 
    VAST (Volume Annotation and Segmentation Tool) with minimal training data.
    
Reference:
    Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021).
    nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation.
    Nature methods, 18(2), 203-211.

"""

import os
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from PIL import Image
import json

# CONFIGURATION
ORGANELLE_IDS = {
    'mitochondria': 1,
    'nucleus': 2, 
    'er': 3,
    'lipid_droplets': 4,
    'cell_boundaries': 5
}

ORGANELLE_NAMES = {
    1: 'Mitochondria',
    2: 'Nucleus', 
    3: 'ER',
    4: 'Lipid Droplets',
    5: 'Cell Boundaries'
}

VISUALIZATION_COLORS = {
    0: [0, 0, 0],      # Background
    1: [255, 0, 0],    # Mitochondria - Red
    2: [0, 0, 255],    # Nucleus - Blue
    3: [0, 255, 0],    # ER - Green
    4: [255, 255, 0],  # Lipid Droplets - Yellow
    5: [255, 0, 255]   # Cell Boundaries - Magenta
}

DATASET_ID = 2
DATASET_NAME = f'Dataset{DATASET_ID:03d}_SingleTraining'
MAX_TRAINING_TIME = 14400  # 4 hours

# ENVIRONMENT SETUP
def setup_nnunet_environment() -> Path:
    """Initialize nnU-Net environment variables and directory structure."""
    print("Setting up nnU-Net environment...")
    
    # Set environment variables
    os.environ['nnUNet_raw'] = '/content/nnUNet_raw'
    os.environ['nnUNet_preprocessed'] = '/content/nnUNet_preprocessed'
    os.environ['nnUNet_results'] = '/content/nnUNet_results'
    
    # Create dataset directory structure
    dataset_path = Path('/content/nnUNet_raw') / DATASET_NAME
    subdirs = ['imagesTr', 'labelsTr', 'imagesTs', 'labelsTs']
    
    for subdir in subdirs:
        (dataset_path / subdir).mkdir(parents=True, exist_ok=True)
    
    print("Environment setup complete")
    return dataset_path

def install_dependencies() -> None:
    """Install required packages for the segmentation pipeline."""
    print("Installing dependencies...")
    
    packages = ['nnunetv2', 'seaborn', 'scipy']
    for package in packages:
        subprocess.run(['pip', 'install', package, '--quiet'], check=True)
    
    print("Dependencies installed")

# DATA PREPROCESSING
def convert_to_nifti(input_path: str, output_path: str) -> bool:
    """
    Convert image to nnU-Net compatible NIfTI format.
    
    Args:
        input_path: Path to input image file
        output_path: Path for output NIfTI file
        
    Returns:
        Success status
    """
    try:
        if not os.path.exists(input_path):
            print(f"File not found: {input_path}")
            return False
            
        img = Image.open(input_path)
        img_array = np.array(img)
        
        # Convert to grayscale if needed
        if len(img_array.shape) == 3:
            img_array = img_array[:, :, 0]
        
        sitk_img = sitk.GetImageFromArray(img_array)
        sitk.WriteImage(sitk_img, output_path)
        return True
        
    except Exception as e:
        print(f"Error converting {input_path}: {e}")
        return False


def create_multiclass_mask(mask_paths: Dict[str, str], output_path: str) -> bool:
    """
    Combine individual organelle masks into single multi-class mask.
    
    Args:
        mask_paths: Dictionary mapping organelle names to mask file paths
        output_path: Path for output combined mask
        
    Returns:
        Success status
    """
    try:
        # Find existing masks
        existing_masks = {org: path for org, path in mask_paths.items() 
                         if os.path.exists(path)}
        
        if not existing_masks:
            print("No mask files found")
            return False
        
        # Load reference mask for dimensions
        first_mask_path = next(iter(existing_masks.values()))
        ref_mask = np.array(Image.open(first_mask_path))
        
        if len(ref_mask.shape) == 3:
            ref_mask = ref_mask[:, :, 0]
        
        # Initialize combined mask
        combined_mask = np.zeros_like(ref_mask, dtype=np.uint8)
        
        # Assign organelle class IDs
        for organelle, mask_path in existing_masks.items():
            mask = np.array(Image.open(mask_path))
            if len(mask.shape) == 3:
                mask = mask[:, :, 0]
            
            binary_mask = (mask > 0).astype(np.uint8)
            combined_mask[binary_mask > 0] = ORGANELLE_IDS[organelle]
        
        # Save as NIfTI
        sitk_mask = sitk.GetImageFromArray(combined_mask)
        sitk.WriteImage(sitk_mask, output_path)
        return True
        
    except Exception as e:
        print(f"Error creating mask: {e}")
        return False


def prepare_training_data(dataset_path: Path, data_config: Dict) -> int:
    """
    Process training images and masks for nnU-Net.
    
    Args:
        dataset_path: Path to dataset directory
        data_config: Configuration dictionary with image and mask paths
        
    Returns:
        Number of successfully processed training cases
    """
    print("Processing training data...")
    
    success_count = 0
    
    for i, (case_name, case_data) in enumerate(data_config.items(), 1):
        print(f"Processing {case_name}...")
        
        # Convert image
        image_output = dataset_path / 'imagesTr' / f'Case_{i:04d}_0000.nii.gz'
        image_success = convert_to_nifti(case_data['image'], str(image_output))
        
        # Create combined mask
        mask_output = dataset_path / 'labelsTr' / f'Case_{i:04d}.nii.gz'
        mask_success = create_multiclass_mask(case_data['masks'], str(mask_output))
        
        if image_success and mask_success:
            success_count += 1
            print(f"Successfully processed {case_name}")
        else:
            print(f"Failed to process {case_name}")
    
    print(f"Training data prepared: {success_count} cases")
    return success_count


def prepare_test_data(dataset_path: Path, test_config: List[Dict]) -> int:
    """
    Process test images with ground truth masks.
    
    Args:
        dataset_path: Path to dataset directory
        test_config: List of test case configurations
        
    Returns:
        Number of successfully processed test cases
    """
    print("Processing test data...")
    
    success_count = 0
    
    for i, test_case in enumerate(test_config, 1):
        print(f"Processing test case {i}...")
        
        # Convert image
        image_output = dataset_path / 'imagesTs' / f'test_{i:03d}_0000.nii.gz'
        image_success = convert_to_nifti(test_case['image'], str(image_output))
        
        # Create combined mask
        mask_output = dataset_path / 'labelsTs' / f'test_{i:03d}.nii.gz'
        mask_success = create_multiclass_mask(test_case['masks'], str(mask_output))
        
        if image_success and mask_success:
            success_count += 1
            print(f"Successfully processed test case {i}")
        else:
            print(f"Failed to process test case {i}")
    
    print(f"Test data prepared: {success_count} cases")
    return success_count


def prepare_prediction_data(dataset_path: Path, prediction_dir: str) -> int:
    """
    Process unlabeled images for prediction.
    
    Args:
        dataset_path: Path to dataset directory
        prediction_dir: Directory containing unlabeled images
        
    Returns:
        Number of successfully processed prediction images
    """
    print("Processing prediction data...")
    
    if not os.path.exists(prediction_dir):
        print(f"Prediction directory not found: {prediction_dir}")
        return 0
    
    # Get image files
    image_extensions = ('.tif', '.tiff', '.png', '.jpg', '.jpeg')
    image_files = [f for f in os.listdir(prediction_dir)
                   if f.lower().endswith(image_extensions)]
    image_files.sort()
    
    print(f"Found {len(image_files)} images for prediction")
    
    success_count = 0
    for i, filename in enumerate(image_files):
        input_path = os.path.join(prediction_dir, filename)
        output_path = dataset_path / 'imagesTs' / f'pred_{i:03d}_0000.nii.gz'
        
        if convert_to_nifti(input_path, str(output_path)):
            success_count += 1
    
    print(f"Prediction data prepared: {success_count} images")
    return success_count


def create_dataset_config(dataset_path: Path, num_training: int, num_test: int) -> None:
    """
    Create dataset.json configuration file for nnU-Net.
    
    Args:
        dataset_path: Path to dataset directory
        num_training: Number of training images
        num_test: Total number of test images
    """
    print("Creating dataset configuration...")
    
    dataset_json = {
        "channel_names": {"0": "grayscale"},
        "labels": {
            "background": 0,
            "mitochondria": 1,
            "nucleus": 2,
            "ER": 3,
            "lipid_droplets": 4,
            "cell_boundaries": 5
        },
        "numTraining": num_training,
        "numTest": num_test,
        "file_ending": ".nii.gz"
    }
    
    config_path = dataset_path / 'dataset.json'
    with open(config_path, 'w') as f:
        json.dump(dataset_json, f, indent=2)
    
    print("Dataset configuration created")

# MODEL TRAINING
def run_preprocessing() -> bool:
    """Run nnU-Net preprocessing step."""
    print("Running nnU-Net preprocessing...")
    
    try:
        cmd = f"nnUNetv2_plan_and_preprocess -d {DATASET_ID} -c 2d"
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("Preprocessing completed successfully")
            return True
        else:
            print(f"Preprocessing failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Preprocessing timed out")
        return False
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return False

def run_training() -> bool:
    """Run nnU-Net training with time limit."""
    print(f"Starting model training (max {MAX_TRAINING_TIME//3600} hours)...")
    
    try:
        cmd = f"timeout {MAX_TRAINING_TIME} nnUNetv2_train {DATASET_ID} 2d all --npz"
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("Training completed successfully")
        elif result.returncode == 124:  # timeout exit code
            print("Training stopped due to time limit")
        else:
            print(f"Training failed: {result.stderr}")
            return False
        
        # Prepare checkpoint for inference
        checkpoint_dir = (Path('/content/nnUNet_results') / DATASET_NAME / 
                         'nnUNetTrainer__nnUNetPlans__2d' / 'fold_all')
        
        best_checkpoint = checkpoint_dir / 'checkpoint_best.pth'
        final_checkpoint = checkpoint_dir / 'checkpoint_final.pth'
        
        if best_checkpoint.exists() and not final_checkpoint.exists():
            import shutil
            shutil.copy(str(best_checkpoint), str(final_checkpoint))
            print("Checkpoint prepared for inference")
        
        return True
        
    except Exception as e:
        print(f"Training error: {e}")
        return False

# PREDICTION AND EVALUATION
def run_prediction(dataset_path: Path) -> bool:
    """Run nnU-Net prediction on test images."""
    print("Running prediction...")
    
    try:
        input_dir = dataset_path / 'imagesTs'
        output_dir = '/content/predictions'
        
        cmd = (f"nnUNetv2_predict -i {input_dir} -o {output_dir} "
               f"-d {DATASET_ID} -c 2d -f all")
        
        result = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=3600)
        
        if result.returncode == 0:
            print("Prediction completed successfully")
            return True
        else:
            print(f"Prediction failed: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        print("Prediction timed out")
        return False
    except Exception as e:
        print(f"Prediction error: {e}")
        return False


def calculate_segmentation_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate segmentation performance metrics for each organelle.
    
    Args:
        prediction: Predicted segmentation mask
        ground_truth: Ground truth segmentation mask
        
    Returns:
        Dictionary containing metrics for each organelle
    """
    metrics = {}
    
    for class_id in range(1, 6):
        organelle = ORGANELLE_NAMES[class_id]
        pred_mask = (prediction == class_id).astype(np.uint8)
        gt_mask = (ground_truth == class_id).astype(np.uint8)
        
        # Calculate confusion matrix components
        tp = np.sum(pred_mask * gt_mask)
        fp = np.sum(pred_mask * (1 - gt_mask))
        fn = np.sum((1 - pred_mask) * gt_mask)
        
        # Calculate metrics
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        dice = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0
        iou = tp / (tp + fp + fn) if (tp + fp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics[organelle] = {
            'dice': dice,
            'iou': iou,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    return metrics

def apply_color_mapping(mask: np.ndarray) -> np.ndarray:
    """Apply color mapping to segmentation mask for visualization."""
    h, w = mask.shape
    colored = np.zeros((h, w, 3), dtype=np.uint8)
    for class_id, color in VISUALIZATION_COLORS.items():
        colored[mask == class_id] = color
    return colored

def visualize_segmentation_results(image: np.ndarray, prediction: np.ndarray, 
                                 ground_truth: Optional[np.ndarray] = None, 
                                 title: str = "Segmentation Results") -> None:
    """
    Visualize segmentation results with color overlay.
    
    Args:
        image: Original grayscale image
        prediction: Predicted segmentation mask
        ground_truth: Ground truth segmentation mask (optional)
        title: Plot title
    """
    pred_colored = apply_color_mapping(prediction)
    
    if ground_truth is not None:
        # Evaluation visualization
        gt_colored = apply_color_mapping(ground_truth)
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(gt_colored)
        axes[1].set_title('Ground Truth')
        axes[1].axis('off')
        
        axes[2].imshow(pred_colored)
        axes[2].set_title('AI Prediction')
        axes[2].axis('off')
        
        # Error visualization
        error_mask = (prediction != ground_truth).astype(np.uint8)
        axes[3].imshow(image, cmap='gray')
        axes[3].imshow(error_mask, alpha=0.7, cmap='Reds')
        axes[3].set_title('Prediction Errors')
        axes[3].axis('off')
    else:
        # Prediction only
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(image, cmap='gray')
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(pred_colored)
        axes[1].set_title('AI Prediction')
        axes[1].axis('off')
        
        axes[2].imshow(image, cmap='gray')
        axes[2].imshow(pred_colored, alpha=0.6)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
    
    plt.suptitle(title, fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()

def evaluate_test_results(dataset_path: Path, num_test_cases: int) -> Dict:
    """
    Evaluate model performance on test images with ground truth.
    
    Args:
        dataset_path: Path to dataset directory
        num_test_cases: Number of test cases to evaluate
        
    Returns:
        Dictionary containing all evaluation metrics
    """
    print("Evaluating test results...")
    
    all_metrics = {}
    
    for i in range(1, num_test_cases + 1):
        test_name = f"test_{i:03d}"
        
        try:
            # Load images
            test_img = sitk.GetArrayFromImage(
                sitk.ReadImage(str(dataset_path / 'imagesTs' / f'{test_name}_0000.nii.gz'))
            )
            prediction = sitk.GetArrayFromImage(
                sitk.ReadImage(f'/content/predictions/{test_name}.nii.gz')
            )
            ground_truth = sitk.GetArrayFromImage(
                sitk.ReadImage(str(dataset_path / 'labelsTs' / f'{test_name}.nii.gz'))
            )
            
            # Calculate metrics
            metrics = calculate_segmentation_metrics(prediction, ground_truth)
            all_metrics[test_name] = metrics
            
            # Display results
            print(f"\nResults for {test_name}:")
            print(f"{'Organelle':<15} {'Dice':<6} {'IoU':<6} {'Precision':<6} {'Recall':<6}")
            print("-" * 55)
            for organelle, scores in metrics.items():
                print(f"{organelle:<15} {scores['dice']:.3f}  {scores['iou']:.3f}  "
                      f"{scores['precision']:.3f}  {scores['recall']:.3f}")
            
            # Visualize results
            visualize_segmentation_results(test_img, prediction, ground_truth, f"Results: {test_name}")
            
        except Exception as e:
            print(f"Error processing {test_name}: {e}")
    
    return all_metrics

def print_summary_statistics(all_metrics: Dict) -> None:
    """Print average performance statistics across all test cases."""
    if not all_metrics:
        print("No metrics available for summary")
        return
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    
    # Calculate averages
    summary = {}
    for organelle in ORGANELLE_NAMES.values():
        dice_scores = [metrics[organelle]['dice'] for metrics in all_metrics.values()]
        iou_scores = [metrics[organelle]['iou'] for metrics in all_metrics.values()]
        
        summary[organelle] = {
            'dice_mean': np.mean(dice_scores),
            'dice_std': np.std(dice_scores),
            'iou_mean': np.mean(iou_scores),
            'iou_std': np.std(iou_scores)
        }
    
    # Print results
    print(f"{'Organelle':<15} {'Dice Score':<15} {'IoU Score':<15}")
    print("-" * 50)
    for organelle, scores in summary.items():
        print(f"{organelle:<15} {scores['dice_mean']:.3f}±{scores['dice_std']:.3f}    "
              f"{scores['iou_mean']:.3f}±{scores['iou_std']:.3f}")
    
    # Identify best and worst performing organelles
    organelle_performance = [(org, summary[org]['dice_mean']) for org in summary.keys()]
    organelle_performance.sort(key=lambda x: x[1], reverse=True)
    
    best_organelle, best_score = organelle_performance[0]
    worst_organelle, worst_score = organelle_performance[-1]
    
    print(f"\nBest performing: {best_organelle} (Dice: {best_score:.3f})")
    print(f"Most challenging: {worst_organelle} (Dice: {worst_score:.3f})")

def show_prediction_samples(dataset_path: Path, num_samples: int = 3) -> None:
    """Display sample predictions for visual inspection."""
    print(f"\nShowing {num_samples} prediction samples:")
    
    for i in range(num_samples):
        try:
            pred_img = sitk.GetArrayFromImage(
                sitk.ReadImage(str(dataset_path / 'imagesTs' / f'pred_{i:03d}_0000.nii.gz'))
            )
            prediction = sitk.GetArrayFromImage(
                sitk.ReadImage(f'/content/predictions/pred_{i:03d}.nii.gz')
            )
            
            visualize_segmentation_results(pred_img, prediction, title=f"Prediction Sample {i+1}")
            
        except Exception as e:
            print(f"Error showing prediction sample {i+1}: {e}")

# MAIN PIPELINE
def configure_data_paths() -> Tuple[Dict, List[Dict], str]:
    """
    Configure data paths for training, testing, and prediction.
    
    Returns:
        Tuple containing training config, test config, and prediction directory
    """
    # UPDATE THESE PATHS WITH YOUR ACTUAL DATA
    training_config = {
        'train_case': {
            'image': 'path/to/your/training_image.tif',
            'masks': {
                'mitochondria': 'path/to/mito_mask.png',
                'nucleus': 'path/to/nucleus_mask.png',
                'er': 'path/to/er_mask.png',
                'lipid_droplets': 'path/to/lipid_mask.png',
                'cell_boundaries': 'path/to/boundaries_mask.png'
            }
        }
    }
    
    test_config = [
        {
            'image': 'path/to/test_image_1.png',
            'masks': {
                'mitochondria': 'path/to/test1_mito.png',
                'nucleus': 'path/to/test1_nucleus.png',
                'er': 'path/to/test1_er.png',
                'lipid_droplets': 'path/to/test1_lipid.png',
                'cell_boundaries': 'path/to/test1_boundaries.png'
            }
        },
        # Add more test cases as needed
    ]
    
    prediction_dir = "path/to/unlabeled_images/"
    
    return training_config, test_config, prediction_dir

def main() -> None:
    """Execute the complete liver organelle segmentation pipeline."""
    print("Multi-Organelle Liver Segmentation Pipeline")
    
    # Setup environment
    dataset_path = setup_nnunet_environment()
    install_dependencies()
    
    # Configure data paths
    training_config, test_config, prediction_dir = configure_data_paths()
    
    # Prepare data
    num_training = prepare_training_data(dataset_path, training_config)
    num_test = prepare_test_data(dataset_path, test_config)
    num_prediction = prepare_prediction_data(dataset_path, prediction_dir)
    
    if num_training == 0:
        print("No training data found. Please update paths in configure_data_paths()")
        return
    
    # Create dataset configuration
    total_test_images = num_test + num_prediction
    create_dataset_config(dataset_path, num_training, total_test_images)
    
    # Training pipeline
    if not run_preprocessing():
        print("Preprocessing failed. Stopping pipeline.")
        return
    
    if not run_training():
        print("Training failed. Stopping pipeline.")
        return
    
    # Prediction pipeline
    if not run_prediction(dataset_path):
        print("Prediction failed. Stopping pipeline.")
        return
    
    # Evaluation
    if num_test > 0:
        all_metrics = evaluate_test_results(dataset_path, num_test)
        print_summary_statistics(all_metrics)
    
    # Show prediction samples
    if num_prediction > 0:
        show_prediction_samples(dataset_path, min(3, num_prediction))
    
    print("\nPipeline completed successfully")

if __name__ == "__main__":
    main()
