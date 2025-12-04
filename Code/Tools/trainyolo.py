# =================================================================================================
# --- model.train() Arguments Documentation ---
# =================================================================================================
# For the most up-to-date and complete list, always refer to the official Ultralytics documentation:
# https://docs.ultralytics.com/modes/train/#arguments
#
# Below is a comprehensive list of arguments available for model.train() as of late 2024.
#
# --- Data and Model Arguments ---
#   model (str, optional): Path to the model file to use (e.g., 'yolov8n.pt'). Overrides the model specified in YOLO(). Defaults to None.
#   data (str): Path to the dataset configuration file (e.g., 'coco128.yaml'). This is a required argument.
#   epochs (int): Total number of training epochs. Defaults to 100.
#   time (float, optional): Maximum training time in hours. If set, it can stop training before reaching the total epochs. Defaults to None.
#   patience (int): Epochs to wait for no observable improvement before stopping the training early. Defaults to 50.
#   batch (int): Number of images per batch. Use -1 for AutoBatch, which automatically determines the best batch size. Defaults to 16.
#   imgsz (int or list): The image size for training. Can be a single integer for square images (e.g., 640) or [height, width]. Defaults to 640.
#   save (bool): Whether to save training checkpoints and final model. Defaults to True.
#   save_period (int): Save a checkpoint every 'x' epochs. If -1, it's disabled. Defaults to -1.
#   cache (bool or str): Cache images for faster training. True caches to RAM, 'disk' caches to a local file. Defaults to False.
#   device (str or list): The device to run on, e.g., 'cpu', '0' (for GPU 0), or '0,1,2,3' for multiple GPUs. Defaults to None (auto-select).
#   workers (int): The number of worker threads for data loading. More workers can speed up training on multi-core systems. Defaults to 8.
#   project (str, optional): The name of the project directory to save results to. Defaults to 'runs/detect'.
#   name (str, optional): The name of the specific experiment/run. If not provided, it's automatically named (e.g., 'train', 'train2').
#   exist_ok (bool): If True, allows overwriting an existing experiment with the same name. Defaults to False.
#   pretrained (bool or str): Start training from a pretrained model. Can be True (use the model specified in YOLO()) or a path to a specific weights file. Defaults to True.
#   optimizer (str): The optimizer to use. Options include 'SGD', 'Adam', 'AdamW', 'NAdam', 'RAdam', 'RMSProp'. Defaults to 'auto'.
#   verbose (bool): Whether to print detailed logs during training. Defaults to True.
#   seed (int): A random seed for ensuring reproducible results. Defaults to 0.
#   deterministic (bool): Forces deterministic algorithms, which can aid reproducibility at a potential cost of performance. Defaults to True.
#   single_cls (bool): Treats all classes in a multi-class dataset as a single class. Defaults to False.
#   rect (bool): Use rectangular training, which pads images to be rectangular instead of square, potentially speeding up inference. Defaults to False.
#   cos_lr (bool): Use a cosine learning rate scheduler. Defaults to False.
#   close_mosaic (int): Disables mosaic augmentation for the last 'x' epochs of training. Recommended to be 10 for best results. Defaults to 10.
#   resume (bool or str): Resume training from the last saved checkpoint. Can be True or a path to a specific 'last.pt'. Defaults to False.
#   amp (bool): Enables Automatic Mixed Precision (AMP) training, which can speed up training and reduce memory usage on GPUs. Defaults to True.
#   fraction (float): The fraction of the dataset to use for training (e.g., 0.8 for 80% of images). Defaults to 1.0.
#   profile (bool): Profile ONNX and TensorRT export speeds during training. Defaults to False.
#
# --- Hyperparameter and Augmentation Arguments ---
#   lr0 (float): Initial learning rate (e.g., 0.01). Defaults to 0.01.
#   lrf (float): The final learning rate, calculated as lr0 * lrf. Defaults to 0.01.
#   momentum (float): SGD momentum or Adam beta1. Defaults to 0.937.
#   weight_decay (float): Optimizer weight decay. Defaults to 0.0005.
#   warmup_epochs (float): Number of epochs for learning rate warmup. Can be fractional. Defaults to 3.0.
#   warmup_momentum (float): Initial momentum during the warmup phase. Defaults to 0.8.
#   warmup_bias_lr (float): The learning rate for bias layers during the warmup phase. Defaults to 0.1.
#   box (float): The weight/gain for the box loss function. Defaults to 7.5.
#   cls (float): The weight/gain for the classification loss. For classification models, it's scaled by (batch_size / 64). Defaults to 0.5.
#   dfl (float): The weight/gain for the Distribution Focal Loss (DFL). Defaults to 1.5.
#   hsv_h (float): Image HSV-Hue augmentation (fraction). Defaults to 0.015.
#   hsv_s (float): Image HSV-Saturation augmentation (fraction). Defaults to 0.7.
#   hsv_v (float): Image HSV-Value augmentation (fraction). Defaults to 0.4.
#   degrees (float): Image rotation augmentation (+/- degrees). Defaults to 0.0.
#   translate (float): Image translation augmentation (fraction). Defaults to 0.1.
#   scale (float): Image scaling/zoom augmentation (fraction). Defaults to 0.5.
#   shear (float): Image shear augmentation (+/- degrees). Defaults to 0.0.
#   perspective (float): Image perspective augmentation (fraction). Defaults to 0.0.
#   flipud (float): Probability of flipping an image upside-down. Defaults to 0.0.
#   fliplr (float): Probability of flipping an image left-right. Defaults to 0.5.
#   mosaic (float): Probability of applying mosaic augmentation (combining 4 images). Defaults to 1.0.
#   mixup (float): Probability of applying mixup augmentation (linearly combining 2 images and their labels). Defaults to 0.0.
#   copy_paste (float): Probability of applying copy-paste augmentation. Defaults to 0.0.
#
# --- Logging, Plotting, and Exporting ---
#   plots (bool): Save plots and images during and after training (e.g., confusion matrix, PR curve). Defaults to False.
#   val (bool): Perform validation during training. Defaults to True.
#   v5_metric (bool): Use YOLOv5 metric for validation instead of the standard pycocotools metric. Defaults to False.
#   noval (bool): Disable validation altogether. Only train on the training set. Defaults to False.
#   nosave (bool): Do not save the final model checkpoints. Defaults to False.
#   pbar (bool): Display a tqdm progress bar. Defaults to True.
#   format (str): The format to export the model to upon completion (e.g., 'onnx', 'tflite', 'coreml'). Defaults to 'torchscript'.
#   half (bool): Use half-precision (FP16) for model export. Defaults to False.
#   vid_stride (int): The stride for video frame processing during validation. Defaults to 1.
# =================================================================================================

from ultralytics import YOLO
import argparse
"""A tool to easily manage and apply YOLOv8 training augmentations with presets.
medium = python trainyolo.py --data /path/to/your/data.yaml --model yolov8s.pt --epochs 100
heavy python trainyolo.py --data /path/to/your/data.yaml --model yolov8s.pt --epochs 100 --aug heavy
no = python trainyolo.py --data /path/to/your/data.yaml --model yolov8s.pt --epochs 100 --aug none
"""
# =============================================================================
# --- Augmentation Configuration Tool ---
# =============================================================================

def get_augmentation_presets():
    """
    Defines and returns a dictionary of augmentation presets.
    Each key is a string representing a level of augmentation ('light', 'medium', 'heavy').
    The values are dictionaries containing the hyperparameters for model.train().
    """
    presets = {
        # --- Light Augmentation ---
        # Good for datasets that are already large and varied, or for initial baseline training.
        # Focuses on simple flips and minor color shifts.
        'light': {
            'hsv_h': 0.01,    # Hue
            'hsv_s': 0.4,     # Saturation
            'hsv_v': 0.4,     # Value
            'degrees': 0.0,
            'translate': 0.1,
            'scale': 0.1,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,    # Most common and safe augmentation
            'mosaic': 0.5,    # Reduced mosaic probability
            'mixup': 0.0,
            'copy_paste': 0.0,
        },
        
        # --- Medium Augmentation (Recommended Default) ---
        # A balanced set of augmentations suitable for most datasets.
        # Introduces rotation and scaling, providing more variety without being too extreme.
        'medium': {
            'hsv_h': 0.015,
            'hsv_s': 0.7,
            'hsv_v': 0.4,
            'degrees': 10.0,  # Added moderate rotation
            'translate': 0.1,
            'scale': 0.5,     # Increased scaling range
            'shear': 2.0,     # Added slight shear
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.5,
            'mosaic': 1.0,    # Standard mosaic
            'mixup': 0.1,     # Added light mixup
            'copy_paste': 0.1,# Added light copy-paste
        },

        # --- Heavy Augmentation ---
        # Best for smaller datasets or when the model is severely overfitting.
        # These transformations are more aggressive and create significant variations.
        'heavy': {
            'hsv_h': 0.02,
            'hsv_s': 0.7,
            'hsv_v': 0.5,
            'degrees': 20.0,  # Increased rotation
            'translate': 0.2, # Increased translation
            'scale': 0.7,     # Increased scaling
            'shear': 5.0,     # Increased shear
            'perspective': 0.001, # Added slight perspective
            'flipud': 0.2,    # Added chance of vertical flip
            'fliplr': 0.5,
            'mosaic': 1.0,
            'mixup': 0.2,     # Increased mixup
            'copy_paste': 0.2,# Increased copy-paste
        },
        
        # --- No Augmentation ---
        # Use this to establish a baseline performance without any online augmentation.
        'none': {
            'hsv_h': 0.0,
            'hsv_s': 0.0,
            'hsv_v': 0.0,
            'degrees': 0.0,
            'translate': 0.0,
            'scale': 0.0,
            'shear': 0.0,
            'perspective': 0.0,
            'flipud': 0.0,
            'fliplr': 0.0,
            'mosaic': 0.0, # Mosaic should be turned off if you want no augmentation
            'mixup': 0.0,
            'copy_paste': 0.0,
            'close_mosaic': 0, # Disable the close mosaic phase
        }
    }
    return presets

def train_with_config(model, train_params, aug_params):
    """
    A wrapper function to start YOLOv8 training with a combined configuration.

    Args:
        model (YOLO): An instance of the YOLO model.
        train_params (dict): A dictionary of core training parameters (data, epochs, etc.).
        aug_params (dict): A dictionary of augmentation parameters.
    """
    # Combine the training and augmentation parameters into a single dictionary
    # The ** operator unpacks the dictionaries.
    full_config = {**train_params, **aug_params}

    print("--- Starting Training with the Following Configuration ---")
    for key, value in full_config.items():
        print(f"{key}: {value}")
    print("---------------------------------------------------------")

    # Start the training
    model.train(**full_config)

# =============================================================================
# --- Example Usage ---
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="A tool to easily manage and apply YOLOv8 training augmentations.")
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to the initial model weights (e.g., yolov8n.pt).')
    parser.add_argument('--data', type=str, required=True, help='Path to the dataset YAML file.')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs.')
    parser.add_argument('--batch', type=int, default=8, help='Batch size.')
    parser.add_argument('--imgsz', type=int, default=640, help='Image size.')
    parser.add_argument('--aug', type=str, default='medium', choices=['light', 'medium', 'heavy', 'none'], help='The augmentation preset to use.')
    parser.add_argument('--pat', type=int, default=10, help='Stop if no improvement for x epochs.')
    args = parser.parse_args()

    # --- Step 1: Load the YOLO model ---
    model = YOLO("models/" + args.model)

    # --- Step 2: Define the core training parameters ---
    # These are the settings you would normally change for any training run.
    training_parameters = {
        'data': args.data,
        'epochs': args.epochs,
        'batch': args.batch,
        'imgsz': args.imgsz,
        'plots': True,  # Save plots like confusion matrix and PR curves
        'project': 'modeltraining/runs/', # Project folder for results
        'name': f'{args.model.split(".")[0]}_{args.aug}_aug_e{args.epochs}', # Experiment name
        'patience': args.pat  # Stop if no improvement for 10 epochs
    }

    # --- Step 3: Get the augmentation presets and select one ---
    augmentation_presets = get_augmentation_presets()
    selected_augmentations = augmentation_presets[args.aug]

   
    # If you want to test a specific change, you can easily override a value from the preset.
    # For example, let's start with 'medium' but increase the rotation.
    #
    # Uncomment the line below to try it:
    # selected_augmentations['degrees'] = 25.0 
    
    # --- Step 5: Start training using the wrapper function ---
    train_with_config(model, training_parameters, selected_augmentations)




## 2. EVALUATE ON THE TEST SET (AFTER TRAINING IS COMPLETE)
# from ultralytics import YOLO

# # Load the best model that was saved during training
# final_model = YOLO("path/to/your/runs/train/exp/weights/best.pt")

# # Evaluate the final model's performance on the test set
# # Make sure your data.yaml has a 'test:' key pointing to your test images
# print("\n--- Evaluating on the TEST set ---")
# metrics = final_model.val(data="path/to/your/data.yaml", split='test')

# print("\n--- Test Set Metrics ---")
# print(f"mAP50-95: {metrics.box.map:.4f}")
# print(f"   mAP50: {metrics.box.map50:.4f}")
# print(f"   mAP75: {metrics.box.map75:.4f}")