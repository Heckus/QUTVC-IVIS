import sys
import json
from ultralytics import YOLO
from PIL import Image
import os

def run_inference(model_path, image_path):
    try:
        # Check if the image path exists
        if not os.path.exists(image_path):
            print(json.dumps({"error": f"Image path does not exist: {image_path}"}))
            sys.exit(1)
            
        model = YOLO(model_path)
        results = model.predict(image_path, conf=0.25, verbose=False)
        
        # Get the first result object
        result = results[0]
        
        # Get original image dimensions
        img = Image.open(image_path)
        img_width, img_height = img.size

        # Prepare data for JSON output
        output = {
            "boxes": result.boxes.xyxy.cpu().numpy().tolist(),
            "class_ids": result.boxes.cls.cpu().numpy().tolist(),
            "width": img_width,
            "height": img_height
        }
        
        print(json.dumps(output))

    except Exception as e:
        print(json.dumps({"error": str(e)}))
        sys.exit(1)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(json.dumps({"error": "Usage: python run_yolo.py <model_path> <image_path>"}))
        sys.exit(1)
    
    model_path_arg = sys.argv[1]
    image_path_arg = sys.argv[2]
    run_inference(model_path_arg, image_path_arg)