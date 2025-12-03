YOLOv8 Dataset Manager & Annotation Tool
A modern web application built with React, TypeScript, and Node.js to visualize, manage, analyze, and edit YOLOv8 datasets with an enhanced user experience.
Features

File Manager Grid View:

View dataset splits (train, valid, test) in a responsive, virtualized grid for optimal performance.
Toggle between annotated and background images.
Select images to enter the annotation editor.


Advanced Image Management:

Flag Images: Mark images for review, saved to flagged_images.txt.
Delete Images: Remove images and their labels with confirmation.


Enhanced Annotation Editor:

Draw, transform, and delete bounding boxes using Fabric.js.
Real-time class selection per box, with undo/redo and zoom support.
Immediate save functionality with instant grid view updates.


Dataset Analytics:

Visualize class distribution, image dimensions, and bounding box sizes.
Display total images, bounding boxes, and average boxes per image.


Model Testing:

Test YOLOv8 .pt models on selected images with real-time results.



Setup and Installation

Clone the repository or save the files into a directory.

Install Node.js dependencies:
npm install


Install Python dependencies for YOLOv8 inference (in a virtual environment):
pip install pyyaml opencv-python-headless pillow torch ultralytics



How to Run

Start the backend server:
node server.js


Open index.html in a browser or serve it via a static file server (e.g., npx serve).

Provide the data.yaml path when prompted by the browser.


Dependencies

Frontend: React 18.2.0, Fabric.js 5.3.0, Axios 1.4.0, react-window 1.8.9, Tailwind CSS 2.2.19
Backend: Express, js-yaml, canvas, ultralytics, node-cache

Notes

Ensure the data.yaml file follows the YOLOv8 format.
The backend assumes YOLOv8 .pt models are available in the specified directory.
For optimal performance, run the backend on a machine with sufficient resources for YOLO inference.
