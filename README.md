## Ischaemia Detection: Medical Image Classification
Binary classification of medical images (Ischaemia) using deep learning — PyTorch & TensorFlow/Keras implementations.

### Project overview
This project focuses on the binary classification of medical images to detect Ischaemia. Two complete deep-learning pipelines are provided:
A PyTorch implementation (baseline CNN).
A TensorFlow / Keras implementation (deeper CNN, best performance).
Both pipelines include data loading, augmentation, training loops, validation, and inference on random test samples.

### Objective
Given a medical image, classify it as:

0 — Negative (no Ischaemia)
1 — Positive (Ischaemia detected)

### Dataset
The dataset contains two folders of images:
text
Ischaemia/
├── Aug-Negative/     # Negative samples (label = 0)
│   └── *.jpg
└── Aug-Positive/     # Positive samples (label = 1)
    └── *.jpg
Dataset statistics
Split	Negative	Positive	Total
Train + Val	~2961	~2961	~5922
Test	~987	~987	~1974
Total	~3948	~4935	~9870
Splits
train : 60 %
validation : 20 %
test : 20 %
Stratified splits are used to preserve class balance.

### Project structure
text
ischaemia-detection/
│
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_pytorch_cnn.ipynb           # PyTorch baseline CNN
│   └── 02_tensorflow_cnn.ipynb        # TensorFlow/Keras CNN (best)
│
├── data/
│   └── Ischaemia/
│       ├── Aug-Negative/
│       └── Aug-Positive/
│
├── models/
│   ├── pytorch_model.pth
│   └── tensorflow_model.h5
│
└── assets/
    └── sample_predictions.png


### Approaches implemented
🔹 Pipeline 1 — 01_pytorch_cnn.ipynb
Custom CNN in PyTorch
Layer	Details
Conv2d (3 → 32)	kernel 5×5, padding 2, ReLU
MaxPool2d	2×2
Conv2d (32 → 64)	kernel 5×5, padding 2, ReLU
MaxPool2d	2×2
Dropout (0.5)	
FC (→ 128)	ReLU
FC (→ 1)	Sigmoid
Loss : BCELoss
Optimizer : SGD (lr = 1e-3)
Epochs : 10
Input size : 128 × 128 × 3
Batch size : 32

🔹 Pipeline 2 — 02_tensorflow_cnn.ipynb
Deeper CNN in TensorFlow/Keras — the best-performing model.
Architecture (best run, ~92 % test accuracy)
Layer	Details
Conv2D (32 filters)	kernel 7×7, ReLU
MaxPooling2D	2×2
Conv2D (32 filters)	kernel 5×5, ReLU
MaxPooling2D	2×2
Conv2D (64 filters)	kernel 3×3, ReLU
MaxPooling2D	2×2
Flatten	
Dense (256)	ReLU
Dense (1)	Sigmoid
Loss : BinaryCrossentropy
Optimizer : Adam (lr = 1e-4)
Epochs : 20
Input size : 128 × 128 × 3
Batch size : 32

An alternative variant with Dropout (0.3) after the first dense layer reaches ~90 % test accuracy.

### Evaluation
Metrics tracked during training and evaluation:
Binary accuracy
Binary cross-entropy loss
Train vs. validation curves
Reported on the held-out test set.

### Installation
bash
git clone https://github.com/<your-username>/ischaemia-detection.git
cd ischaemia-detection
python -m venv venv
source venv/bin/activate        # Linux / macOS
venv\Scripts\activate         # Windows
pip install -r requirements.txt
requirements.txt
txt
torch>=2.0
torchvision
tensorflow>=2.13
numpy
pandas
scikit-learn
matplotlib
seaborn
opencv-python
Pillow
tqdm

### Usage
1. Prepare the data
Place the Ischaemia/ folder (with Aug-Negative/ and Aug-Positive/) under data/.

2. Run a notebook
PyTorch baseline

bash
jupyter notebook notebooks/01_pytorch_cnn.ipynb
TensorFlow CNN (best)

bash
jupyter notebook notebooks/02_tensorflow_cnn.ipynb
3. Inference on a random test image
python
image = test_dataset[3]
prediction = model.predict(np.expand_dims(image, axis=0))
binary_label = 1 if prediction >= 0.5 else 0
print(f"Prediction: {binary_label} | Ground truth: {test_label[3]}")

### Results
Framework	Model	Train acc.	Val acc.	Test acc.	Test loss
PyTorch	Custom CNN	~96.4 %	~91.0 %	~92.0 %	~0.24
TensorFlow	Deep CNN (best)	~95.0 %	~91.0 %	~92.0 %	~0.24
TensorFlow	CNN + Dropout	~93 %	~90 %	~90 %	—
Observations
The TensorFlow implementation with 3 convolutional blocks and a 256-unit dense layer reaches the best trade-off between accuracy and generalization.
Training accuracy (~96 %) is higher than validation/test (~91-92 %), which suggests moderate overfitting. Mitigations:
Data augmentation (flips, rotations, brightness jitter)
Early stopping
Dropout in the dense layers
Regularization (L2)
The PyTorch baseline trains fast but plateaus earlier without BatchNorm / augmentation.

### Perspectives
□ Transfer learning with ResNet50, EfficientNet, or DenseNet (pretrained on ImageNet)
□ Data augmentation : rotation, zoom, shift, flip, brightness
□ Cross-validation (k-fold) for a more robust accuracy estimate
□ Class activation maps (Grad-CAM) for interpretability — critical in a medical context
□ Ensembling (PyTorch + TensorFlow predictions)
□ Export to ONNX / TorchScript for deployment

### Author
Kawtar HOUSNI  — Student / Data Scientist, 
Abdellah TIAIR - Student / Computer Science and Electronics for Embedded Systems
