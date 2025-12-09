Adaptive Image Compression Using Domain-Specific Autoencoders

This project implements an adaptive, domain-aware image compression system using multiple CNN-based autoencoders, each trained on a specific image category.
A classifier automatically selects the correct autoencoder for each input image, resulting in higher reconstruction quality compared to a single generic autoencoder.

The project includes training, evaluation, automatic routing, and visualization for the following four image domains:

Natural Images (STL10)

Satellite Images (EuroSAT)

Cartoon / Synthetic Images (PyTorch FakeData)

Text Images (Custom synthetic dataset)

✨ Features

✔ Four specialized autoencoders (one per image domain)

✔ Image-type classifier to choose the best autoencoder

✔ End-to-end adaptive compression pipeline

✔ PSNR & SSIM evaluation for reconstruction quality

✔ Visualization of original vs reconstructed images

✔ Unified evaluation across all datasets

✔ Model saving and loading support

📂 Project Structure
├── datasets/                 # All datasets downloaded or generated
├── saved_models/             # Trained AEs and classifier weights
├── main.ipynb / script.py    # Full training + evaluation code
├── README.md                 # Documentation
└── requirements.txt          # Dependencies

🧠 Methodology
1. Datasets
Domain	Dataset	Purpose
Natural	STL10	Real-world photography
Satellite	EuroSAT	Aerial remote sensing
Cartoon	FakeData	Synthetic cartoon-like images
Text	Custom PIL-rendered text	OCR-style images

The images are resized to 128×128 and normalized to [0, 1].

🧩 Models
1. Domain-Specific Autoencoders

Each domain uses its own autoencoder with:

Convolutional encoder (downsampling ×4)

Latent bottleneck (512 channels)

Transposed-convolution decoder (upsampling ×4)

Sigmoid activation for 0–1 output

Autoencoder names:

Domain	Autoencoder
Cartoon	ae_cartoon
Natural	ae_natural
Satellite	ae_satellite
Text	ae_text
2. Image Type Classifier

The classifier is a CNN-based architecture with:

4 Conv + ReLU + MaxPool layers

Flatten + Linear → 512 → 4 output classes

Softmax for prediction

Classes:

0 = Cartoon
1 = Natural
2 = Satellite
3 = Text

🔄 Training Workflow
Step 1: Train Autoencoders

Each dataset trains its own autoencoder independently.

AE loss: MSELoss
Optimizer: Adam (lr = 1e-3)
Epochs: 30

Step 2: Train Classifier

Datasets are combined using ConcatDataset.

Classifier loss: CrossEntropyLoss
Optimizer: Adam (lr = 1e-3)
Epochs: 30

Step 3: Adaptive Compression

For every input image:

Classifier predicts domain

Corresponding autoencoder selected

Image encoded → compressed → decoded

Compute reconstruction quality

Display results

📊 Evaluation Metrics

The system computes:

PSNR (Peak Signal-to-Noise Ratio)

Measures pixel-level accuracy (higher = better).

SSIM (Structural Similarity Index)

Measures perceptual similarity (higher = better).

Both are widely used in image compression research.

🖼️ Visualization

The visualize_results() function displays:

Original image

Reconstructed image

PSNR value

SSIM score

Predicted domain

Compression statistics

Example:

Original | Reconstruction
PSNR: 29.8, SSIM: 0.91, Pred: Natural
Compression: 49152 → 8192

▶️ How to Run
1. Install dependencies
pip install -r requirements.txt

2. Run the script
python main.py

3. Trained model files will appear in:
saved_models/

4. View visualizations in the notebook or display windows.
