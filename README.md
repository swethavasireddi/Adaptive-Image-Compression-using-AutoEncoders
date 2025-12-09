Adaptive Image Compression Using Domain-Specific Autoencoders

This repository contains an adaptive image compression framework that uses multiple domain-specialized CNN autoencoders along with a classifier-based routing mechanism.
The objective is to select the best autoencoder for each image type, improving reconstruction quality (PSNR/SSIM) compared to a single shared model.

The project supports four distinct image domains:

🏞️ Natural Images (STL10)

🛰️ Satellite Images (EuroSAT)

🎨 Cartoon/Synthetic Images (FakeData)

✍️ Text Images (Synthetic PIL dataset)

The system automatically predicts the image type and applies the corresponding compression model.

🚀 Features

🔥 Four independent autoencoders, each trained per domain

🧠 CNN classifier to auto-select the correct autoencoder

📉 Adaptive compression–reconstruction pipeline

📊 Evaluation using PSNR + SSIM

🖼️ Side-by-side visualization of original vs reconstructed images

💾 Model saving & loading (PyTorch)

⚙️ Designed for GPU (CUDA) or CPU mode

📁 Dataset

The project uses four datasets:

Domain	Dataset	Source
Natural	STL10	torchvision.datasets
Satellite	EuroSAT	torchvision.datasets
Cartoon / Synthetic	FakeData	torchvision.datasets
Text	Synthetic	PIL-generated text images

All images are automatically resized to 128×128 RGB.

Directory structure:

datasets/
    natural/
    satellite/
saved_models/
main.ipynb or main.py

🧠 Model Overview
1️⃣ Autoencoders

Each image domain has its own CNN-based Residual Autoencoder:

Encoder: 4 convolutional blocks

Latent space: 512 feature channels

Decoder: 4 transposed-convolution blocks

Output: 128×128 reconstructed RGB image

2️⃣ Image-Type Classifier

The classifier distinguishes between 4 categories:

0 — Cartoon  
1 — Natural  
2 — Satellite  
3 — Text  


It consists of:

4× Conv + ReLU + MaxPool blocks

Fully-connected classifier head

Softmax output

3️⃣ Adaptive Compression

Pipeline:

Input image → classifier predicts domain

Select corresponding autoencoder

Encode → compress latent representation

Decode → reconstructed image

Compute PSNR & SSIM

🛠 Usage
1️⃣ Clone the repository
git clone https://github.com/yourusername/adaptive-image-compression.git
cd adaptive-image-compression
pip install -r requirements.txt

2️⃣ Train the models

Run:

python main.py


or open training.ipynb (if provided) and run all cells.

This will:

Download datasets

Train 4 autoencoders

Train classifier

Save all models in saved_models/

3️⃣ Run inference

To test adaptive compression on new images:

Use the adaptive_compress_recon() function in the script

Or open inference.ipynb

The system automatically:

Predicts the image domain

Routes the image to the best autoencoder

Outputs reconstruction + metrics
