🚀 Mars Rover Terrain Segmentation

AI-Powered Martian Surface Analysis Using Deep Learning

[![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)](https://python.org)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)](https://pytorch.org)
[![Computer Vision](https://img.shields.io/badge/Domain-Computer%20Vision-purple)](https://github.com/yash200731-design/mars-rover-terrain-segmentation)
[![NASA](https://img.shields.io/badge/Dataset-NASA%20AI4Mars-red)](https://data.nasa.gov/)
[![Vercel](https://img.shields.io/badge/Vercel-Deployed-000000?logo=vercel)](https://vercel.com)
[![Status](https://img.shields.io/badge/Status-Complete-green)](https://github.com/yash200731-design/mars-rover-terrain-segmentation)

🌌 Overview

Mars Rover Terrain Segmentation is a Computer Vision and Deep Learning project that aims to identify and classify different types of Martian terrain from images captured by NASA's Mars rovers.

The project uses the NASA JPL AI4Mars dataset to develop a semantic image segmentation model capable of classifying individual pixels into four terrain categories:

- 🟤 Soil
- ⚫ Bedrock
- 🟡 Sand
- 🪨 Big Rocks

The ultimate goal is to support research into autonomous rover navigation by helping identify different surface materials and potential terrain hazards.

🎯 Problem Statement

Rovers exploring Mars must navigate across different surface conditions. Loose sand, rocks and uneven terrain can create significant mobility challenges.

NASA's Spirit rover became stuck in soft soil in 2009 and could not continue its mission.

This project explores how Deep Learning and semantic segmentation can help identify Martian terrain and provide useful information for future rover navigation systems.

🧠 Project Objectives

- Develop a semantic image segmentation model for Mars rover imagery.
- Classify Martian surfaces into four terrain categories.
- Process and analyze NASA AI4Mars image annotations.
- Implement a U-Net architecture for pixel-level classification.
- Evaluate segmentation performance using standard metrics.
- Build an interactive interface for visualizing predictions.

📂 Dataset

NASA JPL AI4Mars Dataset

Source: https://data.nasa.gov/

The AI4Mars dataset contains Mars rover images annotated by thousands of volunteers through NASA's citizen science initiative.

Initial development uses the Mars Science Laboratory (MSL) Curiosity rover's Mastcam imagery from the AI4Mars Merged Dataset v0.6.

Dataset components:

- Original rover images.
- Human-annotated terrain labels.
- Label mapping information.

🛠️ Tech Stack

Technology| Purpose
---|---
Python| Programming language
PyTorch| Deep Learning framework
U-Net| Segmentation architecture
ResNet34| Feature extraction encoder
OpenCV| Image processing
NumPy| Numerical operations
Pandas| Data analysis
Matplotlib| Visualization
Google Colab| Model development and GPU training
Streamlit| Interactive web application
FastAPI| Vercel Serverless Function API
Vercel| Edge deployment & global CDN
Git & GitHub| Version control

🏗️ Model Architecture

The project uses U-Net with a ResNet34 encoder for semantic segmentation.

U-Net consists of:

1. Encoder – Extracts important visual features from Mars images.
2. Bottleneck – Learns deeper representations.
3. Decoder – Reconstructs spatial information.
4. Segmentation Head – Produces pixel-level terrain predictions.

Model Configuration

Parameter| Configuration
---|---
Architecture| U-Net
Encoder| ResNet34
Input channels| 3
Output classes| 4
Input resolution| 256 × 256
Optimizer| Adam
Loss Function| Cross Entropy Loss (with Ignore Index 255)

⚙️ Project Workflow

1. Dataset collection and exploration.
2. Image and mask preprocessing.
3. Label encoding and handling unlabeled pixels.
4. Training and validation data preparation.
5. U-Net model development.
6. Model training.
7. Performance evaluation.
8. Terrain prediction visualization.
9. Web application development.

📊 Evaluation Metrics

The model will be evaluated using:

- Pixel Accuracy
- Intersection over Union (IoU)
- Mean Intersection over Union (mIoU)
- Dice Coefficient
- Confusion Matrix

📁 Project Structure

```text
mars-rover-terrain-segmentation/
│
├── Mars_Rover_Terrain_Segmentation.ipynb   # Complete 12-Step Google Colab Notebook
│
├── src/                                    # Modular Source Package
│   ├── dataset.py                          # Dataset loader, transforms & pair matcher
│   ├── model.py                            # U-Net ResNet-34 architecture
│   └── utils.py                            # Metrics, color maps, and overlay blend
│
├── checkpoints/
│   └── mars_terrain_model.pth              # Saved model weights
│
├── api/
│   ├── index.py                            # Vercel FastAPI Serverless Function
│   └── requirements.txt                    # Serverless dependencies (< 35MB)
│
├── index.html                              # Web Dashboard (Mission Control HUD)
├── style.css                               # Space theme styling & Mars orange accents
├── app.js                                  # Interactive UI & telemetry controller
├── vercel.json                             # Vercel deployment & routing configuration
│
├── app.py                                  # Python Streamlit Web Application
├── train.py                                # CLI Training Script
├── evaluate.py                             # CLI Evaluation Script
├── requirements.txt                        # Lightweight Vercel dependencies
├── requirements-train.txt                  # Full PyTorch & CUDA training requirements
├── README.md
└── .gitignore
```

🚀 Getting Started

1. Clone the Repository

```bash
git clone https://github.com/yash200731-design/mars-rover-terrain-segmentation.git
cd mars-rover-terrain-segmentation
```

2. Install Dependencies

- **For Model Training (Google Colab / Local GPU):**
  ```bash
  pip install -r requirements-train.txt
  ```

- **For Web Application / Vercel Serverless Function:**
  ```bash
  pip install -r requirements.txt
  ```

3. Open Google Colab

Upload `Mars_Rover_Terrain_Segmentation.ipynb` and connect Google Drive containing the AI4Mars dataset.

4. Train the Model

Execute the notebook cells sequentially to preprocess the data, train U-Net and evaluate model performance.

5. Run the Local Web Applications

- **Streamlit App:**
  ```bash
  streamlit run app.py
  ```

- **Static Space Dashboard:**
  Open `index.html` directly in any web browser!

⚡ Deploying on Vercel

This repository is optimized for **zero-configuration deployment on Vercel**:
- Bundle size optimized to **< 35 MB** (well under Vercel's 250 MB serverless function limit).
- Static assets (`index.html`, `style.css`, `app.js`) are served globally via Vercel's Edge CDN.

### Steps to Deploy on Vercel:
1. Log in to [vercel.com](https://vercel.com).
2. Click **"Add New Project"** and import `yash200731-design/mars-rover-terrain-segmentation`.
3. In settings:
   - **Framework Preset:** `Other` (or leave default)
   - **Root Directory:** `./`
4. Click **Deploy**! The build completes in ~15 seconds.

🌍 Application Features

- Upload Mars rover images or select built-in Curiosity presets.
- Real-time pixel-level terrain segmentation.
- Visualize original, ground-truth, predicted mask, and hazard overlay.
- Interactive alpha-transparency overlay slider.
- Surface composition breakdown (Soil, Bedrock, Sand, Big Rocks).
- Autonomous traversability advisory HUD.

🔮 Future Enhancements

- Improve segmentation accuracy using advanced architectures.
- Experiment with DeepLabV3+ and other segmentation models.
- Integrate terrain traversability analysis.
- Explore obstacle detection and rover path planning.
- Investigate deployment for autonomous navigation research.

⚠️ Limitations

Terrain segmentation is not equivalent to determining whether a route is safe. Actual rover navigation requires additional analysis of terrain geometry, slope, traction, obstacles and rover-specific mobility constraints.

👨‍💻 Author

**Yash Trivedi**  
Computer Science | AI & Machine Learning | Deep Learning | Computer Vision  
GitHub: [https://github.com/yash200731-design](https://github.com/yash200731-design)  
LinkedIn: [https://www.linkedin.com/in/yash-trivedi-3941a43a7/](https://www.linkedin.com/in/yash-trivedi-3941a43a7/)

📜 Acknowledgements

- NASA Jet Propulsion Laboratory (JPL)
- NASA AI4Mars Project
- AI4Mars citizen science volunteers
- PyTorch and open-source Computer Vision community

---

⭐ If you find this project interesting, consider giving the repository a star!
