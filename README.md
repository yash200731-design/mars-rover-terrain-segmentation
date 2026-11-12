🚀 Mars Rover Terrain Segmentation

AI-Powered Martian Surface Analysis Using Deep Learning

"Python" (https://img.shields.io/badge/Python-3.10+-blue?logo=python)
"PyTorch" (https://img.shields.io/badge/Framework-PyTorch-orange?logo=pytorch)
"Computer Vision" (https://img.shields.io/badge/Domain-Computer%20Vision-purple)
"NASA" (https://img.shields.io/badge/Dataset-NASA%20AI4Mars-red)
"Status" (https://img.shields.io/badge/Status-In%20Development-yellow)

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
Architecture| U-Net
Encoder| ResNet34
Input channels| 3
Output classes| 4
Input resolution| 256 × 256
Optimizer| Adam
Loss Function| Cross Entropy Loss

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

Actual evaluation results will be added after model training.

📁 Project Structure

mars-rover-terrain-segmentation/
│
├── notebooks/
│   └── Mars_Rover_Terrain_Segmentation.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model.py
│   ├── train.py
│   └── evaluate.py
│
├── models/
│   └── mars_terrain_model.pth
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore

The structure will be updated as development progresses.

🚀 Getting Started

1. Clone the Repository

git clone https://github.com/YOUR_USERNAME/mars-rover-terrain-segmentation.git

cd mars-rover-terrain-segmentation

2. Install Dependencies

pip install -r requirements.txt

3. Open Google Colab

Upload the project notebook and connect Google Drive containing the AI4Mars dataset.

4. Train the Model

Execute the notebook cells sequentially to preprocess the data, train U-Net and evaluate model performance.

5. Run the Web Application

streamlit run app.py

🌍 Expected Application Features

- Upload Mars rover images.
- Generate pixel-level terrain segmentation.
- Visualize original and predicted images.
- Display terrain classification masks.
- Show terrain class distribution.

🔮 Future Enhancements

- Improve segmentation accuracy using advanced architectures.
- Experiment with DeepLabV3+ and other segmentation models.
- Integrate terrain traversability analysis.
- Explore obstacle detection and rover path planning.
- Investigate deployment for autonomous navigation research.

⚠️ Limitations

Terrain segmentation is not equivalent to determining whether a route is safe. Actual rover navigation requires additional analysis of terrain geometry, slope, traction, obstacles and rover-specific mobility constraints.

👨‍💻 Author

Yash Trivedi

Computer Science | AI & Machine Learning | Deep Learning | Computer Vision

GitHub: https://github.com/yash200731-design

LinkedIn: https://www.linkedin.com/in/yash-trivedi-3941a43a7/

📜 Acknowledgements

- NASA Jet Propulsion Laboratory (JPL)
- NASA AI4Mars Project
- AI4Mars citizen science volunteers
- PyTorch and open-source Computer Vision community

---

⭐ If you find this project interesting, consider giving the repository a star!
