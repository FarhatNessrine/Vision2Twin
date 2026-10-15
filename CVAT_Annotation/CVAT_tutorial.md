# 🛠️ Installation Environment Setup: WSL, Ubuntu, Docker, CVAT, and Nuclio

To support a robust and reproducible environment for AI-based computer vision workflows, we configured a modular system using Windows Subsystem for Linux (WSL), Ubuntu, Docker, CVAT, and Nuclio. This setup enables seamless development and deployment of annotation and inference services on a Windows host while leveraging Linux-native tools and containers.

## 📦 System Requirements

- Windows 10/11 with WSL2 support
- Docker Desktop
---

## 1. Windows Subsystem for Linux (WSL) & Ubuntu

We began by enabling the Windows Subsystem for Linux (WSL), which allows Linux distributions to run directly on Windows without the need for a virtual machine. WSL version 2 was selected for its enhanced performance and compatibility with Docker. The following command was executed in an elevated PowerShell terminal to install WSL:
wsl --install
This command installs both the WSL 2 engine and Ubuntu by default. If needed, the available distributions can be listed and manually installed with:
wsl --list --online
wsl --install -d Ubuntu
After installation, Ubuntu can be launched from the Start Menu, and the user is prompted to configure a Unix username and password.

## 2. Docker Installation and Configuration
Docker was used to manage containerized services such as CVAT and Nuclio. We installed Docker Desktop for Windows, ensuring that WSL 2 integration was enabled during setup. After installation, Docker is accessible both from Windows and within the Ubuntu terminal. To verify the installation, the following command can be used:
docker --version
docker run hello-world
If Docker Compose is not included, it can be installed in Ubuntu with:
sudo apt update
sudo apt install docker-compose
Docker Desktop must be running whenever using containers in WSL.

## 3. CVAT (Computer Vision Annotation Tool) Deployment
To facilitate dataset annotation, we deployed CVAT using Docker Compose. First, the official repository was cloned from GitHub:
git clone https://github.com/opencv/cvat.git
cd cvat
Inside the CVAT folder, the following command launches the system:
docker compose -f docker-compose.yml up -d
Once initialized, the CVAT web interface becomes available at http://localhost:8080. The first user account is typically created manually via the interface or using CVAT’s user management scripts.

For more details: https://docs.cvat.ai/docs/administration/basics/installation/

## 4. Nuclio Serverless Framework Installation
To enable automatic AI inference within CVAT, we deployed Nuclio, a serverless function platform optimized for data science and machine learning tasks. Nuclio runs entirely via Docker:
docker compose -f docker-compose.yml -f docker-compose.dev.yml -f components/serverless/docker-compose.serverless.yml up -d --build

This command launches the Nuclio Dashboard on http://localhost:8070, where users can create projects and deploy serverless functions (e.g., YOLO-based inference or segmentation tools). Functions can be created from local code, Git repositories, or predefined templates.

For more details: https://docs.cvat.ai/docs/administration/advanced/installation_automatic_annotation/

## 🧠 AI Inference Functions Installation

Once Nuclio is running, we deployed serverless functions such as **SAM**, **MILTracker**, and **YOLOv7** to enable automatic annotation and object tracking directly in CVAT, using the following commands:
./serverless/deploy_gpu.sh ./serverless/pytorch/facebookresearch/sam/nuclio/
./serverless/deploy_cpu.sh ./serverless/onnx/WongKinYiu/yolov7/nuclio/

