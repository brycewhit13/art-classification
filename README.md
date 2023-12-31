# Deepfake or Real?

## Microservice ML Endpoint for Artwork Classification

### Description

Is this photo AI Generated or real art? This project is a powerful and innovative application that utilizes Azure services to build a microservice-based machine learning endpoint for classifying artwork into two categories: AI-generated or real. Leveraging cutting-edge computer vision techniques, this project provides a seamless and efficient solution for art enthusiasts, galleries, and AI researchers to automatically distinguish between AI-generated masterpieces and authentic human-created artwork. The core of the system is a highly accurate machine learning model, which was trained using a meticulously curated dataset comprising a diverse collection of AI-generated and real artwork. The model has undergone rigorous testing and optimization, ensuring exceptional accuracy in classifying a wide range of visual artworks. This project offers a user-friendly API interface, allowing developers and researchers to easily integrate the ML endpoint into their applications, enabling real-time predictions. With detailed installation and deployment instructions, contributors can effortlessly deploy the microservice to Azure cloud, making it accessible to a global audience. We welcome contributions from the open-source community, fostering continuous improvement and enhancing the system's capabilities. By providing this powerful tool to the art community, we aim to promote creativity, preserve artistic authenticity, and facilitate AI research in the realm of computer-generated art. 

This project was developed by Bryce Whitney, Andrew Bonafede, and Bruno Valan as the final deliverable for Duke AIPI 561 course in MLOPs.

### Table of Contents

1. [Local Installation](#local-installation)
2. [Web Usage](#web-usage)
3. [Workflows](#workflows)

### Local Installation

To get started with the Microservice ML Endpoint for Artwork Classification on your local machine, follow these steps:

1. Clone the repository from GitHub.
2. Navigate to the project directory.

```bash
git clone https://github.com/brycewhit13/art-classification.git
cd art-classification
```

3. Install the required dependencies using pip.

```bash
pip install -r requirements.txt
```


4. Train and run the model using the provided script.

```bash
python scripts/train_model.py
```


This will initiate the model training process, leveraging the curated dataset to create an accurate artwork classification model. Once the training is complete, the model will be ready for predictions. You can then interact with the ML endpoint by making API requests and obtaining AI-generated or real artwork predictions with ease.

With these simple steps, you can quickly set up the project on your local machine and explore its capabilities. Feel free to experiment, modify, and contribute to this exciting project!


### Web Usage

In addition to be able to run this app locally. The Microservice ML Endpoint for Artwork Classification is also accessible through a web application deployed on Azure at this [link](https://art-classification-webapp2.azurewebsites.net/). Leveraging Azure's robust capabilities, the web app provides a user-friendly interface for users to upload images and receive instant predictions on whether the artwork is AI-generated or real. By utilizing common ML endpoint techniques in Azure, such as containerization and scalable cloud infrastructure, the application ensures high availability and performance, catering to a wide range of users seeking quick and accurate artwork classifications in real-time.

### Workflows

The project also incorporates continuous integration (CI) to streamline the development and deployment process through GitHub Actions. The workflow is defined in the `.github/workflows/main_art-classification-code.yml` file. This workflow is triggered automatically on each push to the `main` branch and can also be manually executed using the GitHub Actions workflow_dispatch event. The CI workflow consists of two jobs: `build` and `deploy`.

The `build` job runs on an Ubuntu environment and handles setting up the Python environment, installing project dependencies specified in `requirements.txt`, and creating a virtual environment to isolate dependencies. Upon successful completion, the build job uploads the project artifacts (excluding the virtual environment) using the `actions/upload-artifact` action.

The `deploy` job is triggered when the `build` job completes successfully and is responsible for deploying the application to Azure Web App. It runs on another Ubuntu environment and first downloads the uploaded artifacts using the `actions/download-artifact` action. Subsequently, it utilizes the Azure Web Apps Deploy action (`azure/webapps-deploy@v2`) to deploy the Python app to the specified Azure Web App named 'art-classification-code' in the 'Production' slot. The deployment is facilitated using the provided Azure publish profile stored securely in the GitHub repository secrets.

By leveraging this CI workflow, we were confidently able to push changes to the `main` branch, knowing that the CI process will automatically build, test, and deploy the latest version of the application to the Azure Web App environment, ensuring a seamless and efficient deployment process.