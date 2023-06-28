# Machine Learning Classifier for Color and Texture Prediction
This README provides a detailed overview of the machine learning classifier developed to predict color and texture in images. The aim of this work is to apply machine learning techniques to a recent open-ended research problem.

## Project Objectives
The primary objectives of this project are as follows:

- Select and Train a Suitable Classification Model: The project focuses on selecting an appropriate machine learning classification model that can effectively predict color and texture attributes in images. Various models will be explored and evaluated to determine the best-performing one.

- Evaluate and Compare Model Performance: The performance of different machine learning models will be assessed and compared to identify the model with the highest accuracy and predictive capability. Evaluation metrics such as precision, recall, and F1 score will be used to measure the performance.

- Technical Reporting and Justification: A comprehensive technical report will be prepared, explaining the methodology, model selection process, and the justification behind the chosen approach. The report will provide a detailed analysis of the results and highlight the significance of the findings.

## Dataset
The dataset used for this project is a subset of the GQA (Visual Genome Question Answering) dataset, specifically focused on learning about relations. The dataset has been preprocessed and divided into two subsets: the train dataset and the test dataset.

- Train Dataset: This subset serves as the training data for the machine learning classifier. It contains annotated images with labeled boundary boxes, along with relevant attributes and relations.

- Test Dataset: This subset serves as real-world data with unknown targets. It consists of images without annotated labels, and the machine learning classifier will be applied to predict the color and texture attributes of these images.

## Getting Started
To reproduce the results and run the machine learning classifier, follow these steps:

- Clone the project repository from GitHub.
```bash
git clone https://github.com/DavidIkechi/machine_learning.git
```
- Install the required dependencies and libraries specified such as seaborn, matplotlib, numpy, scikit-learn, and pandas.
```bash
pip install seaborn
pip install matplotlib
pip install numpy
pip install scikit-learn
pip install pandas
```
- run the main.py file using
  ```bash
  python main.py
  ```

## Conclusion
This project aims to develop a machine learning classifier for color and texture prediction. By selecting and training a suitable model, evaluating its performance, and providing a detailed technical report, we can gain insights into the effectiveness of machine learning algorithms for predicting color and texture attributes in images.
