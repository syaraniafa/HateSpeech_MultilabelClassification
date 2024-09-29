# Hate Speech Multilabel Classification Project (Text Mining)

## Case Study - Final Examination
Hate speech on social media is a disruptive phenomenon, and automating its identification can help relevant authorities take action efficiently. In this project, the objective is to build a multilabel classification model using a Large Language Model (LLM) to identify different types of hate speech on social media. The classification includes labels for Hate Speech (HS), Abusive language, and more specific categories like HS_Individual, HS_Group, HS_Religion, HS_Race, HS_Physical, HS_Gender, and HS_Other.

## Project Overview
This project focuses on building a robust multilabel classification model for identifying hate speech on social media using pretrained Large Language Models (LLMs). The model will be trained to classify the given data into multiple labels representing different forms of hate speech. Various hyperparameter tuning techniques will be employed to optimize the model, and performance metrics such as accuracy, precision, recall, and F1-score will be reported for training, validation, and testing datasets

## Steps Involved
1. **Data Preprocessing**
   The provided hate speech dataset undergoes the following preprocessing steps:
    - Text cleaning (removal of URLs, mentions, special characters)
    -  Tokenization
    -  Label encoding for multilabel classification
    -  Text vectorization using appropriate embedding techniques (e.g., BERT embeddings)
2. **Model Selection: Large Language Model (LLM)**
We utilize a pretrained LLM such as BERT or its variants (e.g., RoBERTa, DistilBERT) for the multilabel classification task. The model is fine-tuned on the hate speech dataset, and the following hyperparameters are tuned:
    - Learning rate
    - Batch size
3. **Hyperparameter Tuning**
   To improve the model’s performance, we conduct hyperparameter tuning. Two key hyperparameters—learning rate and batch size—are tuned using cross-validation or a grid search strategy to identify the optimal settings.
4. **Model Training**
   The LLM is fine-tuned using the training dataset with the tuned hyperparameters. A multilabel classification approach is employed, allowing the model to assign multiple hate speech-related labels to each text.
5. **Model Evaluation**
   After training, the model’s performance is evaluated on the training, validation, and test datasets. The following metrics are computed and analyzed:
    - Accuracy: The overall percentage of correct predictions.
    - Precision: How many of the predicted hate speech labels are relevant.
    - Recall: How many of the actual hate speech labels are correctly identified.
    - F1-Score: The harmonic mean of precision and recall, providing a balance between the two.
   These metrics are presented for each label and provide insight into the model's ability to handle multiple forms of hate speech.

## Results and Analysis
The performance results are presented in terms of accuracy, precision, recall, and F1-score. These metrics help evaluate the model’s effectiveness in identifying different forms of hate speech:
    - Training Data: Results for training the model.
    - Validation Data: Performance on the validation set, which is used to tune the model.
    - Test Data: The final performance on unseen data to assess the model’s generalization capabilities.
The analysis of these metrics provides insight into how well the model captures various forms of hate speech and the trade-offs between precision and recall for each label.

## Conclusion
This project demonstrates how to leverage pretrained Large Language Models (LLMs) to build a multilabel hate speech classification model. The model is fine-tuned and optimized to achieve high performance in classifying multiple forms of hate speech on social media.
