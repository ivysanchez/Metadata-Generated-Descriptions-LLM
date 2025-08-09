# Metadata-Generated-Descriptions-LLM

![](UTA-DataScience-Logo.png)


This repository presents a project that explores the use of various Large Language Models (LLMs) to generate product descriptions using metadata from the 'Amazon product data' dataset. The goal is to evaluate how different LLMs and prompting techniques perform in generating high-quality product descriptions based on the provided title, product type ID, and bullet points.

Dataset: [Amazon Product Data](https://www.kaggle.com/datasets/piyushjain16/amazon-product-data)

## Overview

The goal of this project is to generate compelling and informative product descriptions from structured metadata. The task is framed as a text generation problem where the input is a combination of product title, product type ID, and bullet points, and the output is a product description.

The pipeline includes data loading, data preprocessing (handling missing values and formatting the data), generating descriptions using different LLMs (Baseline, FLAN-T5, GPT-3.5-turbo, and Gemini 1.5 Pro) with both zero-shot and few-shot prompting, and evaluating the generated descriptions using various metrics (BLEU, ROUGE-1, ROUGE-2, BERTScore-F1, and Flesch Reading Ease).

A small sample of 25 rows was used for the LLM generation and evaluation due to computational constraints.


## Summary of Work Done



### Data
* **Dataset**: Amazon Product Data
* **Type**: Tabular data (.csv))
  * **Input**: `product_id`,`title`, `product_type_id`, `bullet_points`
  * **Output**: `description`
* **Size**: Original dataset contains over 2.2 million rows. A cleaned subset of over 1 million rows was created, and a sample of 25 rows was used for LLM experiments.
* **Columns Used:** `title`, `bullet_points`, `description`, 

* **Instances (Train, Test, Validation Split)**: 


***


***verify
#### Preprocessing / Clean Up
*   Selected relevant columns (`title`, `bullet_points`, `description`).
*   Renamed the `description` column to `target_description`.
*   Dropped rows with missing values in any of the selected columns.
*   Ensured all relevant columns were treated as strings.
*   Removed rows with empty strings, only whitespace, or the literal string "nan".
*   Combined `title`, `product_type_id`, and `bullet_points` into a single `input_text` column for LLM input.
*   Sampled 25 rows for LLM generation and evaluation.



### Problem Formulation

* **Input**: 
* **Output**: 
* **Task**: Generate description using LLMs using different prompt methods





### Methodology

*   Baseline Model: A simple concatenation of title, product type ID, and bullet points.
*   LLMs Used:
    *   FLAN-T5 (base)
    *   GPT-3.5-turbo (via OpenAI API)
    *   Gemini 1.5 Pro (via Google Generative AI API)
*   Prompting Techniques:
    *   Zero-shot: Providing the LLM with only the instruction and the input data.
    *   Few-shot: Providing the LLM with the instruction, a few examples of input-output pairs, and the input data.
* Metrics: BLEU, ROUGE-1, ROUGE-2, BERTScore-F1, and Flesch Reading Ease




***
### Conclusions

Model Comparison (With Augmentation - ResNet50)
* The models were evaluated based on the selected metrics, with a focus on BERTScore-F1, ROUGE-1, and ROUGE-2 for comparing the quality and similarity of the generated descriptions to the target descriptions.
* **Results**:
  * 



### Future Work
*  Experimenting with different hyperparameters for each model (learning rate, batch size, number of epochs) to optimize their performance.
* Exploring more sophisticated prompting techniques, such as Chain-of-Thought prompting or incorporating more diverse few-shot examples, to guide the models toward generating better descriptions.
* Using different model architectures that might be better suited for text generation from structured data.
* Combining the predictions of multiple models to potentially leverage their individual strengths and improve overall performance.
* Increasing the size and diversity of the training data since only a small sample was used (25).


## How to reproduce results

To reproduce the results of this project, follow these steps:

1. Download the dataset: Download the "Amazon product Data" dataset from Kaggle. The notebook contains code to downlload the dataset using 'kagglehub'.
2. Open notebooks in the following order:
    - `Data_Loader.ipynb`
    - `Data_Preprocessing.ipynb`
    - `Base_model.ipynb`
    - `LLM_Generated_Descriptions.ipynb`
    - `Compare_Evaluation.ipynb`

3. Run the code cells: Execute the code cells/notebooks in a single notebook sequentially to reproduce the results.
   
**Resources:**
* Google Colab: Use Google Colab or Jupyter Notebook to run the code and leverage its computational resources.
* Kaggle: Access the dataset and potentially explore other related datasets.

### Overview of files in repository

| File Name                         | Description                                                                 |
|----------------------------------|-----------------------------------------------------------------------------|
| `Data_Loader.ipynb`               | Loads the dataset and neccessary libraries                                            |
| `Data_Preprocessing.ipynb`           | Preparing data for LLM use                                             |
| `Base_Model.ipynb` | Creating a baseline model of generated descriptions                                 |
| `LLM_Generated_Descriptions.ipynb`      | Using 3 different LLMs for text generation                              |
| `Compare_Evaluation.ipynb`        | Taking a closer look at metrics for comparison                                        |
| `Final_LLM_Generated.ipynb`     | Full pipeline from data loading to evaluation                       



### Software Setup
* Required Packages: This project uses the following Python packages:
  * Standard Libraries:
  * * pandas
   * numpy
   * matplotlib
   * seaborn
   * re
   
* Additional Libraries:
   * kagglehub (For downloading the dataset from Kaggle)
   * tranformers
   * datasets
   * accelerate
   * evaluate
   * bert_score
   * nltk
   * rouge_score
   * textstat
   * openai
   * google-generativeai






## **Citations**
