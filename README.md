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
### Conclusions/Results

* The models were evaluated based on the selected metrics, with a focus on BERTScore-F1, ROUGE-1, and ROUGE-2 for comparing the quality and similarity of the generated descriptions to the target descriptions.

| Model                       | BLEU   | ROUGE-1 | ROUGE-2 | BERTScore-F1 | Flesch   |
| :-------------------------- | :----- | :------ | :------ | :----------- | :------- |
| Gpt-3.5-turbo (few-shot)    | 0.0634 | 0.2745  | 0.0845  | 0.8407       | 50.18    |
| Gemini 1.5 Pro (few-shot)   | 0.0402 | 0.2640  | 0.0747  | 0.8367       | 54.60    |
| Gpt-3.5-turbo (zero-shot)   | 0.0464 | 0.2635  | 0.0704  | 0.8356       | 47.72    |
| Baseline                    | 0.1006 | 0.2889  | 0.1271  | 0.8336       | 33.60    |
| FlanT5 (few-shot)           | 0.0795 | 0.2608  | 0.1134  | 0.8323       | 35.02    |
| FlanT5 (zero-shot)          | 0.0715 | 0.2447  | 0.0845  | 0.8247       | 33.54    |
| Gemini 1.5 Pro (zero-shot)  | 0.0199 | 0.2086  | 0.0531  | 0.8110       | 43.47    |


* FLAN-T5 Model: This model performed reasonably well, particularly with few-shot prompting, achieving the high BLEU and ROUGE-2 scores. This suggests it was good at generating descriptions with similar phrasing to the original text. However, its Flesch Reading Ease score was low, indicating that its generated text might be harder to read.
 * OpenAI GPT-3.5 Model: This model, especially with few-shot prompting, demonstrated strong performance across several metrics, including high ROUGE-1 and BERTScore-F1 scores. It also produced descriptions with relatively high readability (Flesch Reading Ease). The example outputs showed that GPT-3.5 generated more detailed and human-like descriptions compared to FLAN-T5.
* Gemini 1.5 Pro Model: While its BLEU and ROUGE-2 scores were low in the few-shot setting, Gemini 1.5 Pro achieved a high Flesch Reading Ease score, suggesting it generated the most readable descriptions. The example outputs showed that Gemini 1.5 Pro also produced detailed and well-structured
* Based on the BERTScore-F1, GPT-3.5-turbo (few-shot) and Gemini 1.5 Pro (few-shot) models performed the best, indicating higher semantic similarity to the original descriptions. The few-shot prompting technique generally improved performance for the LLMs.



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
    * pandas
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
* Piyushjain16. (2023, April 25). Amazon Product Dataset. Kaggle. https://www.kaggle.com/datasets/piyushjain16/amazon-product-data 
