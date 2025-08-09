# Metadata-Generated-Descriptions-LLM

![](UTA-DataScience-Logo.png)


This repository presents a project that explores the use of various Large Language Models (LLMs) to generate product descriptions using metadata from the ***Amazon Product Data*** dataset. The goal is to evaluate how different LLMs and prompting techniques perform in generating high-quality product descriptions based on the provided title, product type ID, and bullet points.

Dataset: [Amazon Product Data](https://www.kaggle.com/datasets/piyushjain16/amazon-product-data)

## Overview

The aim of this project is to generate compelling and informative product descriptions from structured metadata. The task is framed as a text generation problem where the input is a combination of product title, product type ID, and bullet points, and the output is a product description. The pipeline includes data loading, data preprocessing (handling missing values and formatting the data), generating descriptions using different LLMs (FLAN-T5, GPT-3.5-turbo, and Gemini 1.5 Pro) with both zero-shot and few-shot prompting, and evaluating the generated descriptions using various metrics (BLEU, ROUGE-1, ROUGE-2, BERTScore-F1, and Flesch Reading Ease) with a focus on the BERTScore-F1.

Due to computational limits, only 25 rows from a cleaned 1M+ row subset were used for generation and evaluation.
* Best overall: GPT-3.5-turbo (few-shot), with the highest BERTScore-F1 (0.8407).
* Close Competitors: Other models scored between 0.810 and 0.837.
* Most readable: Gemini 1.5 Pro (few-shot), with the highest Flesch score (~54.6).
* Best phrasing similarity: FLAN-T5 (few-shot) with strong BLEU and ROUGE-2.
  
Few-shot prompting consistently outperformed zero-shot across all models.



## Summary of Work Done


### Data
* **Dataset**: Amazon Product Data
* **Type**: Tabular data (.csv)
  * **Input**: `product_id`,`title`, `product_type_id`, `bullet_points`
  * **Output**: `description`
* **Size**: Original dataset contains over 2.2 million rows. A cleaned subset of over 1 million rows was created, and a sample of 25 rows was used for LLM experiments.
* **Columns Used:** `title`, `product_id_type`,`bullet_points`, `description`, 
* **Instances (Train, Test, Validation Split)**: N/A (LLM zero-shot and few-shot setting with small data sample of 25)


***


### Preprocessing
*   Selected relevant columns (`title`, `bullet_points`, `description`).
*   Renamed the `description` column to `target_description`.
*   Dropped rows with missing values in any of the selected columns.
*   Ensured text columns were strings.
*   Removed empty, whitespace-only, or "nan" strings.
*   Combined `title`, `product_type_id`, and `bullet_points` into a single `input_text` column for LLM input.
*   Sampled 25 rows for LLM generation and evaluation.



### Problem Formulation

* **Input**: Structured product metadata (`title`, `product_type_id`, `bullet_points`)
* **Output**: Natutral language product description
* **Task**: Generate high-quality product descriptions with different prompting strategies



### Methodology

*   Baseline Model: A simple concatenation of `title`, `product type ID`, and `bullet points`. This method serves as a non-LLM benchmark to compare against generative model outputs.
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

* **FLAN-T5 Model:** Performed well with few-shot prompting, achieving high BLEU and ROUGE-2, indicating strong phrasing similarity, but had low readability (Flesch score).
* **GPT-3.5 Model:** Few-shot prompting yielded strong overall performance with high ROUGE-1, BERTScore-F1, and good readability.
* **Gemini 1.5 Pro Model:** Lower BLEU and ROUGE-2 but highest readability, producing the most user-friendly text.

| Model                       | BLEU   | ROUGE-1 | ROUGE-2 | BERTScore-F1 | Flesch   |
| :-------------------------- | :----- | :------ | :------ | :----------- | :------- |
| Gpt-3.5-turbo (few-shot)    | 0.0634 | 0.2745  | 0.0845  | 0.8407       | 50.18    |
| Gemini 1.5 Pro (few-shot)   | 0.0402 | 0.2640  | 0.0747  | 0.8367       | 54.60    |
| Gpt-3.5-turbo (zero-shot)   | 0.0464 | 0.2635  | 0.0704  | 0.8356       | 47.72    |
| Baseline                    | 0.1006 | 0.2889  | 0.1271  | 0.8336       | 33.60    |
| FlanT5 (few-shot)           | 0.0795 | 0.2608  | 0.1134  | 0.8323       | 35.02    |
| FlanT5 (zero-shot)          | 0.0715 | 0.2447  | 0.0845  | 0.8247       | 33.54    |
| Gemini 1.5 Pro (zero-shot)  | 0.0199 | 0.2086  | 0.0531  | 0.8110       | 43.47    |

* Overall for generating descriptions the GPT-3.5-Turbo (few-shot) model was the best performing in terms of the BERTScore- F1. However other models were very close in score such as the Gemini 1.5 Pro (few-shot) model.
  <img width="981" height="490" alt="Unknown-27" src="https://github.com/user-attachments/assets/a3e5407e-e4a4-46b7-b2f3-f7dc3d421326" />

* The few-shot prompting technique generally led to better performance compared to zero-shot for the LLMs.


### Future Work
* Tune hyperparameters (learning rate, batch size, epochs)
* Explore advanced prompting (Chain-of-Thought, diverse few-shot examples)
* Test additional architectures suited for structured-to-text generation
* Use model ensembling for improved outputs
* Increase training sample size beyond 25 rows


## How to reproduce results

To reproduce the results of this project, follow these steps:

1. Download the dataset: Download the "Amazon Product Data" dataset from Kaggle (code provided in `Data_Loading` notebook via kagglehub).
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
| `Data_Loader.ipynb`               | Loads dataset and required libraries                                           |
| `Data_Preprocessing.ipynb`           | Cleans and formats data                                             |
| `Base_Model.ipynb` | Baseline description generation                                |
| `LLM_Generated_Descriptions.ipynb`      | Runs LLM-based generation                              |
| `Compare_Evaluation.ipynb`        | Compares models using metrics                                        |
| `Final_LLM_Generated.ipynb`     | Full pipeline from data loading to evaluation                       



### Software Setup
* Required Packages: This project uses the following Python packages:
  * Standard Libraries:
    * pandas
    * numpy
    * matplotlib
    * seaborn
   
  * Additional Libraries:
    * kagglehub (For downloading the dataset from Kaggle)
    * tranformers
    * accelerate
    * evaluate
    * bert_score
    * nltk
    * rouge_score
    * openai
    * google-generativeai


## **Citations**
* Piyushjain16. (2023, April 25). Amazon Product Dataset. Kaggle. https://www.kaggle.com/datasets/piyushjain16/amazon-product-data 
