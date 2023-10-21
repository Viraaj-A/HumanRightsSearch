# Human Rights Search
The Human Rights Search (HRS), a Flask/Python webapp, applies human centred design principles to the European Court of Human Rights (ECHR) to make human rights accessible to people that lack financial resources, do not have professional legal experience, or face accessibility issues.

The specific Human Rights Search function, occurs after a user utilises the Issue Identifier and the Predictor.

Overall, the HRS utilises human-centered design principles and machine learning tools to create a new method of accessing typically inaccessible material, by:
- Creating summaries of all legal cases (HSR Summaries);
- Creating and implementing advanced search filters, where users will be able to filter by labels by semantic searches (Advanced Filtering); and
- Present key aspects of cases (Alternative Display).

The HSR is in it's beta phase and is currently operating to show its potential as a proof of concept. 

The current beta of this app can be accessed here: https://walrus-app-9dcb8.ondigitalocean.app/

## Search 

The current Search is built using FAISS indexing library for efficient similarity search and tested across three embedding systems:
-	Word embedding - distilbert-base-uncased;
-	Sentence embedding - sentence-transformers/multi-qa-mpnet-base-dot-v1; and 
-	Sentence embedding - sentence-transformers/all-MiniLM-L6-v2.

The indexes can be found in the 'faiss_index' folder. Further data cleaning has to occur to create better results for the FAISS search. These issues exist due to the underlying data quality issues with the HUDOC database itself. 

We have chosen FAISS as it has outperformed many other search systems including ElasticSearch, refer to the following academic work – ‘Efficient comparison of sentence embeddings’. 

### Search Improvements

The HSR will be improved in several respects. The following outlines the improvements and the process to build the improvements. 

1. HSR Summaries 

Currently, the ECHR has provided summaries for a minority of all cases they published. The HSR will utilise the full case and summary as training pairs to finetune:

- a T5 seq to seq model; or
- BART.

One of the above models will be chosen based on BLEU and ROUGE metrics, along side a human review. 

After the model is finetunes, all cases will be ran through the model to create summaries. 

2. Advanced Filtering

By combining traditional database filtering with semantic search, HSR will offer a powerful and efficient search experience. The backend logic would be a combination of database operations and machine learning-based search. The frontend would provide an intuitive interface for users to select and apply filters as per their requirements. Overall it would include the following steps

- Create further sentence embeddings for each section of a case (Facts, Conclusion, Summaries); and
- Requisuite FAISS indices.

3. Alternative Display

Many current databases offer displays that fall short of user needs. For example, HUDOC's traditional setup makes it challenging for users to gauge a case's relevance to their specific concerns at a glance. The HSR addresses this gap, spotlighting the most crucial facets of a case. This approach empowers users to quickly scan cases, determining if they warrant a deeper read. Often, this cursory view may suffice for the research needed.

While the Alternative Display isn't rooted in complex technology, its effectiveness lies in continuous refinement. We're committed to iterative feedback, actively engaging with both legal professionals and marginalized communities. This ensures we present the most pertinent information to our users.

For a preview of how such information and search is displayed, please refer to the search function in My-Rights.info - https://my-rights.info/search/


## Issue Identifier

The Issue Identifier takes a user prompt, which we assume will be in simple English, and converts it into a legalistic style. The purpose of this conversion/translation is to:
1.	Assist the Predictor to classify human rights violations more accurately as the prompt will be in a style similar to the training data; and 
2.	Assist the Search to identify appropriate cases as the closest neighbours will be more accurate. 

The Issue Identifier works through the Chat GPT4 API, where the system prompt is:
"Use the following steps to respond to user inputs. The user will input their facts that relate to a human rights violation in simple language. The output should be just a conversion of their input to legal writing, specifically, a legal formulation of 'facts'. In providing the output make no mention to: the date, any conclusion as to a human rights violation, and the reference to the person should always be "I”.”

### Issue Identifier Improvements

We will move from the Chat GPT4 API to seq2seq T5 model that reformulate a user's input query, which will be in simple language, to text that is styled as a legal fact. This model will be fine-tuned on the new dataset.

To create training pairs for the model, we plan to utilise Chat GPT4 API, with various temperature settings, to create the necessary pairs. Specifically, we will input actual legal facts and ask the model to generate simple language versions of the facts. The training pairs will assist the model in converting simple language commands to legal styled facts. 

The following provides an example of the way we would conduct the finetuning of the Issue Identifior: 

For example, these are the facts of a randomly selected ECTHR case: 

"The applicants complained that the authorities had failed to take adequate steps to protect the life of their family member, C.C., who had died of a drug overdose while being held in police custody."

Using a system prompt of: "Please rephrase the following legal statement in very simple language for someone who isn't familiar with legal terms. Create 5 different formulations of the simple language." 

The above provided:
1. The family says the police didn't keep C.C. safe, and he died from taking too many drugs.
2. C.C. took too many drugs and died while with the police. His family is upset about it.
3. C.C. was with the police when he died from drugs. His family thinks the police could have helped.
4. The police had C.C., and he died because of drugs. His family is sad and blames the police.
5. While C.C. was in jail, he died from drugs. His family says it's the police's fault.

The output provided is clearly written in simple language, i.e. by non-experts, and will assist in reformulating user queries. 

## Predictor
A multilabel classifier, fine tuned on the Hugging Face: lex_glue, ecthr_a dataset, takes a user prompt and outputs a multi-label classifier for 10 labels. The dataset contains 11k rows that correspond to the 'facts' section of 11k cases from the European Court of Human Rights. The 10 labels relate to the following human rights articles pursuant to the European Convention on Human Rights:
•	Article 2: Right to life
•	Article 3: Prohibition of torture
•	Article 5: Right to liberty and security
•	Article 6: Right to a fair trial
•	Article 8: Right to respect for private and family life
•	Article 9: Freedom of thought, conscience, and religion
•	Article 10: Freedom of expression
•	Article 11: Freedom of assembly and association
•	Article 14: Prohibition of discrimination
•	Article 1 of Protocol 1: Protection of property

The Predictor takes the output of the Issue Identifier to make the inference. 

The Predictor comes with two models, one where truncation is set to 512 tokens (RoBerta) and the second implements a sliding window technique to allow for a varying sequence length that accepts all sequences lengths (Bert). The f1 scores are provided in the table below.

| Epoch | Training Loss | Validation Loss | Accuracy | Precision/macro | Recall/macro | F1/macro | Precision/micro | Recall/micro | F1/micro | Roc Auc  |
|-------|---------------|-----------------|----------|-----------------|--------------|----------|-----------------|--------------|----------|----------|
| 1     | 0.162600      | 0.180850        | 0.493000 | 0.562606        | 0.470115     | 0.500645 | 0.694989        | 0.601887     | 0.645096 | 0.934406 |
| 2     | 0.135800      | 0.165844        | 0.535000 | 0.574869        | 0.476420     | 0.482516 | 0.696819        | 0.661321     | 0.678606 | 0.947212 |
| 3     | 0.114700      | 0.166618        | 0.532000 | 0.723237        | 0.575300     | 0.605141 | 0.698444        | 0.677358     | 0.687739 | 0.948240 |
| 4     | 0.093300      | 0.171745        | 0.542000 | 0.704298        | 0.591423     | 0.613898 | 0.702370        | 0.699057     | 0.700709 | 0.946493 |
| 5     | 0.082100      | 0.173036        | 0.536000 | 0.724979        | 0.598535     | 0.629181 | 0.702574        | 0.695283     | 0.698909 | 0.946950 |


To note, the RoBerta Model was chosen as all academic articles across the board, identified this model as producing the most accurate results for a multilabel classification exercise. We agree with this finding as it has beaten out BigBird, Bert, and HIER-Bert. The academic sources can be found below (‘Neural Legal Judgment Prediction in English’, ‘Classifying European Court of Human Rights Cases Using Transformer-Based Techniques’, 

512 Token limit RoBerta Notebook (utilised): https://colab.research.google.com/drive/1T78zSPTc_yZII09Y7V3ucCRCG765a39m#scrollTo=oLw6b3M7l6gV
Varying Length Bert Notebook (not utilised): https://colab.research.google.com/drive/1KqXLdZIAMSfkC6hj_eaEKZ6O71bYSY8m

### Predictor Improvements

The following improvements to the model are planned
1. Increase the dataset from the Huggingface Publicly Available library, which contains errors, to the My-Rights.info database that contains 3x the amount of cases and has corrected for underling data errors.
2. Conduct Named Entity Recognition and run multiple regex scripts to clean the dataset and improve bias scores, for examples, remove names of States that could lead to confounding issues for the Predictor and remove paragraph number labels, remove references to gender, remove references to specific religions.
3. Utilise multiple different models to identify which model produces the best results at various sequence lenghts.
4. Link the questionnaire from the My-Rights.info website to the Prediction Tool to create one pathway for a user to identify their legal situation without having the need to know the law.


## Dataset - Planned Improvements

Firstly, we will expand the dataset from the Huggingface Publicly Available, lex_glue, dataset to the My-Rights Dataset. This will have the following benefits:
- Expand the number of cases from 11k cases to roughly 30k cases;
- Expand the number of human rights articles from 11 to all the substantive human rights articles of 24;
- Account for numerous underlying labelling and text issues in the dataset that are due to HUDOC (the ECtHR's official database) itself;
- Conduct NER and RegEx to remove spurious statistical patterns; and
- Create balanced datasets for testing.  


## Accessibility - Planned Improvements

The entire website will be improved to have the following accessibility principles incorporated:
-	All text in the website will be written in simple and easily understood language;
-	The color scheme is designed to be easily recognised for people with visual impairment; 
-	The website will be screen reader friendly; and
-	There will be methods to convert the prompt and assist the user input their facts. 

## Academic Work Relied Upon

The ECTHR Prediction App, has been created from understanding and applying multiple leading academic sources, where the code base is not publicly available. The following freely accessible work has been relied upon:
 - Classifying European Court of Human Rights Cases Using Transformer-Based Techniques - https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=10130544
 - Neural Legal Judgment Prediction in English - https://aclanthology.org/P19-1424/
 - Zero-shot Transfer of Article-aware Legal Outcome Classification for European Court of Human Rights Cases  - https://aclanthology.org/2023.findings-eacl.44.pdf
 - Deconfounding Legal Judgment Prediction for European Court of Human Rights Cases Towards Better Alignment with Experts - https://aclanthology.org/2022.emnlp-main.74.pdf
 - Classifying European Court of Human Rights cases using transformer based Models - https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3033966/no.ntnu%3ainspera%3a112296943%3a24749335.pdf?sequence=1&isAllowed=y
 - LexGLUE: A Benchmark Dataset for Legal Language Understanding in English - https://arxiv.org/pdf/2110.00976v4.pdf
 - Efficient comparison of sentence embeddings - https://arxiv.org/pdf/2204.00820.pdf

Overall, the ECTHR Predictor App utilises state of the art mechanisms and the most up to date academic work to democratise legal prediction and search to ensure that all people irrespective of their characteristics can still access justice. 

## Usage

```
#Navigatge to project directories
cd /path/to/your_projects/

#Clone GitHub repo
git clone https://github.com/Viraaj-A/ecthr_prediction.git

#Navigate to the Github repo folder and install Pipenv
pip install pipenv

#Initialize new environment and activate environment
pipenv --python 3.11
pipenv shell

#Install dependencies 
pipenv install 

#Run app 
python app.py 
```
