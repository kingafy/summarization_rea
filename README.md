# summarization_rea
https://ai.googleblog.com/2020/02/exploring-transfer-learning-with-t5.html
https://github.com/icoxfog417/awesome-text-summarization
https://github.com/vineetk1/Automatic-Text-Summarization-Literature-Review
https://github.com/google-research/pegasus
Summarization baiscs:-
https://ryanong.co.uk/2020/01/15/day-15-textrank-for-summarisation-code-gensim/
https://ryanong.co.uk/2020/01/16/day-16-textrank-manual-implementation-code/
https://ryanong.co.uk/2020/01/22/day-22-tfidf-for-summarisation-putting-it-all-together/   ==best one selfmade ranking 
https://ryanong.co.uk/2020/02/04/day-35-learn-nlp-with-me-mrc-new-trends-ii/
https://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html
https://github.com/DISCOSUMO/query-based_summarization/blob/master/query-based_baseline.py
https://github.com/Wendy-Xiao/Extsumm_local_global_context  ==imp
https://ryanong.co.uk/2020/05/01/day-122-nlp-papers-summary-applying-bert-to-document-retrieval-with-birch/
Biomedical dataset https://github.com/sahilmishra0012/Text-Summarization-For-Biomedical-Domain-Content/blob/master/77.pdf

https://medium.com/analytics-vidhya/text-summarization-using-bert-gpt2-xlnet-5ee80608e961

https://ryanong.co.uk/2020/03/24/day-84-mini-nlp-data-science-project-implementation-v-text-clustering-iii/  ==TOPIC MODELLING
https://ryanong.co.uk/2020/03/25/day-85-mini-nlp-data-science-project-implementation-vi-topic-modelling-analysis/  =TOPIC MODELLING ANALYSIS

https://ryanong.co.uk/2020/03/26/day-86-mini-nlp-data-science-project-implementation-vii-text-similarity/ ==CLUSTER SENTENCS

From <https://ryanong.co.uk/2020/02/16/day-47-learning-pytorch-autograd-automatic-differentiation/> 


ROUGE-1 (unigram overlap), ROUGE-2 (bi-gram overlap), and ROUGE-L (longest common subsequence) are the most popular ROUGE scores 

From <https://ryanong.co.uk/2020/01/23/day-18-summarisation-evaluation-metrics/> 




Imp approach :-->
https://ryanong.co.uk/2020/04/22/day-113-nlp-papers-summary-on-extractive-and-abstractive-neural-document-summarization-with-transformer-language-models/
A Summarization System For Scientific Documents

From <https://ryanong.co.uk/2020/04/23/day-114-nlp-papers-summary-a-summarization-system-for-scientific-documents/> 

 https://www.aclweb.org/anthology/D19-3036.pdf

From <https://ryanong.co.uk/2020/04/23/day-114-nlp-papers-summary-a-summarization-system-for-scientific-documents/> 


Clinincal BERT sum(extractive)
https://web.stanford.edu/class/cs224n/reports/custom/report29.pdf

ClinicalBertSum: RCT Summarization by Using Clinical BERT Embeddings

Data-Driven Summarization Of Scientific Articles

From <https://ryanong.co.uk/2020/04/25/day-116-nlp-papers-summary-data-driven-summarization-of-scientific-articles/> 


https://ryanong.co.uk/2020/04/25/day-116-nlp-papers-summary-data-driven-summarization-of-scientific-articles/

Extractive Summarization Of Long Documents By Combining Global And Local Context

From <https://ryanong.co.uk/2020/04/27/day-118-nlp-papers-summary-extractive-summarization-of-long-documents-by-combining-global-and-local-context/> 

Source: https://www.aclweb.org/anthology/D19-1298.pdf

From <https://ryanong.co.uk/2020/04/27/day-118-nlp-papers-summary-extractive-summarization-of-long-documents-by-combining-global-and-local-context/> 



https://colab.research.google.com/github/google-research/text-to-text-transfer-transformer/blob/master/notebooks/t5-trivia.ipynb

http://nlpprogress.com/english/summarization.html
https://pypi.org/project/bert-extractive-summarizer/

Datet :-http://nlpprogress.com/english/summarization.html


Below are some good beginner document summarization datasets.
• Legal Case Reports Data Set. A collection of 4 thousand legal cases and their summarization.
• TIPSTER Text Summarization Evaluation Conference Corpus. A collection of nearly 200 documents and their summaries.
• The AQUAINT Corpus of English News Text. Not free, but widely used. A corpus of news articles.

From <https://machinelearningmastery.com/datasets-natural-language-processing/> 


https://blog.usejournal.com/nlp-for-topic-modeling-summarization-of-financial-documents-10-k-q-93070db96c1d


https://www.aclweb.org/anthology/W19-8902.pdf

https://towardsdatascience.com/summarization-has-gotten-commoditized-thanks-to-bert-9bb73f2d6922

https://arxiv.org/pdf/2004.15011.pdf
https://ryanong.co.uk/2020/05/03/day-124-nlp-papers-summary-tldr-extreme-summarization-of-scientific-documents/


https://medium.com/jatana/unsupervised-text-summarization-using-sentence-embeddings-adb15ce83db1

https://awesomeopensource.com/projects/summarization

Word Embedding base dSummarization

Code for Text Rank:-https://ryanong.co.uk/2020/01/16/day-16-textrank-manual-implementation-code/


https://ryanong.co.uk/2020/01/18/day-18-tfidf-for-summarisation-implementation-ii-term-frequency-tf-matrix/

Metrics:-ROGUE/BLEU and METEOR :---not so good with abstractive summarizaton

Objective and Contribution
The objective is to use existing lead bias in news data to pretrain summarisation models on unlabelled datasets. We want the model to predict lead sentences using the rest of the article. Lead bias is a common problem in news dataset, where few sentences at the beginning of the article contains the most important information and so models trained on news dataset has a bias towards selecting those sentences and ignore sentences later on in the article.
Datasets
We have collected 21.4M articles (June 2016 – June 2019) after filtering articles based on the overlapping non-stopping words ratio between the top 3 sentences and the rest of the article. A high overlapping non-stopping words ratio tells us that there is a strong semantic connection.
Evaluation is made on three benchmark news summarisation datasets:
	1. New York Times (NYT) corpus – 104K news articles
	2. Xsum – 227K news articles
	3. CNN/Daily Mail – 312K news articles
Methodology

Given a news article, we take the lead-3 as the target summary and use the rest of the article as the news content as shown in the figure above. This allows us to utilise unlabelled news datasets to train our summarisation models. This pretraining method can be apply to any datasets with structural bias, for example, academic papers with abstracts or books with tables of contents. However, the pretraining needs careful examine and cleaning to ensure we have a good target summary for our content.
Experiments
The abstractive summarisation model is the traditional transformer encoder-decoder architecture. We won’t go into details the architecture here. The pretraining with unlabelled Lead-3 (PL) with finetuning on target datasets is denoted PL-FT and without finetuning is denoted PL-NoFT.
WHAT’S THE DATA CLEANING PROCESS?
	1. Remove media agencies, dates and other irrelevant contents using regular expressions
	2. Only keep articles with 10 – 150 words in the lead-3 sentences and 150 – 1200 words in the rest of the article. In addition, remove any articles where lead-3 sentences are repeated in the rest of the article. This is to filter out articles that are too long or too short and to encourage abstractive summaries
	3. Remove articles that have “irrelevant” lead-3 sentences. The relevancy is computed using the ratio of overlapping words between lead-3 sentences and rest of the article. A high overlapping words ratio means that the lead-3 sentences is a good representative summary of the rest of the article. The threshold ratio is 0.65.
MODELS COMPARISON
	• Lead-X: uses the top X sentences as summary (X = 3 for NYT and CNN/DM and X = 1 for XSum)
	• PTGen: pointer-generator network
	• DRM: uses deep reinforcement learning for summarisation
	• TConvS2S: convolutional neural network
	• BottomUp: Two-step approach for summarisation
	• SEQ: uses reconstruction and topic loss
	• GPT-2: pretrained language model
Results
The evaluation metric is the traditional ROUGE scores (ROUGE-1, ROUGE-2, and ROUGE-L). The results for all three evaluation datasets are shown in the figures below:



	• PL-FT model outperformed all baseline models on both NYT and Xsum dataset. On CNN/Daily Mail, it outperformed all except BottomUp
	• PL-NoFT outperformed all the unsupervised models on CNN/Daily Mail with a significant margins. It also performed well in Xsum. PL-NoFT is the same model across all three datasets, showcasing its generalisation ability
ABSTRACTIVENESS
The summaries generated by both PL-noFT and PL-FT have more novel unigram than reference summaries. PL-noFT has similar novelty ratio as reference in other n-grams but PL-FT has a relatively low novelty ratio post finetuning.

HUMAN EVALUATION
Perform human evaluation on the summaries generated by the PL models and pointer-generator network. The scoring system and results are shown below. Results show that both PL-noFT and PL-FT outperformed pointer-generator network. This showcase the power of both the pretraining and finetuning strategy.

Conclusion and Future Work
The paper uses the lead bias existed in news data as the target summary and pretrain summarisation models. Our pretrained model without finetuning achieve SOTA results over different news summarisation datasets. Performance improved further with finetuning. Overall, this pretraining method can be apply to any datasets where there are structural bias.
Source: https://arxiv.org/pdf/1912.11602.pdf

From <https://ryanong.co.uk/2020/04/16/day-107-nlp-research-papers-make-lead-bias-in-your-favor-a-simple-and-effective-method-for-news-summarization/> 


![Uploading image.png…]()
