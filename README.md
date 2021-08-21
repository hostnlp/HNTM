#### Datasets
The raw datasets used in the paper can be downloaded via:

20NEWS:   
http://qwone.com/~jason/20Newsgroups/

Reuters:   
https://www.nltk.org/book/ch02.html

Wikitext-103:   
https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/

We use the same preprocessing steps as described in Miao et al. (2016), Wu et al. (2020), Nan et al. (2019) to obtain the vocabulary of 20NEWS, Reuters and Wikitext-103 respectively.


#### Model
The model can be trained on Reuters by running:

    python HNTM.py

The best hyperparameter values on the validation set are as follows:  
  

decay\_rate = 0.03  
discrete\_rate = 0.1  
balance\_rate = 0.01  
manifold\_rate = 0.3  
learning\_rate = 5e-4  
batch\_size = 64  
n\_epoch = 100  
hidden\_size = 256

#### Requirements
tensorflow==1.12.0  
numpy==1.14.5
