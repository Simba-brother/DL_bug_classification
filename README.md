# DL_bug_classification
This repository stores our experimental code, dataset and results.

## Dataset

Our dataset is acquired from two aspects.

First, we revisited the dataset from the previous studies and got a total of 208 DL bugs. Second, we further filtered and manually reviewed a portion of the bugs on Stack Overflow. Specifically, we download the post data from the [Stack Exchange Data Dump](https://archive.org/details/stackexchange), which contains more than 58 million posts from Stack Overflow spanning from September 2008 to April 2024. Then, we perform an initial screening based on the following criteria:
- The PostTypeId is 1, which indicates that the post is a question post, not an answer post.
- The Score, AcceptedAnswerId of the post are greater than 0, which guarantees a high quality of posts.
- The post title does not include "how", "install", or "build", which can avoid general how-to questions and requests for installation instructions.
- The post contains code. Because posts about bugs usually contain code snippets.
- Tags of the post contain "keras", "tensorflow" or "pytorch", which ensures that the post is about deep learning.

After applying the criteria mentioned above, a total of 19149 posts remained. Given the time-intensive nature of the manual review process, we randomly select a subset of these posts for detailed inspection, from which we identify 292 DL bugs. Additionally, we randomly select 100 non–DL-bug posts from the 58 million posts in the Stack Exchange Data Dump to enrich our dataset. In the end, our dataset includes 500 (208+292) DL bugs and 100 non–DL bugs, each corresponding to a Stack Overflow post.

The ``dataset.csv`` file contains basic information about the dataset, including Stack Overflow post ID, the processed text (title+body), tags, and the label.

## Source Code
We've tested our code on Ubuntu 22.04 with Python 3.11.9.

### Requirments
- torch == 2.3.0
- pandas == 2.2.2
- transformers == 4.41.2
- scikit-learn == 1.5.0

### Run
Our model's training and testing code is contained in ``train.py``. Since our model is based on SOBERT, you first need to download SOBERT from the provided [URL](https://figshare.com/s/7f80db836305607b89f3), and then run the training code.
```
python train.py
```

## Results
The ``results`` folder contains the full results of our experiment.
- The folders ``sobert``, ``codebert``, ``roberta``, ``chatgpt``, ``claude``, ``tfidf``, and ``word2vec`` ccontain the experimental results of different methods. Each folder includes both the test set outputs and a summary of results across various evaluation metrics.
- The ``sobert_nocode`` folder contains the results of training and testing using a dataset that excludes code snippets.
- The ``delete_word.csv`` and ``delete_word_mul.csv`` files contain the our model's performance results after removing single words and words combinations, respectively.