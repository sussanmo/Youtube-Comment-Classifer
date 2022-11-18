# Youtube-Comment-Classifer
uses machine learning classifiers: SVM, Naive Bayes, Random Forest, and Decision Trees to classify YouTube comments.


YouTube Comments Data set:

The YouTube comments dataset contains 6000 comments from 12 coding tutorial videos selected from multiple coding channels using the YouTube Data API. The data was collected on September 6, 2016.

The dataset contains a sample of 500 comments from each video. These comments were retrieved from YouTube.

Each comment in the dataset has been labeled by a team of five annotators to identify comments that carry some sort of useful feedback to content creators and viewers. Each comment in the dataset is placed in one of two categories, content concerns or miscellaneous.

These categories can be described as follows:

Content Concerns: 
These comments include questions or concerns about certain parts of the video content that needs further explanation or comments that point out errors within the video. Content concerns can also include requests for certain future content and suggestions to improve the quality of the tutorial by giving advice to the creator.

Miscellaneous: 
This category includes all other comments that do not provide any technical information about the content of the video. For instance, comments including praise, insults, or spam. Such comments are not beneficial to the content creators or the viewers.

Source: Poché et al. "Analyzing user comments on YouTube coding tutorial videos." In Proceedings of the 25th IEEE/ACM International Conference on Program Comprehension (ICPC-17), pp. 196-206, IEEE, 2017.


Research Question: 
Can informative content-relevant comments be automatically identified and classified?\

Program operations: 
Reads and parses through catalog of comment files

Utilizes machine learner classifiers to automatically classifying YouTube Comments on programming videos into informative (technical concerns) and uninformative (miscellaneous) comments. 

Use Accuracy, Precision, Recall and F1 score as evaluation metrics for classifier by evaluating the classifiers using 10 fold cross validation.

