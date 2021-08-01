# NLP-Question-Answering
Springboard Capstone



**1. What is the problem you want to solve? Why is it an interesting problem?**

The task is to build a model that automatically answer questions posed by humans in a natural language. It is easy for humans to answer questions from text but very difficult for machines because the task requires machines which can only take numerical values as input to obtain language intelligence. In essense, the task is to select the best short and long answers from Wikipedia articles to the given questions.

**2. What data are you going to use to solve this problem? How will you acquire this data?**

The project is based on the Tensorflow 2.0 Question Answering from Kaggle which has the data in JSON format. Each sample contains a Wikipedia article, a related question, and the candidate long form answers. The training examples also provide the correct long and short form answer or answers for the sample, if any exist.
### Data fields
*	document_text - the text of the article in question (with some HTML tags to provide document structure). The text can be tokenized by splitting on whitespace.
* question_text - the question to be answered
* long_answer_candidates - a JSON array containing all of the plausible long answers.
* annotations - a JSON array containing all the correct long + short answers. Only provided for train.
* document_url - the URL for the full article. Provided for informational purposes only. This is NOT the simplified version of the article so indices from this cannot be used directly. The content may also no longer match the html used to generate document_text. Only provided for train.
* example_id - unique ID for the sample.


**3. In brief, outline your approach to solving this problem (knowing that you may not know everything in advance and this might change later). This might include information like:**

The problem is a supervised classification problem. 

* What are you trying to predict? \
For each article + question pair, I must predict / select long and short form answers to the question drawn directly from the article. - A long answer would be a longer section of text that answers the question - several sentences or a paragraph. - A short answer might be a sentence or phrase, or even in some cases a YES/NO. The short answers are always contained within / a subset of one of the plausible long answers. - A given article can (and very often will) allow for both long and short answers, depending on the question.

* What will you use as predictors? \
The word embedding technique is used to convert words to vectors as input. Features are NLP metrics calculated using word vectors

* Will you try a more “traditional” machine learning approach, a deep learning
approach, or both? \

Both Deep learning and traditional machine learning approaches are explored here.

**4. EVALUATION
* **Traditional ML model (LightGBM)** with 11 NLP metrics where see below, model f1 score being 0.15 higher than random prediction 0.03.
![lightGBM](https://user-images.githubusercontent.com/57920705/127776696-cb59db2d-c2e8-437e-8617-01cc1b04c287.JPG)
* a predition sample is shown
![test2](https://user-images.githubusercontent.com/57920705/127776595-f79a021d-a071-47ed-823f-39257d60b5ad.JPG)

* **LSTM recurrent neural network with Transfer Learning from FASTTEXT Model** model f1 score being 0.43 higher than random prediction 0.09.
* ![LSTM_Fasttext](https://user-images.githubusercontent.com/57920705/127777738-1881f91a-8695-487d-8c56-4c364d92c668.JPG)



**5. What computational resources would you need at a minimum to do this project?**

CPU for traditional ML approach and GPU for deep learning approach
