# NLP-Question-Answering
Springboard Capstone



**1. What is the problem you want to solve? Why is it an interesting problem?**

The task is to build a system that automatically answer questions posed by humans in a natural language. It is easy for humans to answer questions from text but very difficult for machines because the task requires machines which can only take numerical values as input to obtain language intelligence.

**2. What data are you going to use to solve this problem? How will you acquire this data?**

The project is based on the Tensorflow 2.0 Question Answering from Kaggle which has the data in JSON format.

**3. In brief, outline your approach to solving this problem (knowing that you may not know everything in advance and this might change later). This might include information like:**
a. Is this a supervised or unsupervised problem?

The problem is a supervised problem.
b. If supervised, is it a classification or regression problem?

It is a classification problem
c. What are you trying to predict?

I am trying to predict the span index from the text as the answer associated to each question.
d. What will you use as predictors?

The word embedding technique is used to convert words to vectors as input. Features are NLP metrics calculated using word vectors
e. Will you try a more “traditional” machine learning approach, a deep learning
approach, or both?

Both Deep learning and traditional machine learning approaches are explored here.

**4. What will be your final deliverable? This is typically an application deployed as a web**
service with an API or (for extra credit) a web/mobile app.

deployed with an API.

**5. What computational resources would you need at a minimum to do this project?**

CPU for traditional ML approach and GPU for deep learning approach
