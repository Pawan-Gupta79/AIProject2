# AIProject2
Repository for Artificial Intelligence project 2.


COMP 472 Artificial Intelligence Summer 2026

Mini project 2

Due date: Sunday, 21 June 2026, 11:59 PM


Objective:
- To gain practical experience with Machine Learning using Python.
- To understand how computers can learn patterns from data.
- To appreciate how powerful AI systems can be built with relatively little code.
- To learn new AI concepts and Python libraries.
- To strengthen Python programming skills..
- To learn the following Python skills
1) File handling
2) Functions and modularity
3) OOP basics
4) Error handling
5) Working with datasets
6) Basic data visualization

   
- To learn the following AI skills using built-in Python libraries and models
1) Supervised Learning
2) Training vs Testing Data
3) Feature Extraction
4) Text Classification
5) Model Evaluation
6) Confusion Matrices

7) 
- You will need to use the following libraries
1) pandas
2) numpy
3) scikit-learn
4) matplotlib
5) seaborn

   
Instructions:
In this project, you will Develop an AI system that automatically classifies emails as: 1) Spam or 2) Not
Spam (Ham).
Concordia wants an intelligent email filtering system.

The program should:
1. Load a dataset of emails
2. Convert email text into numerical features (new thing to learn on your own)
3. Train a machine learning model
4. Predict whether new emails are spam or not
5. Display confidence levels
6. Evaluate model performance
Dataset
You need to use the SMS Spam Collection dataset.

Example:
message label
Win a FREE iPhone today! spam
Meeting at 2pm tomorrow ham
Congratulations! Claim your prize now! spam
The csv file is attached to the assignment on Moodle.


Deliverables
Each group must submit:
1. Python source code
2. Screenshot of program execution
3. Short reflection (1 page)
The short reflection file must contain:
1. Names and ID’s of team members
2. Precise contribution of each team member.
3. Brief list of difficulties faced and how you overcome them.
4. Actual time it took you to understand, develop, and test each component of the project.

   
A demo will be given to one of our marks. No credit will be given if you miss your demo. All team
members must attend the demo. Each team member should be able to answer any questions about
the project. It is encouraged to divide the work between the team members but each one MUST be
able to answer any question about any part of the project. Remember, one of the objectives of
working in teams is to learn from one another.
Demo time slots will be posted on Moodle right after the due date.


Project description:
Concordia University wants you to develop an intelligent email filtering system.
The system should:
1. Detect spam messages.
2. Detect legitimate messages.
3. Learn from examples.
4. Provide confidence scores for predictions.
5. Evaluate its own performance.
Example messages:
"Congratulations! You have won a free iPhone."
"Please submit your assignment before Friday."
"Claim your prize now!"
"Meeting moved to 2 PM tomorrow."
Required features:
1. The program must load a CSV dataset using pandas.
The dataset should contain two columns:
label,message
Example:
spam,"Win a free vacation now!"
ham,"Your appointment is scheduled for tomorrow."
The first row should be:
label,message
2. Feature extraction
The assistant must convert text into numerical features.
You must use:
TfidfVectorizer
from scikit-learn.
This is one of the AI concepts that you need to learn on your own.
Hint:
Computers cannot directly understand words. AI systems must transform text into numerical
representations before learning patterns.
3. Model training
The program must train a machine learning classifier.
You may use:
• Logistic Regression
or
• Naive Bayes
from scikit-learn.
The classifier must learn from the training dataset and predict whether future messages are spam
or not spam.
4. Model evaluation
The program must display:
• Accuracy
• Confusion Matrix
Example:
Accuracy: 96.4%
Confusion Matrix:
Predicted
Spam Ham
Actual Spam 52 3
Actual Ham 2 143
The objective is to evaluate how well the model performs.
5. Confidence Scores
When predicting a message, the program must display:
• Predicted label
• Confidence score
Example:
Prediction: SPAM
Confidence: 98.5%
6. Data Visualization
The program must generate a chart showing:
• Number of spam messages
• Number of non-spam messages
You may use matplotlib.
Example:
Spam: 700
Ham: 4800
A simple bar chart is sufficient.
6. Interactive prediction loop
The program must:
• Continuously accept user input.
• Predict whether the message is spam.
• Display confidence scores.
• Exit when the user types: quit
Estimated timeline:
Task Estimated time
Understanding TF-IDF 2 hour
Environment setup 1 hour
Dataset loading 1 hour
Model training 2 hours
Model evaluation 2 hours
Visualization 1 hour
Testing and debugging 2 hours
Having fun Unlimited as always
Example output:
Welcome to Spam Detection AI
Training model...
Accuracy: 96.8%
Type 'quit' to exit.
Enter message:
Congratulations! You won $5000.
Prediction: SPAM
Confidence: 99.1%
Enter message:
Please submit your project by tomorrow.
Prediction: HAM
Confidence: 97.8%
Enter message:
quit
Goodbye!
Some implementation hints and guidelines:
1. Use pandas to load the dataset.
2. Use TfidfVectorizer from scikit-learn to convert text into features.
3. Split the dataset into training and testing data using train_test_split.
4. Train a classifier using LogisticRegression or MultinomialNB.
5. Evaluate the classifier using:
accuracy_score
confusion_matrix
6. Use matplotlib to create a bar chart showing the number of spam and ham messages.
7. Build an interactive loop that accepts user messages and predicts their class.
Grading rubric
No demo → automatic 0
Not being able to answer questions about or to explain the submitted code → automatic 0
No reflection file automatic 0
Component Marks
Dataset loading 10
Feature extraction (TF-IDF) 20
Model training 20
Model evaluation 20
Confidence score 10
Interactive prediction loop 10
Code quality and documentation 10
Enjoy, lean and have fun
