Prediction of year of publication - Group Assignment for my Machine Learning Course
In this challenge the task is to predict the year of publication of scientific papers, based on their metadata.

The goal of the assignment is to predict the year in which the scientific paper is published.
To do this the train.json file is analyzed based on its features, and their relationship with the target value “year”.
The data is pre-processed to account for null values in the columns.
Thus, new features are then engineered in the newly cleaned data to help the models implemented to perform adequately.
The training data is split into a training and validation set, with a validation set size of 20%. All vectorization is done using CountVectorizer(),
as this is the most appropriate and in this also delivered a lower MAE score over for instance TfidfVectorizer().
The CountVectorizer converts a collection of text documents to a matrix of token counts.
CountVectorizer was deemed to be a better fit for our current task as it correlates with the frequency of words in a text document.
Both Logistic Regression and Random Forest were used as our desired models to see how good they perform.
Logistic Regression is a linear model algorithm for classification problems whereas random forest is an ensemble method,
a collection of decision trees with multiple tree boundaries. The result was based on the random forest model as we believe it is a better fit for our assignment and the data available. 

![image](https://github.com/stefos44/ML-Assignment/assets/151064157/21f3e570-b467-440b-9f6b-ee8197a910d4)
