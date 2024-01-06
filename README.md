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




Data pre-processing and feature engineering:

To start analyzing the correct path of data pre-processing and feature engineering, the baseline code was spectated and was used as a fundamental base. Firstly, the author column which was computed as a list of strings was transformed through the application of a lambda function to a single string separated by commas for easier data manipulation. Secondly after close inspection of the columns, the Editor column seemed to be composed of a lot of missing values so we decided to drop this feature because it would not contribute to the overall evaluation, and it could produce potential problems. The other features are engineered, for each dataset, as follows:
-	Entry type:
One-hot encoding is used, containing dummy variables representing each unique value.
-	Title:
For the Title column the CountVectorizer() method was used in order to convert text data into numerical values and as a result a better group of words representation can arise
-	Publisher: 
For the publisher column there was added a binary feature ‘has_publisher’that would implement if the research paper entry has a publisher or it does not. Respectively if a publisher existed it would be labeled as ‘1’ and if the column was empty (did not have a publisher) it would be labeled as ‘0’. Also for the original publisher column the same technique of applying the CountVectorizer() method was implemented
-	Author:
As mentioned before, after preprocessing the Author column by transforming into a string, the CountVectorizer() method was also implemented here.
-	Abstract:
For the abstract column a new feature ‘has_abstract’ was implemented following the same logic as was used for the publisher column. Additionally when implementing the CountVectorizer() method we set a max_features of 150 which in return limits the maximum number of features to 150 and possibly selecting the most important, relevant or frequent words in the abstract.
-	One-hot encoding for ENTRYTYPE:
This sequence of operations converts the categorical ‘ENTRYTYPE’ feature into multiple binary features of 0 and 1 and represents the presence or absence of each category 0 and 1. In this part we define the columns with ColumnTransformer() in order to transform the text in the title into numerical values for better processing with CountVectorizer(). In the function, we choose a transformer ‘passthrough’ in order the columns are the same through transformation. 

Hyperparameter tuning :

During the selection of parameters different optimizations with various configurations were explored When it came to the RFR model a GridSearch was conducted. The GridSearch creates a grid of all possible combinations of the hyperparameters and their values and evaluates each combination using cross-validation to determine which set of hyperparameters yields the best performance in the current model. We used  parameters such as: 
‘n_estimators’[100,150,200,250,300,350] which sets how many trees there should be in the forest. 
‘max_features’[1,2,4,6,8,12] which specifies the quantity of features to be selected
‘ max_depth’[5,10,15,20,Nine] which sets the depth of each individual tree.
 Lastly despite these variations the model ended up performing worse or degraded with no significant improvement. Thus, due to high computational expenses associated with these settings, we made the decision to just follow the default model. configuration with only a slight adjustment to the number of trees/estimators used from 100 to 200

Discussion of the performance of  our solution:

The logistic Regression MAE was 2.883609442320529 whereas the random forest regression was 3.0762524675002574. Both models performed substantially better than the base model performance of 5.8083 and this proves that the features and alterations to the baseline code that was given were crucial and correct. Overall we believe that the base features of the random forest algorithm helps to capture more complex relationships in the data compared to a linear classification algorithm like Logistic Regression. Even though the MAE of random forest was a bit higher than the one scored with Logistic Regression, we believe that because of the robustness that random forest provides against overfitting it is a more solid choice and since our data had irrelevant features like entrytype or noisy like proceedings, this is proven.

![image](https://github.com/stefos44/ML-Assignment/assets/151064157/2c365a6f-18d0-4ee4-b37d-a473a206d51f)


![image](https://github.com/stefos44/ML-Assignment/assets/151064157/21f3e570-b467-440b-9f6b-ee8197a910d4)
