# Data Science Bootcamp Written Assessment — Answers


## SECTION A


### Q1: What is Jupyter Notebook and how is it used in data science?

Jupyter Notebook is an open-source, web-based interactive environment for writing and running code (mostly Python) in cells, with each cell's output (tables, charts, printed results) shown directly underneath. Code, Markdown text, equations and visualisations all live in one .ipynb document.

In data science it is used to load and explore data step by step, clean and transform it, build quick visualisations, prototype and evaluate machine-learning models, and document the reasoning alongside the code so the analysis can be shared and reproduced.

### Q2 Name the steps in a data science life cycle

1. Problem definition / business understanding – define the question and success criteria.

2. Data collection – gather data from files, databases, APIs, web scraping, etc.

3. Data cleaning / preparation – handle missing values, duplicates, outliers, wrong data types; encode categories; scale features.

4. Exploratory data analysis (EDA) – summary statistics and visualisations to understand distributions and relationships.

5. Feature engineering / selection – create and choose the most useful input variables.

6. Modelling – select algorithms, split into train/test sets and train the models.

7. Model evaluation – measure performance with suitable metrics (e.g. R², RMSE, accuracy, confusion matrix) and tune hyperparameters.

8. Deployment – put the model into production or communicate the results to stakeholders.

9. Monitoring and maintenance – track performance over time and retrain when data changes.

## SECTION B


### Q1: How do you reverse a string in Python?

Use slicing with a step of -1:

```python
s = "hello"
reversed_s = s[::-1]   # 'olleh'
```

Alternatively: ''.join(reversed(s)).

### Q2: How do you access the last element of a list in Python?

Use negative indexing – index -1 refers to the last element:

```python
my_list = [10, 20, 30]
last = my_list[-1]   # 30
```

(my_list[len(my_list) - 1] also works; my_list.pop() returns and removes it.)

## SECTION C


### Q1: How do you import a module in Python?

Use the import statement:

```python
import math                       # import the whole module
import numpy as np                # import with an alias
from math import sqrt             # import a specific name
from sklearn.linear_model import LinearRegression
```

### Q2: What is a NumPy array?

A NumPy array (ndarray) is the core data structure of the NumPy library: a fixed-size, n-dimensional grid of values that all share the same data type, stored in a contiguous block of memory.

Compared with Python lists it is much faster and more memory-efficient and supports vectorised operations (element-wise arithmetic without loops), broadcasting, slicing/indexing and many mathematical, statistical and linear-algebra functions.

```python
import numpy as np
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a.shape)   # (2, 3)
print(a * 2)     # [[2 4 6] [8 10 12]]
```

## SECTION D


### Q1: What is the purpose of exploratory data analysis?

Exploratory data analysis (EDA) is used to understand a dataset before modelling. Its purpose is to summarise the main characteristics of the data (using statistics and visualisations), check data quality (missing values, duplicates, errors, outliers), understand the distribution of each variable, discover relationships and correlations between variables, test assumptions, and generate hypotheses. These insights guide data cleaning, feature selection and the choice of model.

### Q2: What is matplotlib, and how is it used in data science?

Matplotlib is Python's foundational 2-D plotting library (mainly used through its pyplot module). In data science it is used to visualise data during EDA and to communicate results – e.g. line plots, bar charts, histograms, scatter plots, box plots and heatmaps – and to inspect model results such as predictions vs. actual values. Libraries such as Seaborn and pandas plotting are built on top of it.

```python
import matplotlib.pyplot as plt
plt.scatter(df['horsepower'], df['price'])
plt.xlabel('Horsepower'); plt.ylabel('Price')
plt.show()
```

## SECTION E


### Q1: What is a correlation?

Correlation is a statistical measure of the strength and direction of the relationship between two variables. The Pearson correlation coefficient (r) ranges from -1 to +1: +1 is a perfect positive relationship (both increase together), -1 a perfect negative relationship (one increases while the other decreases) and 0 means no linear relationship. Correlation does not imply causation. In pandas it is computed with df.corr().

### Q2: What is linear regression?

Linear regression is a supervised machine-learning algorithm used to predict a continuous numeric target from one or more input features by fitting a straight line (or hyperplane) that best describes their relationship:

y = b0 + b1·x1 + b2·x2 + … + bn·xn

where b0 is the intercept and b1…bn are coefficients. The coefficients are usually found by Ordinary Least Squares, i.e. minimising the sum of squared differences between actual and predicted values. Simple linear regression uses one feature; multiple linear regression uses several. It is evaluated with metrics such as R², MSE and RMSE – e.g. predicting car price from horsepower, engine size and other features.

## SECTION F


### Q1: What is classification, and how is it performed in Python?

Classification is a supervised learning task in which a model learns from labelled data to predict which discrete category (class) a new observation belongs to – e.g. spam vs. not spam, or malignant vs. benign.

In Python it is usually performed with scikit-learn: load and prepare the data, split it with train_test_split, (optionally) scale features, choose a classifier (LogisticRegression, DecisionTreeClassifier, KNeighborsClassifier, RandomForestClassifier, SVC…), train it with .fit(X_train, y_train), predict with .predict(X_test), and evaluate with accuracy_score, confusion_matrix and classification_report.

```python
from sklearn.linear_model import LogisticRegression
model = LogisticRegression().fit(X_train, y_train)
y_pred = model.predict(X_test)
```

### Q2: What is a classification problem?

A classification problem is a problem where the output (target) variable is categorical rather than continuous, so the goal is to assign each input to one of a set of predefined classes. It can be binary (two classes, e.g. yes/no, fraud/not fraud), multi-class (more than two, e.g. classifying types of flowers) or multi-label (several labels at once). This contrasts with a regression problem, where the target is a continuous number such as price.

## SECTION G


### Q1: What is the purpose of hyperparameter tuning?

Hyperparameters are settings chosen before training rather than learned from the data – e.g. max_depth of a decision tree, n_neighbors in KNN, or C and penalty in logistic regression. Hyperparameter tuning searches for the combination that gives the best performance, so the model generalises well to unseen data and avoids underfitting or overfitting. It is usually done with cross-validation using GridSearchCV (tries every combination) or RandomizedSearchCV (tries random combinations).

### Q2: What is a confusion matrix?

A confusion matrix is a table used to evaluate a classification model by comparing actual classes with predicted classes. For binary classification it has four cells:

• True Positives (TP) – positive cases correctly predicted positive  
• True Negatives (TN) – negative cases correctly predicted negative  
• False Positives (FP) – negative cases wrongly predicted positive (Type I error)  
• False Negatives (FN) – positive cases wrongly predicted negative (Type II error)

From it we calculate metrics such as accuracy = (TP+TN)/total, precision = TP/(TP+FP), recall = TP/(TP+FN) and F1-score. In scikit-learn: confusion_matrix(y_test, y_pred).
