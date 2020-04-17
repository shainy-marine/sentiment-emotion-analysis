# Sentiment-Emotion-Analysis #
sentiment-emotion-analysis can help us predict the emotions of the text written by users (passed as input to the service)

# Installation #
<a href="https://www.python.org/downloads/">Python-3.7 </a><br/>
 <a href="https://www.anaconda.com/distribution">Anaconda distribution</a>

# Documentation #
This project aims to classify the emotion on a person's text in a tweet into one of six categories, using Logistic regression Algorithm. This consist dynamic dataset, with six emotions - angry, love, fear, joy, sadness and surprise.

# Development setup #
Install all the dependencies using virtualenv.

virtualenv -p python3 ./
source ./bin/activate
pip install -r ./requirements.txt

Install awscli dependency using virtualenv.

pip install awsebcli

## Important Code Value Mappings
#### Sentiment Analysis
> ``` Positive = 1, Neutral = 0, Negative = -1```
#### Emotion Analysis
> ``` Anger = 1, Fear = 2, Joy = 3, Love = 4, Sadness = 5, Surprise = 6 ```

# Conda environment #
Conda is an open source package management system and environment management system that runs on Windows, macOS and Linux. Conda quickly installs, runs and updates packages and their dependencies. Conda easily creates, saves, loads and switches between environments on your local computer. It was created for Python programs, but it can package and distribute software for any language.
Use the conda install command to install 720+ additional conda packages from the Anaconda repository.

# Algorithm #
Logistic Regression:
Logistic regression is a classification algorithm used to assign observations to a discrete set of classes. Unlike linear regression which outputs continuous number values, logistic regression transforms its output using the logistic sigmoid function to return a probability value which can then be mapped to two or more discrete classes.

# Running Tests #
To run the tests in this directory, make sure you've installed the Development setup.
Then you can run all tests (unit and cross-validation) using python IDE.

# Contributing #
 * Fork it!
 * Create your feature branch: git checkout -b my-new-feature
 * Commit your changes: git commit -am 'Add some feature'
 * Push to the branch: git push origin my-new-feature
 * Submit a pull request!
