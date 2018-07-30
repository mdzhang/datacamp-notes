# Machine Learning

- **machine learning**: ability of computers to learn and make decisions from data w/o being explicitly programmed
    - learning relationship between input and output pairs, x and y

- **unsupervised learning**: uses unlabeled data; finds patterns w/o a specific prediction task in mind
    - **clustering**: group data into categories, where categories not known beforehand
    - **reinforcement learning**: software agent acts within environment, actions are rewarded or punished, agent uses these responses to optimize behavior

- **supervised learning**: uses labeled data to find patterns for a prediction task
    - has **predictor variables/features/independent variables** and a **target/response/dependent variable**
    - **labeled data** has known output i.e. target variable value is known
    - goal is to build a model that can predict target variable given predictor variables
        - to automate time consuming or expensive manual task
        - to make predictions about the future
    - if target variable is continuous, it's a **regression task**
    - if target variable is discrete, it's a **classification task**
        - binary decisions have discrete set of values i.e. true or false


- predictor variables are columns in 2D array
- rows are samples/individual observations
- dimension of dataset == # features

- **human in the loop machine learning system**
  - machine learning algorithm determines a label with some probability
  - human prioritizes time spent based on those probabilities

- start with basic model
  - less can go wrong than with complex models
  - see how much "signal" can be gleaned with basic methods
  - less computationally expensive
  - think carefully about features
