
### Project Overview

The purpose of this project is to use a deep learning neural network to create a binary classifier that can predict, with 75% or greater accuracy, whether applicants will be successful if funded by Alphabet Soup. The dataset contains more than 34,000 organizations that have received funding from Alphabet Soup in the past. The dataset contains information about each organization such as the type of organization, the use of the funding, the amount of funding, and the success of the organization.  

### Project Structure
The project follows a typical machine learning workflow, including data preprocessing, model design, training, evaluation, and optimization. The following steps are taken to complete the project: 
<br> 
<br>

#### Initial Model

**1. Data Preprocessing**

- Depencies are set up.
- Pandas is used to create a DataFrame from a CSV file.   
- Columns with non-beneficial information, `EIN`  and `NAME`, are dropped.
- The categorical variables, `APPLICATION_TYPE` and `CLASSIFICATION`, are binned with thresholds of 250 and 1000, respectively.
- `pd.get_dummies` is used to encode the categorical variables. 
- The target variable `IS_SUCCESSFUL` is identified and separated as the `y` variable.    
- The remaining features are assigned to `X`. 
- The features are split into training and testing datasets. 
- The features are then standardized using `StandardScaler`.   
<br>


**2. Model Design**

- A Sequential model is defined using TensorFlow's Keras API.
- A simple 3-layer neural network is designed with the following characteristics:
    - The input layer has 80 neurons, uses the `relu` activation function, and expects 43 features.    
    - A hidden layer is added with 30 neurons, respectively, and `relu` activation functions are used for both layers.
    - The output layer has 1 neuron with the `sigmoid` activation function suitable for binary classification.
- `LambdaCallback` is used to monitor the training loss and accuracy for every 5 epochs.
- The model is compiled using `binary_crossentropy` as the loss function, `adam` as the optimizer, and `accuracy` as the metric.     
- The model is trained with the scaled training and target variables for 100 epochs.
- The model is evaluated with a verbose level of 2. 
<br>

**3. Results**  

<img src="images/initial_eval.png" alt="Initial Model Evaluation" width="900"/>

<br>
The model achieves an accuracy of around 72.5% and a loss of about 0.56. The model does not reach the target accuracy of 75%. The following steps are taken to optimize the model:  

<br>
<br>
<br>

#### Optimization Attempt 1  

- REDUCE THE NUMBER OF COLUMNS BY REBINNING THE `CLASSIFICATION` AND `APPLICATION_TYPE` COLUMNS WITH GREATER THRESHOLDS
- USE DIFFERENT ACTIVATION FUNCTIONS FOR THE INPUT AND HIDDEN LAYERS
- USE THREE ADDITIONAL HIDDEN LAYERS
- USE 128 NEURONS IN ALL NON-OUTPUT LAYERS  
<br>

**1. Data Preprocessing**

- A new `.ipynb` file is created to optimize the model.   
- Depencies are set up.
- Pandas is used to create a DataFrame from a CSV file.   
- Columns with non-beneficial information, `EIN`  and `NAME`, are dropped.
- The categorical variables, `APPLICATION_TYPE` and `CLASSIFICATION`, are binned with thresholds of 800 and 2000, respectively.

- `pd.get_dummies` is used to encode the categorical variables. 
- The target variable `IS_SUCCESSFUL` is identified and separated as the `y` variable.    
- The remaining features are assigned to `X`. 
- The features are split into training and testing datasets. 
- The features are then standardized using `StandardScaler`.   

<br>

**2. Model Design**

- A Sequential model is define using TensorFlow's Keras API.
- The planned optimization steps listed above are implemented in the model design.
- The model is compiled using `binary_crossentropy` as the loss function, `adam` as the optimizer, and `accuracy` as the metric.     
- The model is trained with the scaled training and target variables for 100 epochs.
- The model is evaluated with a verbose level of 2. 

<br>

**3. Results**   

<img src="images/first_eval.png" alt="Initial Model Evaluation" width="900"/>

The first attempt at optimizing the model also achieves an accuracy of around 72.5% and a loss of about 0.57. The optimized model does not reach the target accuracy of 75%. The following steps are taken to make a second attempt at optimization:  
<br>
<br>

#### Optimization Attempt 2  

- INCREASE THE NUMBER OF NEURONS TO 200 IN ALL THE NON-OUTPUT LAYERS 
- USE `relu` ACTIVATION FUNCTION FOR ALL NON-OUTPUT LAYERS
- USE `sigmoid` ACTIVATION FUNCTION FOR THE OUTPUT LAYER  
- INCREASE THE NUMBER OF EPOCHS TO 200  
<br>

**1. Data Preprocessing**

- The scaled features from the previous optimization attempt are used.      
<br >

**2. Model Design**

- A Sequential model is define using TensorFlow's Keras API.
- The planned optimization steps are implemented.
- The model is compiled, trained, and evaluated using the same steps as the previous optimization attempt.  
<br >

**3. Results** 
 <img src="images/second_eval.png" alt="Initial Model Evaluation" width="900"/> 
The second attempt at optimizing the model achieves an accuracy of around 72.4% and a loss of about 0.80. The second optimization does not reach the target accuracy of 75%, and actually performs worse than the initial and first optimization attempts. The following steps are taken to make a third attempt at optimization: 
<br> 
<br>

#### Optimization Attempt 3  

- USE KERAS TUNER `RandomSearch` TO FIND THE OPTIMAL ARCHITECTURE FOR THE MODEL  
- USE THE KERAS TUNER RECOMMENDATIONS TO BUILD AND TRAIN THE MODEL
<br>

**1. Data Preprocessing**


- The scaled features from the previous optimization attempts are used.      
<br > 

**2. Keras Tuner `RandomSearch` Setup** 
- The appropriate dependencies are imported to use the Keras Tuner.
- The `RandomSearch` object is instantiated with the following parameters: 
    - The model-building function which sets the limits for the hyperparameters, prescribes the output layer, and compiles the model.
    - The `objective` which is set to `val_accuracy` to find the model with the highest validation accuracy.
    - The maximum number of trials which is set to 10.
    - The folder name to save the trial results.
- The `search` method is called to perform the hypertuning.
- The recommended hyperparameters are defined with the `get_best_hyperparameters` method.
- The best model is defined with the `hypermodel.build` method. 
- The recommended hyperparameters are printed.
<br>

**Tuner Results**
 <img src="images/tuner_rec.png" alt="Initial Model Evaluation" width="300"/> 
<br>
<br>

**3. Model Design**

- A Sequential model is define using TensorFlow's Keras API.
- The architecture recommended by the Keras Tuner is implemented.
- The model is compiled, trained on 100 epochs, and evaluated using the same steps as the previous optimization attempts. 
<br>
<br>

**3. Results** 
 <img src="images/tuner_eval.png" alt="Initial Model Evaluation" width="900"/> 
The third attempt at optimizing the model achieves an accuracy of around 72% and a loss of about 0.57. The third optimization does not reach the target accuracy of 75%, and does not suggest that the Keras Tuner has found a better architecture than the previous attempts. 
<br> 
<br>

#### Summary
Despite various attempts to optimize the neural network model, it does not reach the target accuracy of 75%. The dataset or the model architecture may pose challenges that prevent achieving higher accuracy. It's also possible that other machine learning algorithms, such as logistic regression or ensemble methods, might perform better. 

Considering the challenges faced with the deep learning model, it is recommended to explore alternative machine learning algorithms. Logistic regression, random forest, support vector machines, or gradient boosting classifiers could be more effective for further experimentation. Additionally, feature engineering, such as creating new meaningful features, removing features that are not useful, and/or attending to outliers, might provide insights into improving model performance.

A good next step would be to combine feature engineering with Random Forest which handles non-linear relationships well and is less prone to overfitting. Random Forest also tends to perform well without need for extensive tuning.