<div style="position: absolute; top: 0; right: 0;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

# Big Mart Sales Prediction
 
## __Table Of Content__
- (A) [__Brief__](#brief)
  - [__Project__](#project)
  - [__Data__](#data)
  - [__Demo__](#demo) -> [Live Demo](https://ertugruldemir-salesregression.hf.space)
  - [__Study__](#problemgoal-and-solving-approach) -> [Colab](https://colab.research.google.com/drive/10eTjfVlGcO4uwYpYNQtNAVQQzTaFfXSy)
  - [__Results__](#results)
- (B) [__Detailed__](#Details)
  - [__Abstract__](#abstract)
  - [__Explanation of the study__](#explanation-of-the-study)
    - [__(A) Dependencies__](#a-dependencies)
    - [__(B) Dataset__](#b-dataset)
    - [__(C) Pre-processing__](#c-pre-processing)
    - [__(D) Exploratory Data Analysis__](#d-exploratory-data-analysis)
    - [__(E) Modelling__](#e-modelling)
    - [__(F) Saving the project__](#f-saving-the-project)
    - [__(G) Deployment as web demo app__](#g-deployment-as-web-demo-app)
  - [__Licance__](#license)
  - [__Connection Links__](#connection-links)

## __Brief__ 

### __Project__ 
- This is a __regression__ project that uses the  [__Big Mart Sales Dataset__](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) to __predict the sale price__.
- The __goal__ is build a model that accurately __predicts the sales price__  based on the features. 
- The performance of the model is evaluated using several __metrics__, including _MaxError_, _MeanAbsoluteError_, _MeanAbsolutePercentageError_, _MSE_, _RMSE_, _MAE_, _R2_, _ExplainedVariance_ and other imbalanced regression metrics.

#### __Overview__
- This project involves building a machine learning model to predict the sales price based on number of 12 features. The dataset contains 8523 records. The models selected according to model tuning results, the progress optimized respectively the previous tune results. The project uses Python and several popular libraries such as Pandas, NumPy, Scikit-learn.

#### __Demo__

<div align="left">
  <table>
    <tr>
    <td>
        <a target="_blank" href="https://ertugruldemir-salesregression.hf.space"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30">[Demo app] HF Space</a>
      </td>
      <td>
        <a target="_blank" href="https://colab.research.google.com/drive/1XN1KStcRMu5juG2GU39cqh5sm0QrJJ5d"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Demo app] Run in Colab</a>
      </td>
      <td>
        <a target="_blank" href="https://github.com/ertugruldmr/BigMartSalesPrediction/blob/main/study.ipynb"><img src="https://www.tensorflow.org/images/GitHub-Mark-32px.png">[Traning pipeline] source on GitHub</a>
      </td>
    <td>
        <a target="_blank" href="https://colab.research.google.com/drive/10eTjfVlGcO4uwYpYNQtNAVQQzTaFfXSy"><img src="https://www.tensorflow.org/images/colab_logo_32px.png">[Traning pipeline] Run in Colab</a>
      </td>
    </tr>
  </table>
</div>


- Description
    - __predict the sales price__  based on features.
    - __Usage__: Set the feature values through sliding the radio buttons then use the button to predict.
- Embedded [Demo](https://ertugruldemir-salesregression.hf.space) window from HuggingFace Space
    

<iframe
	src="hhttps://ertugruldemir-salesregression.hf.space"
	frameborder="0"
	width="850"
	height="450"
></iframe>

#### __Data__
- The [__Big Mart Sales Dataset__](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) from kaggle platform.
- The dataset contains 12 features, 7 categorical and 5 numerical feature.
- The dataset contains the following features:


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

| Variable                 | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| Item_Identifier          | Unique product ID                                                          |
| Item_Weight              | Weight of product                                                           |
| Item_Fat_Content         | Whether the product is low fat or not                                        |
| Item_Visibility          | The % of total display area of all products in a store allocated to the     |
|                          | particular product                                                          |
| Item_Type                | The category to which the product belongs                                    |
| Item_MRP                 | Maximum Retail Price (list price) of the product                             |
| Outlet_Identifier        | Unique store ID                                                             |
| Outlet_Establishment_Year| The year in which store was established                                      |
| Outlet_Size              | The size of the store in terms of ground area covered                        |
| Outlet_Location_Type     | The type of city in which the store is located                               |
| Outlet_Type              | Whether the outlet is just a grocery store or some sort of supermarket       |
| Item_Outlet_Sales        | Sales of the product in the particular store. This is the outcome variable to|
|                          | be predicted.                                                               |

</td></tr> </table>


<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

| Column                     | Non-Null Count | Dtype    |
| --------------------------| -------------- | --------|
| Item_Identifier            | 8523           | object  |
| Item_Weight                | 7060           | float64 |
| Item_Fat_Content           | 8523           | object  |
| Item_Visibility            | 8523           | float64 |
| Item_Type                  | 8523           | object  |
| Item_MRP                   | 8523           | float64 |
| Outlet_Identifier          | 8523           | object  |
| Outlet_Establishment_Year  | 8523           | int64   |
| Outlet_Size                | 6113           | object  |
| Outlet_Location_Type       | 8523           | object  |
| Outlet_Type                | 8523           | object  |
| Item_Outlet_Sales          | 8523           | float64 |


</td><td>

<div style="flex: 50%; padding-left: 50px;">

|                 | Item_Weight | Item_Visibility | Item_MRP | Outlet_Establishment_Year | Item_Outlet_Sales |
|-----------------|-------------|----------------|----------|---------------------------|-------------------|
| count           | 7060.000000 | 8523.000000      | 8523.000000 | 8523.000000               | 8523.000000       |
| mean            | 12.857645   | 0.066132        | 140.992782 | 1997.831867               | 2181.288914       |
| std             | 4.643456    | 0.051598        | 62.275067  | 8.371760                  | 1706.499616       |
| min             | 4.555000    | 0.000000        | 31.290000  | 1985.000000               | 33.290000         |
| 25%             | 8.773750    | 0.026989        | 93.826500  | 1987.000000               | 834.247400        |
| 50%             | 12.600000   | 0.053931        | 143.012800 | 1999.000000               | 1794.331000       |
| 75%             | 16.850000   | 0.094585        | 185.643700 | 2004.000000               | 3101.296400       |
| max             | 21.350000   | 0.328391        | 266.888400 | 2009.000000               | 13086.964800      |


</div>

</td></tr> </table>

<div style="text-align: center;">
    <img src="docs/images/target_var_dist.png" style="max-width: 100%; height: auto;">
</div>

#### Problem, Goal and Solving approach
- This is a __regression__ problem  that uses the a bank dataset [__Big Mart Sales Dataset__](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction)  from kaggle platform to __predict the sales price__ based on 12 features.
- The __goal__ is to build a model that accurately __predict the sales price__ based on the features.
- __Solving approach__ is that using the supervised machine learning models (linear, non-linear, ensemly).

#### Study
The project aimed classifying the passengers using the features. The study includes following chapters.
- __(A) Dependencies__: Installations and imports of the libraries.
- __(B) Dataset__: Downloading and loading the dataset.
- __(C) Pre-processing__: It includes data type casting, feature engineering, missing value hadnling, outlier handling.
- __(D) Exploratory Data Analysis__: Univariate, Bivariate, Multivariate anaylsises. Correlation and other relations. Roc curve, auc score, confusion matrix and related classification processes. 
- __(E) Modelling__: Model tuning via GridSearch on Linear, Non-linear, Ensemble Models.  
- __(F) Saving the project__: Saving the project and demo studies.
- __(G) Deployment as web demo app__: Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.

#### results
- The final model is __lgbm regression__ because of the results and less complexity.
<div style="flex: 50%; padding-left: 80px;">

|            | MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE          | RMSE         | MAE          | R2          | ExplainedVariance |
|----------- |-----------|------------------|-----------------------------|-------------|-------------|-------------|-------------|-------------------|
| xgbr      | 6191.649| 501.8796         | 58.602868                    | 1.142273e+06| 1068.771829| 749.535617| 0.584005   | 0.584007          |


</div>


- Model tuning results are below.

<table>
<tr><th>Linear Model</th></tr>
<tc><td>

| MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE       | RMSE       | MAE       | R2        | ExplainedVariance |
| -------- | ---------------- | --------------------------- | --------- | ---------- | --------- | --------- | ----------------- |
| lin_reg  | 0.484707         | 907.163517                  | 1.415e+06 | 6322.649   | 712.52    | 0.484637  | 1189.591482      |
| l1_reg   | 6325.649         | 710.2068                     | 103.9429  | 1.415e+06  | 1189.6583 | 0.484579  | 0.48465           |
| l2_reg   | 6326.649         | 712.2068                     | 103.9422  | 1.415e+06  | 1189.702  | 0.484541  | 0.484612          |
| enet_reg | 6326.649         | 712.7472                     | 103.9365  | 1.415e+06  | 1189.7248 | 0.484521  | 0.484593          |

</td><td> </table>


<table>
<tr><th>Non-Linear Model</th><th><div style="padding-left: 175px;">Ensemble Model</div></th></tr>

<tr><td>

| MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2         | ExplainedVariance |
|-----------|------------------|------------------------------|-------------|-------------|-------------|------------|-------------------|
| knn_reg   | 5931.649         | 616.832                      | 88.115279   | 1.477983e+06 | 1215.723344 | 0.461746   | 0.463383          |
| svr_reg   | 6600.649         | 687.332                      | 95.814126   | 1.427330e+06 | 1194.709329 | 0.480193   | 0.483724          |
| dt_reg    | 6256.649         | 497.768                      | 57.281541   | 1.168111e+06 | 1080.792046 | 0.574596   | 0.574596          |


</td><td>
<div style="flex: 50%; padding-left: 175px;">


|           | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE        | RMSE       | MAE        | R2         | ExplainedVariance |
|-----------|---------|------------------|------------------------------|------------|------------|------------|------------|-------------------|
| bag_reg   | 6059.649| 505.5350         | 58.943461                    | 1.230121e+06| 1109.108031| 776.783833| 0.552013   | 0.552408          |
| rf_reg    | 6180.649| 494.0720         | 57.939992                    | 1.152137e+06| 1073.376523| 751.628995| 0.580413   | 0.580442          |
| gbr       | 6050.649| 494.0488         | 58.697493                    | 1.157335e+06| 1075.794971| 753.156017| 0.578520   | 0.578904          |
| xgbr      | 6191.649| 501.8796         | 58.602868                    | 1.142273e+06| 1068.771829| 749.535617| 0.584005   | 0.584007          |
| lgbm_reg  | 6150.649| 504.4014         | 59.950274                    | 1.145456e+06| 1070.259994| 751.686516| 0.582846   | 0.582898          |
| catboost_reg|6497.649| 511.5482        | 72.491653                    | 1.165931e+06| 1079.782708| 776.250712| 0.575390   | 0.575482          |

</div>
</td></tr> </table>

## Details

### Abstract
- [__Big Mart Sales Dataset__](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) is used to predict the sales value. The dataset has 8523 records, 7 categorical 5 numerical totaly 12 features. The problem is supervised learning task as regression. The goal is predicting  the sales value  correctly through using supervised machine learning algorithms such as non-linear, ensemble and smilar model.The study includes creating the environment, getting the data, preprocessing the data, exploring the data, modelling the data, saving the results, deployment as demo app. Training phase of the models implemented through cross validation and Grid Search model tuning approachs. Hyperparameter tuning implemented Greedy Greed Search approach which tunes a hyper param at once a time while iterating the sorted order according the importance of the hyperparams. Models are evaluated with cross validation methods using 5 split. Classification results collected and compared between the models. Selected the basic and more succesful fraud detectore models according to unbalanced data classification metrics which is the __logistic regression__ with __Oversampling__ method. Tuned __lgbm regression__ model has __1060.35__ RMSE , __752.01__ MAE, __0.6093__ R2, __0.6093__ Explained Variance, the other metrics are also found the results section. Created a demo at the demo app section and served on huggingface space.  

### File Structures

- File Structure Tree
```bash
├── demo_app
│   ├── app.py
│   ├── cat_encods.json
│   ├── component_configs.json
│   └── lgbm_model.sav
├── docs
│   └── images
├── env
│   ├── env_installation.md
│   └── requirements.txt
├── LICENSE
├── readme.md
└── study.ipynb
```
- Description of the files
  - demo_app/
    - Includes the demo web app files, it has the all the requirements in the folder so it can serve on anywhere.
  - demo_app/cat_encods.json
    - It includes categorical encoding map.
  - demo_app/component_configs.json :
    - It includes the web components to generate web page.
  - demo_app/loj_reg_os.sav:
    - The trained (Model Tuned) model as logistic Regression instance from sklearn library on Oversampled dataset.
  - docs/
    - Includes the documents about results and presentations
  - env/
    - It includes the training environmet related files. these are required when you run the study.ipynb file.
  - LICENSE.txt
    - It is the pure apache 2.0 licence. It isn't edited.
  - readme.md
    - It includes all the explanations about the project
  - study.ipynb
    - It is all the studies about solving the problem which reason of the dataset existance.    


### Explanation of the Study
#### __(A) Dependencies__:
  -  There is a third-parth installation which is kaggle dataset api, just follow the study codes it will be handled. The libraries which already installed on the environment are enough. You can create an environment via env/requirements.txt. Create a virtual environment then use hte following code. It is enough to satisfy the requirements for runing the study.ipynb which training pipeline.
#### __(B) Dataset__: 
  - Downloading the [__Big Mart Sales Dataset__](https://www.kaggle.com/datasets/devashish0507/big-mart-sales-prediction) via kaggle dataset api from kaggle platform. The dataset has 8523  records. There are 12 features. 7 categorical feaures and 5 numerical features. For more info such as histograms and etc... you can look the '(D) Exploratory Data Analysis' chapter.
#### __(C) Pre-processing__: 
  - The processes are below:
    - Preparing the dtypes such as casting the object type to categorical type.
    - Missing value processes: Finding and handling the missing values using mean or median values of corresponding variable.
    - feature engineering processes:Creating new variables and removing unnessesary variables.
    - Outlier analysis processes: uses  both visual and IQR calculation apporachs. According to IQR approach detected statistically significant outliers are handled using boundary value casting assignment method.

      <div style="text-align: center;">
          <img src="docs/images/outliers.png" style="width: 600px; height: 150px;">
      </div>
 
#### __(D) Exploratory Data Analysis__:
  - Dataset Stats
<table>
<tr><th>Data Info </th><th><div style="padding-left: 50px;">Stats</div></th></tr>
<tr><td>

| Column                     | Non-Null Count | Dtype    |
| --------------------------| -------------- | --------|
| Item_Identifier            | 8523           | object  |
| Item_Weight                | 7060           | float64 |
| Item_Fat_Content           | 8523           | object  |
| Item_Visibility            | 8523           | float64 |
| Item_Type                  | 8523           | object  |
| Item_MRP                   | 8523           | float64 |
| Outlet_Identifier          | 8523           | object  |
| Outlet_Establishment_Year  | 8523           | int64   |
| Outlet_Size                | 6113           | object  |
| Outlet_Location_Type       | 8523           | object  |
| Outlet_Type                | 8523           | object  |
| Item_Outlet_Sales          | 8523           | float64 |


</td><td>

<div style="flex: 50%; padding-left: 50px;">

|                 | Item_Weight | Item_Visibility | Item_MRP | Outlet_Establishment_Year | Item_Outlet_Sales |
|-----------------|-------------|----------------|----------|---------------------------|-------------------|
| count           | 7060.000000 | 8523.000000      | 8523.000000 | 8523.000000               | 8523.000000       |
| mean            | 12.857645   | 0.066132        | 140.992782 | 1997.831867               | 2181.288914       |
| std             | 4.643456    | 0.051598        | 62.275067  | 8.371760                  | 1706.499616       |
| min             | 4.555000    | 0.000000        | 31.290000  | 1985.000000               | 33.290000         |
| 25%             | 8.773750    | 0.026989        | 93.826500  | 1987.000000               | 834.247400        |
| 50%             | 12.600000   | 0.053931        | 143.012800 | 1999.000000               | 1794.331000       |
| 75%             | 16.850000   | 0.094585        | 185.643700 | 2004.000000               | 3101.296400       |
| max             | 21.350000   | 0.328391        | 266.888400 | 2009.000000               | 13086.964800      |

</div>

</td></tr> </table>
  - Variable Analysis
    - Univariate analysis, 
      <div style="text-align: center;">
          <img src="docs/images/feat_dists.png" style="width: 400px; height: 200px;">
          <img src="docs/images/feat_violin.png" style="width: 400px; height: 200px;">
      </div>
    - Bivariate analysis
      <div style="text-align: center;">
          <img src="docs/images/bi_1.png" style="width: 400px; height: 300px;">
          <img src="docs/images/bi_2.png" style="width: 400px; height: 300px;">
      </div>
    - Multivariate analysis.
      <div style="text-align: center;">
          <img src="docs/images/multi_1.png" style="width: 400px; height: 300px;"> 
          <img src="docs/images/multi_2.png" style="width: 400px; height: 300px;">
      </div>
  - Other relations.
    <div style="display:flex; justify-content: center; align-items:center;">
      <div style="text-align: center;">
      <figure>
      <p>Correlation</p>
      <img src="docs/images/feat_corr_heat_map.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
       <div style="text-align: center;">
      <figure>
      <p>Correlation between target</p>
      <img src="docs/images/corr_between_features_and_target.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Variance</p>
      <img src="docs/images/feat_variance.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
      <div style="text-align: center;">
      <figure>
      <p>Covariance</p>
      <img src="docs/images/feat_cov.png" style="width: 450px; height: 200px;">
      </figure>
      </div>
    </div>

#### __(E) Modelling__: 
  - Data Split
    - Splitting the dataset via  sklearn.model_selection.train_test_split (test_size = 0.2).
  - Util Functions
    - Greedy Step Tune
      - It is a custom tuning approach created by me. It tunes just a hyperparameter per step using through GridSerchCV. It assumes the params ordered by importance so it reduces the computation and time consumption.  
    - Model Tuner
      - It is an abstraction of the whole training process. It aims to reduce the code complexity. It includes the corss validation and GridSerachCV approachs to implement training process.
    - Learning Curve Plotter
      - Plots the learning curve of the already trained models to provide insight.
  - Linear Model Tuning Results _without balanciy process_
    - linear, l1, l2, enet regressions
    - Cross Validation Scores
      | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE       | RMSE       | MAE       | R2        | ExplainedVariance |
      | -------- | ---------------- | --------------------------- | --------- | ---------- | --------- | --------- | ----------------- |
      | lin_reg  | 0.484707         | 907.163517                  | 1.415e+06 | 6322.649   | 712.52    | 0.484637  | 1189.591482      |
      | l1_reg   | 6325.649         | 710.2068                     | 103.9429  | 1.415e+06  | 1189.6583 | 0.484579  | 0.48465           |
      | l2_reg   | 6326.649         | 712.2068                     | 103.9422  | 1.415e+06  | 1189.702  | 0.484541  | 0.484612          |
      | enet_reg | 6326.649         | 712.7472                     | 103.9365  | 1.415e+06  | 1189.7248 | 0.484521  | 0.484593          |
    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/linear_regression_f_imp.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_regs_f_imp.png" style="width: 450px; height: 200px;">
      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/linear_regression_l_cur.png" style="width: 150px; height: 200px;">
          <img src="docs/images/lin_regs_l_cur.png" style="width: 450px; height: 200px;">
      </div>
  - Non-Linear Models
    - Logistic Regression, Naive Bayes, K-Nearest Neighbors, Support Vector Machines, Decision Tree
    - Cross Validation Scores _without balanciy process_
      | MaxError   | MeanAbsoluteError | MeanAbsolutePercentageError | MSE         | RMSE        | MAE         | R2         | ExplainedVariance |
      |-----------|------------------|------------------------------|-------------|-------------|-------------|------------|-------------------|
      | knn_reg   | 5931.649         | 616.832                      | 88.115279   | 1.477983e+06 | 1215.723344 | 0.461746   | 0.463383          |
      | svr_reg   | 6600.649         | 687.332                      | 95.814126   | 1.427330e+06 | 1194.709329 | 0.480193   | 0.483724          |
      | dt_reg    | 6256.649         | 497.768                      | 57.281541   | 1.168111e+06 | 1080.792046 | 0.574596   | 0.574596          |

    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/non_lin_reg_f_imp.png" style="width: 800px; height: 200px;">

      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/non_lin_reg_l_cur.png" style="width: 400px; height: 300px;">
      </div>


  - Ensemble Models
    - Random Forest, Gradient Boosting Machines, XGBoost, LightGBoost, CatBoost
    - Cross Validation Scores _without balanciy process_
      |           | MaxError | MeanAbsoluteError | MeanAbsolutePercentageError | MSE        | RMSE       | MAE        | R2         | ExplainedVariance |
      |-----------|---------|------------------|------------------------------|------------|------------|------------|------------|-------------------|
      | bag_reg   | 6059.649| 505.5350         | 58.943461                    | 1.230121e+06| 1109.108031| 776.783833| 0.552013   | 0.552408          |
      | rf_reg    | 6180.649| 494.0720         | 57.939992                    | 1.152137e+06| 1073.376523| 751.628995| 0.580413   | 0.580442          |
      | gbr       | 6050.649| 494.0488         | 58.697493                    | 1.157335e+06| 1075.794971| 753.156017| 0.578520   | 0.578904          |
      | xgbr      | 6191.649| 501.8796         | 58.602868                    | 1.142273e+06| 1068.771829| 749.535617| 0.584005   | 0.584007          |
      | lgbm_reg  | 6150.649| 504.4014         | 59.950274                    | 1.145456e+06| 1070.259994| 751.686516| 0.582846   | 0.582898          |
      | catboost_reg|6497.649| 511.5482        | 72.491653                    | 1.165931e+06| 1079.782708| 776.250712| 0.575390   | 0.575482          |

    - Feature Importance
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_f_imp.png" style="width: 800px; height: 200px;">

      </div>
    - Learning Curve
      <div style="display:flex; justify-content: center; align-items:center;">
          <img src="docs/images/ensemble_l_curve.png" style="width: 800px; height: 400px;">
      </div>

#### __(F) Saving the project__: 
  - Saving the project and demo studies.
    - trained model __lgbm_model.sav__ as pickle format.
#### __(G) Deployment as web demo app__: 
  - Creating Gradio Web app to Demostrate the project.Then Serving the demo via huggingface as live.
  - Desciption
    - Project goal is predicting the sales price based on four features.
    - Usage: Set the feature values through sliding the radio buttons and dropdown menu then use the button to predict.
  - Demo
    - The demo app in the demo_app folder as an individual project. All the requirements and dependencies are in there. You can run it anywhere if you install the requirements.txt.
    - You can find the live demo as huggingface space in this [demo link](https://ertugruldemir-salesregression.hf.space) as full web page or you can also us the [embedded demo widget](#demo)  in this document.  
    
## License
- This project is licensed under the Apache 2.0 License. See the [LICENSE](LICENSE) file for details.

<h1 style="text-align: center;">Connection Links</h1>

<div style="text-align: center;">
    <a href="ertugrulbusiness@gmail.com"><img src="https://ssl.gstatic.com/ui/v1/icons/mail/rfr/gmail.ico" height="30"></a>
    <a href="https://tr.linkedin.com/in/ertu%C4%9Fruldemir?original_referer=https%3A%2F%2Fwww.google.com%2F"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/linkedin/linkedin-original.svg" height="30"></a>
    <a href="https://github.com/ertugruldmr"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/github/github-original.svg" height="30"></a>
    <a href="https://www.kaggle.com/erturuldemir"><img src="https://cdn.jsdelivr.net/gh/devicons/devicon/icons/kaggle/kaggle-original.svg" height="30"></a>
    <a href="https://huggingface.co/ErtugrulDemir"><img src="https://huggingface.co/front/assets/huggingface_logo-noborder.svg" height="30"></a>
    <a href="https://stackoverflow.com/users/21569249/ertu%c4%9frul-demir?tab=profile"><img src="https://upload.wikimedia.org/wikipedia/commons/e/ef/Stack_Overflow_icon.svg" height="30"></a>
    <a href="https://www.hackerrank.com/ertugrulbusiness"><img src="https://hrcdn.net/fcore/assets/work/header/hackerrank_logo-21e2867566.svg" height="30"></a>
    <a href="https://app.patika.dev/ertugruldmr"><img src="https://app.patika.dev/staticFiles/newPatikaLogo.svg" height="30"></a>
    <a href="https://medium.com/@ertugrulbusiness"><img src="https://upload.wikimedia.org/wikipedia/commons/a/a5/Medium_icon.svg" height="30"></a>
    <a href="https://www.youtube.com/channel/UCB0_UTu-zbIsoRBHgpsrlsA"><img src="https://upload.wikimedia.org/wikipedia/commons/thumb/0/09/YouTube_full-color_icon_%282017%29.svg/1024px-YouTube_full-color_icon_%282017%29.svg.png" height="30"></a>
</div>

