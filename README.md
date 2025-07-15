## Analysis Plan 
The analysis will include steps to understanding the implication of different medical and lifestyle factors on a positive cancer diagnosis. This will allow for insights into factors that are both significant and irrelevant to the likelihood positive cancer diagnosis. However, mainly will enable a model to be built that will accurately classify an individual as having cancer or not having cancer, according to the datasets target variable (no cancer, cancer)
The analysis will include steps to understanding both the individual and combined implication of lifestyle factor on insurance charges for a medical scheme member. Will uncover the lifestyle factors that are both significant and irrelevant to the increase in insurance chargers and primarily build a model to generate insurance chargers. 
## Explanatory Data Analysis (EDA): 
Data exploration is a crucial step that needs to be taken before attempting to build a model. EDA helps an analysist to understand the given data. The EDA process identifies issues that may affect machine learning, ensures that data will be of suitable quality and identifies key patterns and trends that may need to be taken into consideration when building a predictive model (Ray, 2024). Various functions from the Pandas, Seaborn and Matplotlib libraries will be used to assist the EDA process. The EDA will include the following steps.

**Data Collection** : The first step to analysing the data is to retrieve it from the given csv file. The `Pandas.read_csv()` method will be used to read the data from the file and store is within a Pandas data frame object. The Pandas data frames can hold huge amount of data in excel like format, making it ideal to store the dataset and perform data manipulation (NumFOCUS Inc, 2024). 

**Data Cleaning**: To help ensure data quality missing vales, duplicates and inconsistencies in data types must be handled.
- Missing Values:` panda.isnull().sum()` will be used to search the dataset and return the sum of missing values for each column. Identified values will either be dropped or replaced with relevant columns average.
- Duplicated data: `panda.duplicated()` will be used to identify and create a list of all duplicates within the dataset. The list of duplicates will then be removed from the dataset to avoid introducing bias. 
- Data type consistency: `panda.info()` will be used to identify data types in the given dataset. This is important to ensure all data is numerical and as input for a linear algorithm. Data types that are not quantitative will be encoded or transformed to numeric form.   

**Data Transformation**: To use machine learning algorithms, all data must be in numerical form, either float or int data types. The categorical features sex, smoker and region must therefore be transformed. 
- Encode data: Features in the dataset that are the object data type will be encoded to transform to numeric form. LabelEncoder from the Scikit-learn preprocessing module will convert the values within sex, smoker and region features. The features will contain numbers to label each different category (Novogroder, 2024).
    
## Univariate Analysis: 
This stage of analysis focuses on analysing a single feature of the dataset at a time. Depending on whether a feature is categorical or numerical the analysis will differ (Ray, 2024). For features that are continuous values, assessment will include statistical summary to understand the distribution, central tendency and the variability for each feature within the dataset (Ray, 2024). Features that are categorical the frequency and count of the feature will be assessed using visualisation such as bar chart or count plots (Ray, 2024). 

## Bivariate Analysis: 
This stage of analysis focuses on two variables, to uncover the association and distribution between them (Ray, 2024). To understand the relationship between two variables visualisation tools such as seaborn and matplotlib will be used to create correlation heatmaps, pair plots, distribution plots and scatterplots.

## Multivariate Analysis:
This is analysis extends the bivariate analysis but is carried out on more than two features. The analysis stage is aimed at understanding the relationship between multiple variables. Tools from seaborn and matplotlib will be used to visualise the relations between variables. Plots such as distribution plots and scatterplots will carry out this stage (Deepanshi, 2023; KIm, 2023). 

## Pre Processing: 
Preprocessing are the final tasks that must be carried out to ensure that the data meets all requirements to be useful machine learning purposes (Novogroder, 2024). Tasks include scaling, encoding and splitting data to ensure that the model performs well (Novogroder, 2024). This stage will include various modules from the Scikit-learn library to prepare the data for processing.

**Data scaling**:Each feature in the dataset may have a different scale, this leads the model  to perceive features with larger values to have greater significance and produce inaccurate predictions. Scaling the dataset ensure that features have fair contribution. The Scikit `MinMaxScaler()` and `StandardScaler()` methods will be considered to scale the features within the dataset (GeeksforGeeks, 2025a).  

**Feature Selection**: Feature selection will be performed through either backward elimination or `sklearn.selectKBest()`. This is important as to only include features that offer meaning contribution to the prediction. By only including significant  features the performance and accuracy of the model will be enhanced (Imarticus, 2024) .

**Splitting data**: Data will be split before input into model. The data will be split into an 80:20 ratio, 80 percent of the data being used as training data and the remaining 20 percent for test data. This is an important step so that after training the model can be tested on unseen data to assess its performance (Gillis, 2024). 

## Build and Evaluate Model
**Model training**: The models used for this analysis will be the Scikit-learn Logistic Regression model. 

**Evaluation**: To evaluate the performance of the models, the Scikit-learn library will be used. The library contains a metrics method that provide the classification model performance metrics, such as accuracy, precision, recall, and f1 score (Martin Ward Powers, 2011). These metrics can be used to assess the performance and accuracy of a model.

**Visualisation of results**: Seaborn and Matplotlib will be used to display the result (Deepanshi, 2023; KIm, 2023). Plots such as: 
- Confusion matrix, to assess a classification models performance, by comparing the predicted values against the actual values (Murel and Kavlakoglu, 2024b).
- Precision Recall Curve, to assess the performance of a model, by analysing its ability to predict the positive class. This plot is used to assess the performance of a classification model when the target variable imbalanced (Tripathi, 2022).


## Exploratory Data Analysis
**Data Collection**: The data was first retrieved form the given csv file using `Pandas.read_csv()` method as displayed in figure 1. The data was stored in a Pandas data frame, theses data frames are able to store large amount of data in excel like format, making it ideal to store insurance dataset and perform data manipulation (NumFOCUS Inc, 2024). The `Pandas.head()` method was used to confirm data was successfully retrieved and assess rows and columns of the data. 
<div align="center">
	  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/cbf4b852-51dc-42cc-bb94-fee8b9f94999" />
  <br>
  <em>Figure 1: Storing Data</em>
</div>

**Data Cleaning:**: `panda.isnull().sum()` was used to the dataset for missing values and return the sum of missing values for each column. There were no missing values found in the dataset, this can be seen in figure 2 below. 


<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/d70a516a-c47d-43b8-8230-75b138d60908" />
	<br>
  <em>Figure 1: Missing Values</em>
</div>

To find duplicates in data `panda.duplicated()` method was used to identify and create a list of all duplicates within the dataset.The dataset did not contain duplicate values. This can be seen in figure 3 below.  
<div align="center">	
	  <img width="700" height="300" alt="image" src="https://github.com/user-attachments/assets/5b0d8d15-0bd1-46db-8dc7-5ee9b8c2a06c" />
	<br>
  <em>Figure 3: Duplicates </em>
</div>

To identify the data types `panda.info()`. The method returned the below information about the dataset features. The method found 9 features within the dataset, which were a combination of ints and floats, as shown in below in figure 4. 
- 2 of the columns, age and children are int datatypes and therefore suitable for a machine learning algorithms input. 
- 2 of the columns, charges and BMI are floats datatypes are too suitable as input for machine learning algorithms.
- 3 of the column’s datatypes are objects or string values which need to be converted to numerical values for to be used as input for the machine learning algorithms.
<div align="center">
	<img width="550" height="350" alt="image" src="https://github.com/user-attachments/assets/7af62f02-222e-42a1-9fa1-617b26672e2e" />
	<br>
  <em>Figure 4: Datatypes </em>
</div>

**Check Outliers**:  The dataset was checked for outliers using the quartile method. Each continuous feature was passed through a for loop which calculated the upper and lower quantile for the feature. No outliers were found in the dataset, this can be seen below in figure 5.
<div align="center">
	<img width="700" height="200" a alt="image" src="https://github.com/user-attachments/assets/0e609af9-d58e-43df-863d-13a749d94f8e" />
	<br>
  <em>Figure 5: Check Outliers</em>
</div>

**Univariate Analysis**  
This stage of analysis focused on analysing individual features in the dataset. For features with continuous values, the `pandas.describe()` method was used to print descriptive statistics. The summary uncovered the distribution, central tendency, and variability of each numeric feature in the dataset (Ray, 2024).  
Figure 6 displays the descriptive statistics:

<div align="center">
	  <img width="800" height="429" alt="image" src="https://github.com/user-attachments/assets/d6b82538-6fe0-423c-ad9f-eee3fbc87eb0" />
  <br />
  <em>Figure 6: Descriptive Statistics</em>
</div>

---

- **Age Feature**:  
The mean (50.32) is very close to the median (51.00), this suggests the distribution of data is approximately symmetrical. The std (17.64) is moderate relative to the mean, inferring a moderate variability within the datasets spread of age. This variability is supported by the 25th and 75th percentiles, which displays 50% of individual's age falls between 35 and 66 years old. Thus, the dataset contains a wide spread of ages, with majority of individuals middle-aged to old.

- **BMI**:  
BMI's mean (27.51) and median (27.60) are also in close proximity. This suggests the distribution of data is approximately symmetrical. Further, on average individuals are overweight. The std (7.23) is low considering the features mean, implying low variability within the spread of dataset feature. This variability is supported by the 25th and 75th percentiles, which shows that 50% of individuals BMI falls between 21.48 and 33.85. Thus, majority of individuals in the dataset are overweight to obese.

- **Physical Activity**:  
The mean (4.89) and median (4.83) are in close proximity. This suggests a near symmetrical distribution of data. Additional, on average individuals spend 4 hours physical active. The std (2.87) is moderate relative the features mean, indicating moderate variability in the spread of data. Furthermore, the min (0.00) and max (10.00) indicate activity ranges from inactive to highly active.  This variability is supported by the 25th and 75th percentiles, which indicates that 50% of individuals are physical active for 2.43 to 7.40 hours per day. Thus, the dataset contains wide range of physical activity, with a moderate portion of individuals sedentary to highly active.

- **Charges**:  
The mean (2.42) and median (2.38) are in close vicinity, on average individuals consume approx.  2.5 units of alcohol a week. This suggests a near symmetrical distribution of data. The std (1.41) is moderate relative the mean, indicating moderate variability within the spread of data. The min value is 0.00 and max 5, signalling alcohol behaviour ranges from sober to indulgent.  The 25th and 75th percentiles supporters the variability, which indicates 50% of individuals drink 1.21 to 3.59 unit of alcohol a week. Thus, majority of individuals in the dataset have a moderate alcohol Intake.


Univariate analysis for categorical features was carried out using visualisation tools from seaborn and matplotlib. Seaborn count plots were used to understand the distribution genetic risk, cancer history, smoking and gender in the dataset. The below figure 7 shows the count plot for genetic risk levels in the dataset. The count plot illustrates the frequency distribution of genetic cancer risk within the 1500 individual’s sample. The plot indicates, majority of individuals fall into the low-risk category (approx. 900). The moderate amount (approx. 450) of individuals are categorized as medium risk and s small subset (approx. 150) are categorized as high risk.

<div align="center">
	<img width="800" height="535" alt="image" src="https://github.com/user-attachments/assets/15c4d5c4-793e-48c0-9964-43f06529a13c" />
	<br>
  <em>Figure 7: Genetic Risk Distribution</em>
</div>

**Multivariate Analysis**: This stage of analysis focuses on uncovering the association and distribution between variables in the dataset. To understand the relationship between two variables visualisation tools such as seaborn and matplotlib will be used to create correlation heatmaps, pair plots and distribution plots. 
<div align="center">
	<img width="759" height="588" alt="image" src="https://github.com/user-attachments/assets/ecc2d75b-98ed-4589-ba0f-bcd42f1dfe6f" />
	<br>
  <em>Figure 8: Correlation Heatmap</em>
</div>

Figure 8 above display a correlation heat map. This was used to shows the correlation between all variables in a dataset. The Pandas method `dataframe.corr()` is used to calculate the correlation coefficient for each variable in the dataset ((NumFOCUS Inc, 2025)). The method returns a float between 1 and -1, indicating the correlation between each pair of variables (NumFOCUS Inc, 2025). 1 indicates perfect correlation, 0 indicates no correlation and -1 perfect negative correlation. This visual the matrix of correlations using the seaborn heatmap (Jain, 2024).   

---

**Notable insights were**:
- Diagnosis has a weak positive correlation with Cancer History, Alcohol Intake, Smoking, Gender, Age, BMI and Genetic Risk features. If one of these categorical features are true (yes=1), the likelihood of a positive cancer diagnosis increases.
- Diagnosis has a weak negative correlation with Physical Activity. For member that are physical activity in the day, there is a slightly lower chance of a positive cancer diagnosis
- 0.39 is the highest correlation between Diagnosis and Cancer History. However, it is a moderate correlation.
---
The below plot provides in figure 9  provides an insight into the distribution of smoking and a non-smoking individual, according their cancer diagnosis.
<div align="center">
	<img width="560"  height="500"  alt="image" src="https://github.com/user-attachments/assets/1a25f59c-1548-4672-88d9-78343faf2208" />
	<br>
  <em>Figure 10: Average Charges for Smokers and Non-Smokers</em>
</div>

---

Non-smoker:  
- The bar for non-smokers displays, a greater proportion of individuals have a negative cancer diagnosis and a smaller subset a positive cancer diagnosis.
- For the smoker population, 762 individuals do not have cancer (green), while 334 individuals do have cancer (blue).
- This indicates that non-smokers are more likely to have a negative cancer diagnosis.

Smoker:
- The bar for smoker’s displays, a greater proportion of individuals have a positive cancer diagnosis, while a smaller subset a negative cancer diagnosis
- For the smoker population of 404, 223 individuals have cancer (blue), and 181 individuals do not have cancer (green).
- There is a small difference (42) between a positive cancer diagnosis and a negative cancer diagnosis, this indicates that smoking has a potential association with a have a slightly higher probability of a positive diagnosis.   

Observations suggests that smoking status has a possible association of a positive cancer diagnosis. However, there is a small difference (111) between non-smoking and smoking individuals that have cancer. Therefore, smoking may not be a strong predictor for cancer.

The bar plot in figure 11, provides an insight into the distribution of genetic risk levels, according to cancer diagnosis.
<div align="center">
	<img width="560"  height="500" alt="image" src="https://github.com/user-attachments/assets/1d794596-6ac1-4854-82d8-b4832e5a7b7a" />	<br>
  <em>Figure 11: Genetic Risk by Cancer Diagnosis </em>
</div>

---

#### Low Risk:
- The bar for low genetic risk shows a greater proportion of individuals have a negative cancer diagnosis, while a smaller subset a positive cancer diagnosis.
- For the low-risk population, 282 individuals have cancer (blue), and 613 individuals do not have cancer (green).
- This indicates that individuals with a low genetic risk are more likely to have a negative cancer diagnosis.  
#### Medium Risk:
- The bar for medium genetic risk shows a greater proportion of individuals have a negative cancer diagnosis, while a smaller subset a positive cancer diagnosis.
- For the medium risk population, 142 individuals have cancer (blue), and 305 individuals do not have cancer (green).
- This indicates that individuals with a medium genetic risk are more likely to have a negative cancer diagnosis.   
#### High Risk:
- The bar for high genetic risk shows a greater proportion of individuals have a positive cancer diagnosis, while a smaller subset a negative cancer diagnosis.
- For the high risk population of 158 , 133 individuals have cancer (blue), and 25 individuals do not have cancer (green).
- There is a sizeable difference (108) between individuals who have cancer and do not. This suggests that high genetic risk has a potential association with positive cancer diagnosis.

Observations suggests that low and medium genetic risk may not be strong predictors for cancer. However, high genetic risk shows a strong association with cancer and may be a strong predictor.


The below distribution plots in figure 12, provide insight into the distribution 'Physical Activity' by 'Cancer' diagnosis.
<div align="center">
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/11c8eb2e-9628-4d81-9ac5-7459f6f28301" />
	<br>
  <em>Figure 12: Physical Activity Distribution by Cancer Diagnosis</em>
</div>

---

No Cancer (Green): 
- The distribution for 'No Cancer’ spans the full range of on the x-axis (0 to 10 hr/day). This is seen by the widespread distribution with flattened peaks.
- The distribution of the plot has two slight peaks, falling at approximately 4.5 to 5.5 hours and 8.3 to 9.5 hours.  This suggests a higher density of moderate to highly active individuals.
- The distribution is slightly skewed to the left. Peaks are at higher physical activity range and a long tail extends towards lower levels of activity.
- The plot suggests that individuals with no cancer have moderate to high physical activity level.
Cancer (Red):
- The distribution for 'Cancer' also spans the full range of on the x-axis (0 to 10 hr/day).
- In contrast, the distribution has a defined peak concentrated at low levels of activity. Ranging from approximately 0 to 2.5 hours of activity. This suggests a higher density of individuals with low physical activity.
- The distribution is right-skewed, as the peak is at a low physical activity range and a long tail extends towards higher levels of activity.
- The plot suggests that individuals with cancer have low levels of physical activity.
The no cancer distribution (red) compared to cancer distribution (green) is denser in higher ranges of activity. Furthermore, the cancer distribution (green) is highly concentrated around low levels of activity. Observations suggest that lower level of physical activity are  associated with cancer and may be a strong predictor.

The below distribution plots in figure 13, provide insight into the distribution 'BMI' by 'Cancer' diagnosis.
<div align="center">
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/fa8e20bc-da87-415e-bec3-b9d462a3de30" />
	<br>
  <em>Figure 12: BMI Distribution by Cancer Diagnosis</em>
</div>

---
No Cancer (Green):
- The distribution for 'No Cancer’s spans all BMI values on the x-axis (0 to 40). This is seen by the widespread distribution.
- The distribution of the plot has a broad peak, concentrated at a BMI range of approximately 17 to 24. This indicates a higher density of individuals that are underweight and healthy.
- The distribution is right-skewed, as peaks are at lower BMI ranges and a long tail extends towards the higher range. This indicates less individuals with a high BMI.
- The right skewed distribution suggests individuals with no cancer are more likely to have a lower BMI. 
Cancer (Red) 
- The distribution for 'Cancer' is also spans across all values on the x-axis (0 to 40). Seen by the widespread distribution.
- The distribution has a broader peak concentrated at higher BMI ranges. The peak fall at BMI's of approximately 26 to 36. This indicates a higher density of individuals that are overweight and obese.
- The distribution is skewed to the left, as peaks are at a high BMI range and a long tail extends towards the lower range. This indicates less individuals with a low BMI.
- The left skewed distribution suggests individuals with cancer are more likely to have a higher BMI. 
Individuals with cancer tend to be overweight or obese. Observations suggest that a higher BMI is associated with cancer and may be a strong predictor.


**Pre Processing**: The final tasks carried out before training the model included scaling, encoding and splitting data to ensure that the model performs well (Novogroder, 2024). Various modules from the Scikit-learn library were used to prepare the data for processing.

Feature Selection: Feature selection was performed by `sklearn.selectKBest()` SelectKBest was used with the `mutual_info_classif` parameter and a max output of 7. The method selected the best based upon the F-value, which indicates the variance between each feature and the dependent variable. The features are scored according to the significance of their variance (Kavya D, 2023). The selected features can be seen below in figure 15.
<div align="center">
	<img width="300" height="280" alt="image" src="https://github.com/user-attachments/assets/aae127f6-7913-4698-be09-a217e97b8749" />
	<br>
  <em>Figure 15: Selected Features</em>
</div> 

Data scaling: The Scikit `StandardScaler()` methods was used to scale features within the dataset in preparation for model training. This was to ensure larger values such as BMI, Age and Alcohol Intake did not have disproportionate significance in the model which would result in inaccurate predictions. A pipeline was used to standardised and train the model.

Splitting data: The data used for model training was split into a 75:25 ratio, 75 percent of the data being used as training data and the remaining 25 percent for test data. This is an important step so that after training the model can be tested on unseen data to assess its performance (Gillis, 2024). random state of 23 was used to ensure reproducibility and the stratify parameter was used to class distribution is preserved in both training and testing sets

---

**Build and Evaluate Model**:The classification model trained and built using the Scikit-learn Logistic Regression model. A pipeline was used to standardised and then train the model. The model was trained using the `class_weight='balanced` parameter. This was to account for imbalanced target variable, Diagnosis. This allowed the model to assign appropriate weights to each classification during the training process, to help reduce bias towards No cancer as the majority classification.

To evaluate the performance of the models, the Scikit-learn library will be used. The library contains a metrics method to that provide classification model performance metrics, such as accuracy, precision, recall, and f1 score (Martin Ward Powers, 2011).  Figure 18 show the results of the model.
- Accuracy: The score is a measure of the model’s overall ability to produce correct predictions. It calculates the ratio of correct predictions over the total number of predictions made (Martin Ward Powers, 2011).
- Precision: This score is an indication of the model’s ability to correctly produce and identify positive predictions. The measure evaluates the number of true positive predictions out of all the prediction that were classified as positive (Martin Ward Powers, 2011). 
- Recall: This score is an indication of how well the model can identify positive classifications. The measure evaluates the number of actual positive outcomes that were correctly predicted as positive by the model (Martin Ward Powers, 2011).
- F1-score: This score combined the average precision and recall. Effectively the model balances he precision and recall. It evaluates the overall performance of a model when the positive and negative classifications are unbalanced (Martin Ward Powers, 2011). 
<div align="center">
	<img width="396" height="228" alt="image" src="https://github.com/user-attachments/assets/3ce5e907-1ebc-4baa-8a59-f3fda995d6e5" />
	<br>
  <em>Figure 18: Model Results</em>
</div> 

---

| Metric     | Description |
|------------|-------------|
| **Accuracy**  | The model correctly predicted the diagnosis for **82%** of all instances (both cancer = 1 and no cancer = 0). |
| **Precision** | Out of all predicted positive cancer diagnoses, **75%** were actually correct (cancer = 1). |
| **Recall**    | The model correctly identified **79%** of actual positive cancer cases (cancer = 1). |
| **F1-score**  | The F1 score of **76%** indicates a good balance between correctly identifying positive cancer diagnoses and limiting false positives. |

The model was also evaluated using cross validation, using a KFold. The below figure 19 shows the results.

<div align="center">
	<img width="940" height="381" alt="image" src="https://github.com/user-attachments/assets/d5cb5138-b606-4333-8e4a-5896cf1d4e3c" />
	<br>
  <em>Figure 19: KFold  Results</em>
</div> 
The cross validation indicates that the model’s performance remains consistent across different fold of data. Therefore, model can generalise well on unseen data.

---

| Metric     | Description |
|------------|-------------|
| **Accuracy**  | The model correctly predicted the diagnosis for **82%** of all instances (both cancer = 1 and no cancer = 0). |
| **Precision** | Out of all predicted positive cancer diagnoses, **74%** were actually correct (cancer = 1). |
| **Recall**    | The model correctly identified **82%** of actual positive cancer cases (cancer = 1). |
| **F1-score**  | The F1 score of **77%** indicates a good balance between correctly identifying positive cancer diagnoses and limiting false positives. |


A confusion matrix is used to assess a classification models performance, by comparing the predicted values against the actual values. In a grid, it displays the number of incorrect and correct predictions made by the model.  Which enables a visual illustration of the model performance evaluation metrics, such as accuracy, recall, precision and F1 score (Murel and Kavlakoglu, 2024). Seen in figure 20 below. 
<div align="center">
	<img width="686" height="548" alt="image" src="https://github.com/user-attachments/assets/0666802a-1f73-462a-a1f1-2fcd0af59f38" />
	<br>
  <em>Figure 19: KFold  Results</em>
</div> 

---

- **True Positive (TP)(Top-left box)**: The model was able to correctly predict 782 instances of a positive cancer diagnosis.
782 individuals who did have cancer were correctly identified as having cancer.
- **False Negative (FN)(Top-right box)**: The model incorrectly predicted 162 instances of a negative cancer diagnosis.162 individuals who had cancer were misclassified, as not having cancer. 
- **False Positive (FP)(Bottom-left box)**:The model incorrectly predicted 102 instances of a positive cancer diagnosis.
102 individuals who did not have cancer were misclassified, as having cancer
- **True Negative (TN)(Bottom-right box)**:The model was able to correctly predict 455 instances of a negative cancer diagnosis.
455 individuals who did not have cancer were correctly identified as not having cancer.

The Precision-Recall Curve was used to assess the performance of a model, by analysing its ability to predict the positive class. This makes it useful for unbalanced datasets. The Precision-Recall Curve evaluates the relationship between precision and recall at different thresholds, from 0 to 1. The curve is plotted by estimates of precision-recall pair at these various thresholds. To provide insight into the trade-off between the two metrics. This assists in selecting the best threshold that balances precision and recall in the model (Lyzer, 2024). As seen the below figure 26.
<div align="center">
	<img width="700" height="711" alt="image" src="https://github.com/user-attachments/assets/d8cd76fb-d9ad-4b53-a075-ffd076a3d12e" />
	<br>
  <em>Figure 19: KFold  Results</em>
</div> 

---

- The x-axis represents recall, the proportion of the positive class that was correctly predicted.
- The y-axis represents precision, the total number of the positive class that model has predicted.
- The plots shows that the model has a high precision even as recall increases, only dropping off at very high recall. The area under the curve is large, this indicates the model performance well.

Based on the performance results, this model will be able to accurately predict the likelihood of a member having cancer or not. The model will therefore enable the medical scheme to apply the dreaded disease benefits in an effective manner.


## References
Ajaykumar, 2024. Data Science: Guide to Encoding Nominal Categorical Features. | by Ajaykumar Dev | Medium. [online] Available at: <https://medium.com/@nikaljeajay36/data-science-guide-to-encoding-nominal-categorical-features-bf3e622b1133> [Accessed 23 April 2025].
Akinkugbe, A., 2024. When to Use Linear Regression. Introduction | by Ayo Akinkugbe | Medium. [online] Available at: <https://medium.com/@ayoakinkugbe/when-to-use-linear-regression-6b7057ebd01f> [Accessed 23 April 2025].
Anomalo, 2024. Best Practices for Data Quality in Machine Learning. [online] Available at: <https://www.anomalo.com/blog/data-quality-in-machine-learning-best-practices-and-techniques/> [Accessed 24 April 2025].
Bechtel, M., 2025. Measuring the Strength of Linear Associations with a Correlation Coefficient | Statistics and Probability | Study.com. [online] Available at: <https://study.com/skill/learn/measuring-the-strength-of-linear-associations-with-a-correlation-coefficient-explanation.html> [Accessed 25 April 2025].
Bevans, R., 2020. Simple Linear Regression | An Easy Introduction & Examples. [online] Available at: <https://www.scribbr.com/statistics/simple-linear-regression/> [Accessed 25 April 2025].
Bhandari, A., 2025. Multicollinearity Explained: Causes, Effects & VIF Detection. [online] Available at: <https://www.analyticsvidhya.com/blog/2020/03/what-is-multicollinearity/> [Accessed 25 April 2025].
DATAtab Team, 2025. Linear Regression: A Complete Guide to Modeling Relationships Between Variables. [online] Available at: <https://datatab.net/tutorial/linear-regression> [Accessed 25 April 2025].
Deepanshi, 2023. Linear Regression | Introduction to Linear Regression for Data Science. [online] Available at: <https://www.analyticsvidhya.com/blog/2021/05/all-you-need-to-know-about-your-first-machine-learning-model-linear-regression/> [Accessed 25 April 2025].
GeeksforGeeks, 2024. Assumptions of Linear Regression | GeeksforGeeks. [online] Available at: <https://www.geeksforgeeks.org/assumptions-of-linear-regression/> [Accessed 25 April 2025].
GeeksforGeeks, 2025a. Feature Engineering: Scaling, Normalization, and Standardization | GeeksforGeeks. [online] Available at: <https://www.geeksforgeeks.org/ml-feature-scaling-part-2/> [Accessed 25 April 2025].
GeeksforGeeks, 2025b. Multiple Linear Regression with Backward Elimination | GeeksforGeeks. [online] Available at: <https://www.geeksforgeeks.org/ml-multiple-linear-regression-backward-elimination-technique/> [Accessed 25 April 2025].
Gillis, A., 2024. What is data splitting and why is it important? [online] Available at: <https://www.techtarget.com/searchenterpriseai/definition/data-splitting> [Accessed 25 April 2025].
Gorelick, M.H., 2006. Bias arising from missing data in predictive models. Journal of Clinical Epidemiology, [online] 59(10), pp.1115–1123. https://doi.org/10.1016/j.jclinepi.2004.11.029.
Imarticus, 2024. A Guide to Feature Selection for Linear Regression Models. [online] Available at: <https://imarticus.org/blog/linear-regression-models/> [Accessed 25 April 2025].
Kavya D, 2023. Optimizing Performance: SelectKBest for Efficient Feature Selection in Machine Learning | by Kavya D | Medium. [online] Available at: <https://medium.com/@Kavya2099/optimizing-performance-selectkbest-for-efficient-feature-selection-in-machine-learning-3b635905ed48> [Accessed 25 April 2025].
KIm, D., 2023. How to plot Predicted vs Actual Graphs and Residual Plots | by Dooinn KIm | Medium. [online] Available at: <https://dooinnkim.medium.com/how-to-plot-predicted-vs-actual-graphs-and-residual-plots-dc4e5b3f304a> [Accessed 25 April 2025].
Minitab, 2024. What are categorical, discrete, and continuous variables? - Minitab. [online] Available at: <https://support.minitab.com/en-us/minitab/help-and-how-to/statistical-modeling/regression/supporting-topics/basics/what-are-categorical-discrete-and-continuous-variables/> [Accessed 23 April 2025].
Novogroder, I., 2024. Data Preprocessing in Machine Learning: Steps & Best Practices. [online] Available at: <https://lakefs.io/blog/data-preprocessing-in-machine-learning/> [Accessed 25 April 2025].
NumFOCUS Inc, 2024. pandas.DataFrame.corr — pandas 2.2.3 documentation. [online] Available at: <https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html> [Accessed 25 April 2025].
Orange Data Mining Library, 2015. Regression (regression) — Orange Data Mining Library 3 documentation. [online] Available at: <https://orange3.readthedocs.io/projects/orange-data-mining-library/en/master/reference/regression.html> [Accessed 25 April 2025].
Ray, S., 2024. A Guide to Data Exploration, Steps Data Analysis - Analytics Vidhya. [online] Available at: <https://www.analyticsvidhya.com/blog/2016/01/guide-data-exploration/#h-what-is-data-exploration> [Accessed 25 April 2025].
Saxena, S., 2024. What are Categorical Data Encoding Methods | Binary Encoding. [online] Available at: <https://www.analyticsvidhya.com/blog/2020/08/types-of-categorical-data-encoding/> [Accessed 23 April 2025].
Singh, P., 2023. How do outliers impact linear regression evaluation? [online] Available at: <https://www.linkedin.com/advice/3/how-do-outliers-impact-linear-regression-dz0ff> [Accessed 24 April 2025].
Toxigon Infinite, 2025. Handling Multicollinearity in Regression Analysis: Practical Tips and Techniques - Toxigon. [online] Available at: <https://toxigon.com/handling-multicollinearity-in-regression-analysis?> [Accessed 25 April 2025].
under30ceo, 2024. Homoscedasticity - Under30CEO. [online] Available at: <https://www.under30ceo.com/terms/homoscedasticity/?utm_source=chatgpt.com> [Accessed 25 April 2025].

## License
The MIT License (MIT)

Copyright (c) 2025 Hannah Michaelson

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE




