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
	  <img width="700" height="120" alt="image" src="https://github.com/user-attachments/assets/5b0d8d15-0bd1-46db-8dc7-5ee9b8c2a06c" />
	<br>
  <em>Figure 3: Duplicates </em>
</div>

To identify the data types `panda.info()`. The method returned the below information about the dataset features. The method found 9 features within the dataset, which were a combination of ints and floats, as shown in below in figure 4. 
- 2 of the columns, age and children are int datatypes and therefore suitable for a machine learning algorithms input. 
- 2 of the columns, charges and BMI are floats datatypes are too suitable as input for machine learning algorithms.
- 3 of the column’s datatypes are objects or string values which need to be converted to numerical values for to be used as input for the machine learning algorithms.
<div align="center">
	<img width="600" height="321" alt="image" src="https://github.com/user-attachments/assets/7af62f02-222e-42a1-9fa1-617b26672e2e" />
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

- **Physical Activity: **:  
The mean (4.89) and median (4.83) are in close proximity. This suggests a near symmetrical distribution of data. Additional, on average individuals spend 4 hours physical active. The std (2.87) is moderate relative the features mean, indicating moderate variability in the spread of data. Furthermore, the min (0.00) and max (10.00) indicate activity ranges from inactive to highly active.  This variability is supported by the 25th and 75th percentiles, which indicates that 50% of individuals are physical active for 2.43 to 7.40 hours per day. Thus, the dataset contains wide range of physical activity, with a moderate portion of individuals sedentary to highly active.

- **Charges**:  
The mean (2.42) and median (2.38) are in close vicinity, on average individuals consume approx.  2.5 units of alcohol a week. This suggests a near symmetrical distribution of data. The std (1.41) is moderate relative the mean, indicating moderate variability within the spread of data. The min value is 0.00 and max 5, signalling alcohol behaviour ranges from sober to indulgent.  The 25th and 75th percentiles supporters the variability, which indicates 50% of individuals drink 1.21 to 3.59 unit of alcohol a week. Thus, majority of individuals in the dataset have a moderate alcohol Intake.


Univariate analysis for categorical features was carried out using visualisation tools from seaborn and matplotlib. Seaborn count plots were used to understand the distribution genetic risk, cancer history, smoking and gender in the dataset. The below figure 7 shows the count plot for genetic risk levels in the dataset. The count plot illustrates the frequency distribution of genetic cancer risk within the 1500 individual’s sample. The plot indicates, majority of individuals fall into the low-risk category (approx. 900). The moderate amount (approx. 450) of individuals are categorized as medium risk and s small subset (approx. 150) are categorized as high risk.

<div align="center">
	<img width="844" height="579" alt="image" src="https://github.com/user-attachments/assets/15c4d5c4-793e-48c0-9964-43f06529a13c" />
	<br>
  <em>Figure 7: Genetic Risk Distribution</em>
</div>

**Multivariate Analysis**: This stage of analysis focuses on uncovering the association and distribution between variables in the dataset. To understand the relationship between two variables visualisation tools such as seaborn and matplotlib will be used to create correlation heatmaps, pair plots and distribution plots. 
<div align="center">
	<img width="749" height="587" alt="image" src="https://github.com/user-attachments/assets/ecc2d75b-98ed-4589-ba0f-bcd42f1dfe6f" />
	<br>
  <em>Figure 8: Correlation Heatmap</em>
</div>
Figure 8 above display a correlation heat map. This was used to shows the correlation between all variables in a dataset. The Pandas method "dataframe.corr()" is used to calculate the correlation coefficient for each variable in the dataset ((NumFOCUS Inc, 2025)). The method returns a float between 1 and -1, indicating the correlation between each pair of variables (NumFOCUS Inc, 2025). 1 indicates perfect correlation, 0 indicates no correlation and -1 perfect negative correlation. This visual the matrix of correlations using the seaborn heatmap (Jain, 2024).   

---

**Notable insights were**:
- Diagnosis has a weak positive correlation with Cancer History, Alcohol Intake, Smoking, Gender, Age, BMI and Genetic Risk features. If one of these categorical features are true (yes=1), the likelihood of a positive cancer diagnosis increases.
- Diagnosis has a weak negative correlation with Physical Activity. For member that are physical activity in the day, there is a slightly lower chance of a positive cancer diagnosis
- 0.39 is the highest correlation between Diagnosis and Cancer History. However, it is a moderate correlation.
---
The below plot provides in figure 9  provides an insight into the distribution of smoking and a non-smoking individual, according their cancer diagnosis.
<div align="center">
	<img width="879" height="636" alt="image" src="https://github.com/user-attachments/assets/1a25f59c-1548-4672-88d9-78343faf2208" />
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
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/1d794596-6ac1-4854-82d8-b4832e5a7b7a" />	<br>
  <em>Figure 11: Genetic Risk by Cancer Diagnosis </em>
</div>

---

Low Risk:
- The bar for low genetic risk shows a greater proportion of individuals have a negative cancer diagnosis, while a smaller subset a positive cancer diagnosis.
- For the low-risk population, 282 individuals have cancer (blue), and 613 individuals do not have cancer (green).
- This indicates that individuals with a low genetic risk are more likely to have a negative cancer diagnosis.  
Medium Risk:
- The bar for medium genetic risk shows a greater proportion of individuals have a negative cancer diagnosis, while a smaller subset a positive cancer diagnosis.
- For the medium risk population, 142 individuals have cancer (blue), and 305 individuals do not have cancer (green).
- This indicates that individuals with a medium genetic risk are more likely to have a negative cancer diagnosis.   
High Risk:
- The bar for high genetic risk shows a greater proportion of individuals have a positive cancer diagnosis, while a smaller subset a negative cancer diagnosis.
- For the high risk population of 158 , 133 individuals have cancer (blue), and 25 individuals do not have cancer (green).
- There is a sizeable difference (108) between individuals who have cancer and do not. This suggests that high genetic risk has a potential association with positive cancer diagnosis.

Observations suggests that low and medium genetic risk may not be strong predictors for cancer. However, high genetic risk shows a strong association with cancer and may be a strong predictor.

The below scatter plots in figure 12, provides an insight into the distribution of insurance charges across member BMI. Separated by smoking status. The below plots were intended to further to further investigate the influence of chargers and smoking status in the dataset.
<div align="center">
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/1bc119ef-af57-40b0-bfe4-b26ca3aa2602" />
	<br>
  <em>Figure 12: Charge vs BMI (Non-smoking and Smoking)</em>
</div>

---

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

Data scaling: The Scikit `StandardScaler()` methods was used to scale features within the dataset in preparation for model training. This was to ensure larger values such as BMI and Age did not have disproportionate significance in the model which would result in inaccurate predictions.
<div align="center">
	<img width="480" height="175" alt="image" src="https://github.com/user-attachments/assets/631dba8f-0415-4110-ae91-c1d78e6d44ba" />
	<br>
  <em>Figure 13: Scaled Data</em>
</div>
  
Feature Selection: Feature selection will be performed by both backward elimination and `sklearn.selectKBest()` to compare results. SelectKBest was used with the `f_regression` parameter and a max output of 3. The method selected the best based upon the F-value, which indicates the variance between each feature and the dependent variable. The features are scored according to the significance of their variance (Kavya D, 2023). Backward elimination selects the best features based upon the p-value which indicates probability of a features significance. The best features for selections by removing features that are not significant (GeeksforGeeks, 2025b).  Both methods nominated the same features, smoker, age and BMI. Which are consistent with EDA finding, the results are shown below in figure 14.
<div align="center">
	<img width="600" height="220" alt="image" src="https://github.com/user-attachments/assets/d632494e-3355-4a9a-b4cc-0ad9d37be4fb" />
	<br>
  <em>Figure 14: Scaled Data</em>
</div> 

Splitting data: The data used for model training was split into an 80:20 ratio, 80 percent of the data being used as training data and the remaining 20 percent for test data. This is an important step so that after training the model can be tested on unseen data to assess its performance (Gillis, 2024). The selected features sex, BMI and age were used to train the model. 


## Build and Evaluate Model
The models used for this analysis will be the Scikit-learn Linear Regression model and Lasso Regression model. Linear Regression model has been selected as the native model to apply a linear regression algorithm. Lasso selected for its built-in feature selection capabilities, which is a result of the regularization (Orange Data Mining Library, 2015). The lasso model adds a penalty to the cost function to reduce the coefficients for each variable. The degree of the regulation is controlled by the alpha parameter. Due to regularization convergence in the gradient of descent happens as a slower rate, for this reason the tolerance threshold for convergence must either be decreased or the max iterations increased to for the model to perform well (Orange Data Mining Library, 2015).For these reasons the lasso model had been trained an alpha of 10 and a tolerance of 0,0001, as seen below in figure 15. 
<div align="center">
	<img width="600" height="200" alt="image" src="https://github.com/user-attachments/assets/4927b5d0-7ebb-456a-a224-2a9f988e5111" />
	<br>
  <em>Figure 15: Lasso Model</em>
</div> 

To evaluate the performance of the models, the Scikit-learn library will be used. The library contains a metrics method that provide the coefficient of determination (R^2) , mean absolute error and mean squared error for the results or prediction of a trained model (Deepanshi, 2023; Kim, 2023). The result indicate the Lasso Regression model has a better performance than the Linear regression as seen below on figure 16 and 17. 
<div align="center" style="display: flex; justify-content: center; gap: 20px;">
  <div>
    <img width="450" height="124" alt="Lasso Results" src="https://github.com/user-attachments/assets/0c31dcee-c179-44ec-95d8-cfb490909bef" />
    <p><em>Figure 16: Lasso Results</em></p>
  </div>
  <div>
    <img width="450" height="113" alt="Linear Results" src="https://github.com/user-attachments/assets/7b30d2bf-b710-4f4b-a6c8-9d99c67fea70" />
    <p><em>Figure 17: Linear Results</em></p>
  </div>
</div>

The coefficient of determination (R^2)  for the Lasso model of 0.78 is close 1 well the Linear model falls slightly lower. This implies, that 78% percent variance is explained by the model.  The mean absolute error of the lasso model indicates that results deviated by $4192.78 

--- 

### True vs Predicted Values, to compare predictions to actual datapoint and assess linearity
<div align="center">
<img width="700" height="509" alt="image" src="https://github.com/user-attachments/assets/17150aa2-22aa-4316-9959-99f1ede6f0e0" />
	<br>
</div> 

---

### Residuals vs Predicted Values, to assess the Homoscedasticity of model of model result
<div align="center">
	<img width="700" height="493" alt="image" src="https://github.com/user-attachments/assets/df95edec-5b2b-41fd-96fa-3c0486a6b2bd" />
	<br>
</div> 

---

### Normal Q-Q Residual Plot, to assess if residuals are normally distributed
<div align="center">
	<img width="700" height="503" alt="image" src="https://github.com/user-attachments/assets/115f80b0-1e66-4a88-afce-dcfa1d0b472d" />
	<br>
</div> 


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




