## Analysis Plan 
The objective of this analysis is to create a linear regression model to accurately predictive insurance charges for a medical scheme member. The predictions will be based the observed selection of lifestyle factors for a client. Namly factors such as age, sex, BMI, number of children, smoking habits and geographical region. 
The analysis will include steps to understanding both the individual and combined implication of lifestyle factor on insurance charges for a medical scheme member. Will uncover the lifestyle factors that are both significant and irrelevant to the increase in insurance chargers and primarily build a model to generate insurance chargers. 
## Explanatory Data Analysis (EDA): 
Data exploration is a crucial step that needs to be taken before attempting to build a model. EDA helps an analysist to understand the given data. The EDA process identifies issues that may affect machine learning, ensures that data will be of suitable quality and identifies key patterns and trends that may need to be taken into consideration when building a predictive model (Ray, 2024). Various functions from the Pandas, Seaborn and Matplotlib libraries will be used to assist the EDA process. The EDA will include the following steps.

**Data Collection** : The first step to analysing the data is to retrieve it from the given csv file. The `Pandas.read_csv()` method will be used to read the data from the file and store is within a Pandas data frame object. The Pandas data frames can hold huge amount of data in excel like format, making it ideal to store insurance dataset and perform data manipulation (NumFOCUS Inc, 2024). 

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
**Model training**: The models used for this analysis will be the Scikit-learn Linear Regression model and Lasso Regression model. Linear Regression model has been selected as the native model to apply a linear regression algorithm. Lasso selected for its built-in feature selection capabilities (Orange Data Mining Library, 2015). 

**Evaluation**: To evaluate the performance of the models, the Scikit-learn library will be used. The library contains a metrics methods that provide the coefficient of determination (R^2) , mean absolute error and mean squared error for the results or prediction of a trained model (Deepanshi, 2023; Kim, 2023)  These metrics can be used to assess the performance and accuracy of a model.

**Visualisation of results**: Seaborn and Matplotlib will be used to display the result (Deepanshi, 2023; KIm, 2023). Plots such as  
- True vs Predicted Values, to compare predictions to actual datapoint and assess linearity   
- Residuals vs Predicted Values, to assess the Homoscedasticity of model of model results. 
- Normal Q-Q Residual Plot, to assess if residuals are normally distributed

## Exploratory Data Analysis
**Data Collection**: The data was first retrieved form the given csv file using `Pandas.read_csv()` method as displayed in figure 1. The data was stored in a Pandas data frame, theses data frames are able to store large amount of data in excel like format, making it ideal to store insurance dataset and perform data manipulation (NumFOCUS Inc, 2024). The `Pandas.head()` method was used to confirm data was successfully retrieved and assess rows and columns of the data. 
<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/08258570-b652-4127-912e-7a8e754e4c03" />
  <br>
  <em>Figure 1: Storing Data</em>
</div>

**Data Cleaning:**: `panda.isnull().sum()` was used to the dataset for missing values and return the sum of missing values for each column. There were no missing values found in the dataset, this can be seen in figure 2 below. 


<div align="center">
  <img width="600" height="300" alt="image" src="https://github.com/user-attachments/assets/12f82654-4767-46c7-857b-1a33086ca356" />
  <br>
  <em>Figure 1: Missing Values</em>
</div>
To find duplicates in data `panda.duplicated()` method was used to identify and create a list of all duplicates within the dataset. The list of duplicates returned one duplicate found in the in row 581 of the dataset as shown below in figure 3. This row was dropped from the dataset to avoid bias in the model. 
<div align="center">
	  <img width="700" height="120" alt="image" src="https://github.com/user-attachments/assets/fa881c86-b712-44fe-8ae4-1a020edec4bf" />
	<br>
  <em>Figure 1: Duplicates </em>
</div>

To identify the data types `panda.info()` method called. The method returned the below information about the dataset features. The method found that the 7 features within the dataset was a combination of ints, floats and objects, as shown in below in figure 4. 
- 2 of the columns, age and children are int datatypes and therefore suitable for a machine learning algorithms input. 
- 2 of the columns, charges and BMI are floats datatypes are too suitable as input for machine learning algorithms 
- 3 of the column’s datatypes are objects or string values which need to be converted to numerical values for to be used as input for the machine learning algorithms    
<div align="center">
	<img width="800" height="321" alt="image" src="https://github.com/user-attachments/assets/c964d985-8efe-4ad1-8aa7-1579925bc6b1" />
	<br>
  <em>Figure 4: Datatypes </em>
</div>

**Data Transformation**: To use machine learning algorithms, all data must be in numerical form, either float or int data types. The categorical features found, sex, smoker and region, are stored as the object data type therefore they were encoded. LabelEncoder from the Scikit-learn preprocessing module to encode the features by mapping each feature to a numbers value. These numbers were then you to distinguish the different categories as seen below in figure 5 (Novogroder, 2024). 
<div align="center">
	<img width="700" height="200" alt="image" src="https://github.com/user-attachments/assets/7e86cc33-1bdf-4c53-8169-0b1e0d2b1486" />
	<br>
  <em>Figure 5: Encoding </em>
</div>

**Univariate Analysis**  
This stage of analysis focused on analysing individual features in the dataset. For features with continuous values, the `pandas.describe()` method was used to print descriptive statistics. The summary uncovered the distribution, central tendency, and variability of each numeric feature in the dataset (Ray, 2024).  
Figure 6 displays the descriptive statistics:

<div align="center">
  <img width="800" height="429" alt="Descriptive Stats" src="https://github.com/user-attachments/assets/e8c99772-6b1b-4036-a4af-49463234d732" />
  <br />
  <em>Figure 6: Descriptive Statistics</em>
</div>

---

- **Age Feature**:  
  The mean (39.20) is very close to the median (39), indicating a symmetrical distribution. The standard deviation (14.0) suggests moderate variability. The 25th and 75th percentiles (27 and 64) confirm that 50% of members are aged between 27 and 64 years.

- **BMI**:  
  The mean (30.6) is close to the median (30.4), indicating symmetry. The standard deviation (6.09) indicates low variability. 50% of members have a BMI between 26.29 and 34.69. The max value (53.1) suggests outliers are present.

- **Children**:  
  The mean (1.09) is higher than the median (1.0), suggesting a positive skew likely caused by the max value of 5 children. The standard deviation (1.2) indicates moderate variability. 50% of members have between 0 and 2 children.

- **Charges**:  
  The mean ($13,270) is much higher than the median ($9,382), indicating a positively skewed distribution. The standard deviation ($12,110) shows high variability. 50% of members pay between $4,740 and $16,639. The max value ($63,770) suggests the presence of outliers.



Univariate analysis for categorical features was carried out using visualisation tools from seaborn and matplotlib. Seaborn count plots were used to understand the distribution sex, region, charges and smokers in the dataset. The below figure 7 shows the count plot for smokers and non-smokers members in the dataset. The counter plot indicates that the number of members who are non-smoke in the dataset is greater than the number of smokers in the dataset.
<div align="center">
	<img width="600" height="400" alt="image" src="https://github.com/user-attachments/assets/79508759-2d21-43ed-93fa-d9dea6c9bbb3" />
	<br>
  <em>Figure 7: Smokers Count Plot</em>
</div>


**Bivariate Analysis**: This stage of analysis focuses on uncovering the association and distribution between two variables. To understand the relationship between two variables visualisation tools such as seaborn and matplotlib will be used to create correlation heatmaps, pair plots, distribution plots and scatterplots. 
<div align="center">
	<img width="600" height="500"  alt="image" src="https://github.com/user-attachments/assets/d09228d4-cc9a-4b25-87ca-fb1231cc0b9d"  />
	<br>
  <em>Figure 8: Correlation Heatmap</em>
</div>
Figure 8 above display a correlation heat map. This was used to shows the correlation between all variables in a dataset. The Pandas method `dataframe.corr()` is used to calculate the correlation coefficient for each variable in the dataset (NumFOCUS Inc, 2025). The method returns a float between 1 and -1, indicating the correlation between each pair of variables (NumFOCUS Inc, 2025). 1 indicates perfect correlation, 0 indicates no correlation and -1 perfect negative correlation. This visual the matrix of correlations using the seaborn heatmap (Jain, 2024).  

---

**Notable insights were**:
- The charges variable is positively correlated with age, BMI and children variables. Each of these variables have a positive linear relationship with the target variable. Therefore, for an increase in any one of the variables charges will increase too.
- The correlation heat map confirms minimal correlation between independent variables, indicating that there is no multicollinearity that will contradict the multicollinearity assumption of linear regression.
- 0.79 highest correlation between a pair of variables. It defines a strong positive linear relationship between smoker and charges variables.
With the strongest correlation between features smoker and charges, I was led to more thoroughly investigate the correlation between smoker and charges and further, the influence smoker and charges may have on other features in the dataset.

The below plot provides in figure 9 shows an in-depth view of the distribution of insurance charges for both smokers and non-smokers. 

<div align="center">
	<img width="600" height="530" alt="image" src="https://github.com/user-attachments/assets/f48b977f-a21f-45d7-927b-deb1c6633444" />
	<br>
  <em>Figure 9: Charges Distribution by Smoker Status</em>
</div>

---

*Smokers (Peach)*:
- The plot show there are fewer members with smoking status within the dataset. The bars representing smoker, do not exceed 25 on the y-axis showing a significant difference in the count for smoking vs non-smoking members. 
- The graph shows a bimodal distribution, with peaks approximately falling between $10,000 to $30,000 and $30,000 to $60,000. This distribution suggests that other features may have a strong contribution to a decrease in the charged insurance fee despite smoking status (Frost, 2022).\ 
*Smokers (Blue)*:
- The plot show that most members in dataset are of non-smoking status. The bars representing non-smoker, ranges across all values on the y-axis showing a significant difference in the count for smoking vs non-smoking members. 
- the graph shows a right skewed distribution, with non-smoking members incurring charges of $1,121 (min value) to approximately $15 000. The upper tail of the non-smoking curve does fall within higher charge rate, indicating that there are features may have a strong contribution to an increase in the charged insurance fee despite smoking status.  

The plot of Charges Distribution by Smoker Status, confirmed the strength of the correlation between smoker and the rate of charge, uncovered correlation heatmap (figure 10). Thus confirming the smoking status of a member is a significant in the charges incurred by a member. 


**Multivariate Analysis**: multivariate analysis extends the bivariate analysis by further investigating the significant finding. In this can the correlation between smoker and the charged rate of members. The analysis stage was aimed at understanding the relationship between multiple variables. Tools from seaborn and matplotlib will be used to visualise the relations between variables. Plots such as distribution plots and scatterplots will carry out this stage (Deepanshi, 2023; Kim, 2023). 
The bar plot in figure 10, provides an insight into the distribution of insurance charges for both smokers according to sex.
<div align="center">
	<img width="700" height="400" alt="image" src="https://github.com/user-attachments/assets/b7fa3e40-9698-4a64-a906-e56d6b6350ee" />
	<br>
  <em>Figure 10: Average Charges for Smokers and Non-Smokers</em>
</div>

---

- Non-smoker: For both male and female members, lower insurance charges are incurred by non-smokers
- Smoker: For both male and female members, higher insurance charges are incurred by smokers.
High and low insurance rates are charges for both females and males, therefore the sex of a member when concerning charges are indifferent.This indicated that the sex of a member had no significant influence on the rate of charges on a member. 


The below scatter plots in figure 11, provides an insight into the distribution of insurance charges across member’s age. Separated by smoking status. The below plots were intended to further to further investigate the influence of chargers and smoking status in the dataset.
<div align="center">
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/4feb7708-ceca-4b82-81ef-7fd85c581bf1" />
	<br>
  <em>Figure 11: Charges vs Age (Smokers and Non-Smokers)</em>
</div>

---

Scatter plot 1 (Charge vs age by Non-Smoking Members): 
- By plotting a regression line, a positive correlation between age and charges can be observed for non-smoking members. This indicates the insurance charge for a member increase with their age. Therefore, younger members receive lower rates than older members.
Scatter plot 2 (Charge vs age by Smoking Members):
- Similarly, a plotted regression line, plot 2 shows a positive correlation between age and charges for smoking members. The plot indicates the insurance charge for a member increase with their age. However, the rates for smoking members are much larger as the line is plotted higher up on the y-axis compared to plot 1. Therefore, members that smoke incur higher rates both young and old members.
- Another insight from the second plot is gathered from the separation is clusters. Although the data point follow a linear trend, they are clustered separately in the lower and upper regions of the y-axis. This separation implies that there is another lifestyle factors or factors that cause this variation and thereby contribute to higher charge rates.

These finding indicates that there is a positive correlation between the age and charge features in the dataset, suggesting that age too has significance in the rate of charge for a member. When combined with the smoking feature, the rate of charges becomes even larger. 

The below scatter plots in figure 12, provides an insight into the distribution of insurance charges across member BMI. Separated by smoking status. The below plots were intended to further to further investigate the influence of chargers and smoking status in the dataset.
<div align="center">
	<img width="700"  height="400" alt="image" src="https://github.com/user-attachments/assets/1bc119ef-af57-40b0-bfe4-b26ca3aa2602" />
	<br>
  <em>Figure 12: Charge vs BMI (Non-smoking and Smoking)</em>
</div>

---

Scatter plot 1 (Charge vs BMI of non-smoking members): 
- By plotting a regression line, a moderate positive correlation can be observed between BMI and charges for non-smoking members. This indicates the insurance charge of a member increases with a higher BMI. Therefore, overweight and obese members receive higher charges.
Scatter plot 2 (Charge vs BMI of smoking members):
- Similarly, a plotted regression line, plot 2 shows a strong positive correlation between BMI and charges for smoking members. The plot indicates the insurance charges for a member increase with higher BMI. However, the rates for smoking members are more extreme. Therefore, the combination of smoking and a high BMI result in high insurance rates.  
With highlighting the smoking status of members, a linear relationship become clear. There is a positive relationship between BMI and charges, however the smoking status determines stronger degree of influence between the variables. 

These finding suggesting that BMI feature has significance influence on the rate of charge for a member. When combined with the smoking feature, the rate of charges becomes even larger. 
The finding from the EDA conclude that features BMI, smoker and age have and influence on the dependent variable, charges. 

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




