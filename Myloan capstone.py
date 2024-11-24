#!/usr/bin/env python
# coding: utf-8

# ### **LOAN DEAFULT EDA**
# 
# The core of a banking system’s profitability hinges on its ability to efficiently manage **lending and borrowing operations**. Banks act as intermediaries, accepting deposits at a certain interest rate and lending them out at a higher rate. The **profit margin** is derived from the **spread** between the interest charged on loans and the interest paid on deposits.
#   However, **loan defaults** can significantly impact the bank's financial health and operational efficiency. Therefore, it is crucial to analyze patterns of loan defaults and recommend actionable strategies to minimize risks while optimizing profitability.
# 
# ### **Objective**
# To perform **Exploratory Data Analysis (EDA)** on the given dataset of a leading bank, with a focus on identifying patterns and key factors contributing to loan defaults. The insights gained will help the bank’s management:
# 1. **Mitigate default risks** by identifying high-risk segments.
# 2. **Enhance profitability** by targeting creditworthy customers.
# 3. **Optimize lending strategies** through data-driven decision-making.
# 
# ### **EDA Approach**
# 
# #### **1. Data Understanding and Preprocessing**
# - **Columns Overview**: Analyze the structure of the dataset, including numerical and categorical variables, missing values, and data types.
# - **Key Variables**:
#   - **Loan Details**: Loan type, loan amount, interest rate, property value,LTV, upfront charges, and loan purpose.
#   - **Demographics**: Age, gender, region,income and occupancy type.
#   - **Credit History**: Credit score, co-applicant credit type.
#   - **Loan Performance**: Loan status (default or normal).
# 
# ---
# 
# #### **2. Analyze Loan Default Patterns**
# - **Categorical Variable Analysis**:
#   - **Default Rates by Loan Type**: Determine which loan types  are riskier.
#   - **Default Rates by Loan Purpose**: Determine which loan purpose  are riskier.
#   - **Default Rates by Region**: Identify regional trends and disparities.
#   - **Default Rates by Occupancy Type**: Analyze if occupancy status (e.g., owner-occupied vs. rented) affects loan performance.
# 
# - **Numerical Variable Analysis**:
#   - **Credit Scores of Defaulters**: Investigate whether lower credit scores are associated with higher default rates.
#   - **Loan-to-Value Ratio (LTV)**: Examine if high LTV ratios correlate with defaults.
#   - **Interest Rates**: Assess whether higher interest rates increase the likelihood of defaults.
# 
# ---
# 
# #### **3. Statistical Analysis**
# - **Correlation Analysis**:
#   - Examine the relationships between variables such as loan amount, interest rate, upfront charges, and default status.
#   - Use a **correlation matrix** to uncover linear dependencies.
# 
# - **Hypothesis Testing**:
#   - Conduct tests to validate the significance of relationships. For example:
#    
# 
# 
# 

# In[112]:


#Libraries Used
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
import warnings
warnings.filterwarnings('ignore')


# In[235]:


#Reading the data and copying to a variable named 'df'
df=pd.read_csv('loan.csv')


# In[236]:


#Data set outline
df.head(5)


# #Pointers
# 1)business_or_commercial--There are 2 categories present in the column.They are 'b/c' and 'nob/c'
# b/c--indicates loan is provided for business or commercial purpose
# nob/c-indicates loan is provided for personal purpose.
# 
# 2)Status-It indicates the loan status.
# '0'-indicates loan is not defaulted
# '1'-indicates loan is defaulted

# In[4]:


#The data set has 148670 rows and 20 columns
df.shape


# In[115]:


df.info()


# In[ ]:


#Here loan_limiit,Gener,Loan_type,Loan_purpose,Credit_type,Age,region,occupancy_type are object type.

#Rest all are integer and float type


# In[237]:


#Summary Of Statstics
df.describe().T


# In[238]:


#Categorical columns description
df.describe(exclude=np.number).T


# In[ ]:


#Year-If we look at the 'year' column the time frame is set for 2019

#Loan Amount-The maximum 'loan amount' available here is 3576500 and the average loan amount is 331117 with wide range of values

#Rate_of_interest-The rate of interest hovers around 4% and the maximum provided interest rate is 8%.Again the interest rate mostly falls between 3%-4%.

#Upfront_charges-It is the amount that a customer pays before loan is disbursed.The highest paid amount being 60000 with 
#average amount of 3251 is seen as the general trend.It is intresting feature to check whether it has any correlation with better repayment chances.

#Property Value-Banks provide loan based on the backing against a property.Normally 75-80% of the property value is provided 
#as loan amount.We can check if there is any lower or upper threshold for property value.

#Income-Income of the applicant is also a factor taken into consideration while providing loan.The EMI fixation for repayment 
#depends on the income as normal rule is that it EMI should not excced 40% of income.Mean income falls around 6496.

#Credit_score-It is a 3 digit  number which predicts how likely you are to repay the loan.Basic FICO scores varies from 300 to 850 while industry specific varies from 250 to 900.mean score observed here is 699 with the highest one being 900.

#Status-It is the loan status where 0 indicates loan repaid.1 indicates default.

#LTV-Loan To Value is the ratio of loan amount to property value.The average LTV comes around 72.




# OBSERVATIONS:
# 

# In[ ]:


#DISPLAYING UNIQUE VALUES


# In[118]:


#DISPLAYING UNIQUE VALUES
cat_col_unq=df.select_dtypes(exclude=np.number)
for i in cat_col_unq.columns:
    print(f'Unique Values in {i} are:')
    print(df[i].value_counts(normalize=True))
    print('*'*40)


# In[239]:


df['Gender'].value_counts()


# In[240]:


#Replacing the missing data on  'Gender' column with the mode value.
mode_gender=df['Gender'].mode()[0]
df['Gender']=df['Gender'].replace('Sex Not Available',mode_gender)


# In[241]:


#Function to display the details of the column.
def column_details(df,column):
    print('Details of Column are as follows:')
    print('\nDataType:',df[column].dtype )
    countnull=df[column].isna().sum()
    if countnull==0:
        print('column',column,'has no null values')
    else:
        print('number of non null values are:',countnull)
    print('Unique values are',df[column].nunique())
    print('Distribution of columns is')
    print('\n',df[column].value_counts())


# In[122]:


column_details(df,'Gender')


# In[123]:


column_details(df,'income')


# In[131]:


df['income'].mode()
df['Upfront_charges'].mode()
#It is noted that mode of column income and Upfornt charges is 0.This may have an impact on our EDA.
#To tackle it we replace the null values with median
med_income=df['income'].median()
med_upfront=df['Upfront_charges'].median()
df['income']=df['income'].replace(0,med_income)
df['Upfront_charges']=df['Upfront_charges'].replace(0,med_upfront)


# In[ ]:


#We are dropping ID column and year because ID column does not provide us any valuable insight 
#Year column has constant value of 2019.


# In[132]:


null_col=['loan_limit','loan_purpose','rate_of_interest','Upfront_charges','property_value','income','age','LTV']


# In[242]:


#Analysing the null values in the data set
df.isna().sum()


# In[134]:


#Percentage of Missing Values
(df.isna().sum()/len(df))*100


# In[ ]:


#Insights:
#Columns rate_of_interst and Upfront_charges have highest percentage of null_values followed by property_value and LTV.


# In[137]:


#Function to fill the null values
def null_filler(df,column):
    
    cnull=df[column].isna().sum()
    if cnull!=0:
        mode_val=df[column].mode()[0]
        df[column]=df[column].fillna(mode_val)
        


# In[138]:


#Clearing the null values
for col in null_col:
    null_filler(df,col)


# In[139]:


#Rechecking the null values
df.isna().sum()


# In[ ]:





# In[141]:


#Selecting numerical columns
num_col=df.select_dtypes(include=np.number)


# In[142]:


#Dropping  ID,Year,Status from numerical column
num_col.drop(columns=['ID','Status','year'],inplace=True)


# In[143]:


num_col.reset_index()


# In[244]:


#Checking for outliers
for col in enumerate(num_col):
    
    
    sns.boxplot(x=col[1],data=num_col)
    plt.show()
   


# In[ ]:


#As it is clear we have outliers in almost all the numerical columns except 'credit score'.
#So it is necessary to remove the outliers berfore EDA.
#The method we have used here is IQR(Inter-Quartile range).
#This effectively clips the data below(25 Quartile) and above(75 Quartile).


# In[145]:


#Treating Outliers
Q1=num_col.quantile(0.25)
Q3=num_col.quantile(0.75)
IQR=Q3-Q1
print(IQR)


# In[24]:


#Filtering the data lying outside 25 and 75 quartiles
mask=~((num_col<(Q1-1.5*IQR))|(num_col>(Q3+1.5*IQR))).any(axis=1)


# In[146]:


df_new=df[mask]


# In[147]:


#Dataset after treating outliers
df_new


# In[148]:


df_new.dtypes


# In[149]:


df_new['Status'].astype('category')


# In[150]:


df_sample=df_new.sample(n=10000,random_state=42)


# In[151]:


#Univariate Analysis on Categorical Column
disp_col_cat=['Gender','loan_limit','loan_type','loan_purpose','business_or_commercial','occupancy_type','Region','age']
for i in disp_col_cat:
    sns.countplot(data=df_new,x=i)
    plt.show()


# In[ ]:


Insights:
#Gender--If we look at the gender wise distribution of loans male customers are the highest.The loans were issues jointly to both husband and wife as well.

#Loan Type-The highest number of processed loans were for type 1 followed by type 2 and type 3.

#Loan Purupose-We have 4 categories for 'loan_purpose'.Most of the loans were issued for purpose p4 and p3.The lowest were
#alloted to 'p2'.

#Business or Commercial--Most of the loans were of nature non commercial purpose.

#Occupancy type--Most of the establishments are used for self occupancy.The percentage of establishments for rent out or mixed puprose is very small.

#Region.-Of the 4 regions Northern region has the highest loan takers while North-East has the lowest.

#Age-The age group is spread between 25-74 with most number of applicants between 45-44


# In[245]:


#Univariate Analysis on Numerical Columns
disp_num_col=['loan_amount','rate_of_interest','Upfront_charges','property_value','income','Credit_Score','LTV']
for i in disp_num_col:
    sns.histplot(data=df_sample,x=i,kde=True)
    plt.show()


# #Loan Amount:Mostly  the amount falls between 150000 and 400000.Loan amount follows almost a gaussian distribution.
# 
# #Rate Of Interest-The rate of interest 4 is the most common applied interest on loans.
# 
# #Upfront Charges--The customer base availing loan by paying Upfront charges is comparitively low.
# 
# #Property Value-It follows a gaussian distribution with the value lying in the range between 300000 and 500000
# 
# #Income--It also follows a gaussian distribution with values mostly lying in the range bewteen 3000 and 7000.
# 
# #Credit Score--It is evevly spread between 550 and 900.
# 
# #LTV-Loan to Value.From the plot we assume a value of 80 is taken by the instituion mostly to grant the loan.

# In[246]:


#Converting object type to category

cat_col=['Gender','loan_type','loan_purpose','business_or_commercial','occupancy_type','Region','age','credit_type','co-applicant_credit_type','Status']
for col in cat_col:
    df_new[col]=df_new[col].astype('category')


# In[247]:


df_new.dtypes


# In[248]:


#Percentage distribution of
perc=(df['Status'].value_counts()/len(df['Status']))*100
print(f'The percentage of deafulters in the dataset is {perc} %')
df['Status'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Loan status Distribution')
plt.show()


# In[249]:


perc1=(df['loan_type'].value_counts())/(len(df['loan_type']))*100
print(f'The percentage distributon of loan type is{perc1}')
df['loan_type'].value_counts().plot(kind='pie',autopct='%1.1f%%')
plt.title('Loan Type Distribution')
plt.show()


# In[155]:


from scipy.stats import chi2_contingency
from scipy.stats import ttest_ind
 


# In[252]:


#Function to plot the Stacked Bar Chart
def disp_plot(col1,col2):
    cross_tab = pd.crosstab(df_new[col1], df_new[col2])

# Plot the stacked bar chart
    cross_tab.plot(kind='bar', stacked=True, color=['green', 'red'], figsize=(8, 5))
    plt.title(f' {col1} vs {col2}')
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.legend(title=col2)
    plt.show()


# # Gender & Loan Status

# In[253]:


#Null Hypothesis--No Statstical relation between gender and status of loan approval
#Alternate Hypothesis--There is a statstical significance between gender and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['Gender'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
    
disp_plot('Gender','Status')    


# In[251]:


df_new.groupby(['Status','Gender'])['loan_amount'].describe()


# In[159]:


sns.boxplot(data=df_new,x='Status',y='loan_amount',hue='Gender')
plt.title('Loan Amount Distribution by Status and Gender')
plt.xlabel('Loan Status')
plt.ylabel('Loan Amount')
plt.show()


# In[ ]:


#Inference--
#Gender do have an impact on default status .
#It is also noteworthy that loans with coobligants have lower default rate.
#Male customer contributes maximum application
#Loan amount is higher if there is a coobligant
#Loan amount for females is the lowest among 3 categories


# # Region & Loan Status

# In[254]:


#Null Hypothesis--No relation between Region and status of loan approval
#Alternate Hypothesis--There is a relation between Region and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['Region'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
disp_plot('Region','Status')    


# In[161]:


pd.crosstab(df_new['Status'],df_new['Region'])


# In[162]:


df_new.groupby('Status')['Region'].describe()


# In[255]:


geographic_data = df_new.groupby('Region').agg(
    total_loans=('loan_amount', 'count'),
    total_loan_amount=('loan_amount', 'sum'),
    default_rate=('Status', lambda x: (x == 1).mean())
).reset_index()

sns.barplot(data=geographic_data,x='Region',y='default_rate',palette='coolwarm')
plt.title('Default rates distribution by Region')
plt.show()


# In[234]:


df_new['Status']=df_new['Status'].astype('int')
default_rates_region=df_new.groupby(['Region','Status'])['loan_amount'].mean()
default_rates_region


# In[ ]:


#Insights--
#The percentage wise distribution of defaulters is higer for NorthEast side followed by central.
#It is the lowest for Northern Region
#Highest amount of loan allocation is in Northern Side.
#Lowest loan allocation is on the North-East Side
#Highest number of defaulters are on South side.


# # Loan Limit & Loan Status

# In[294]:


#Null Hypothesis--No relation between loan limit and status of loan approval
#Alternate Hypothesis--There is a relation between loan limit and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['loan_limit'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
disp_plot('loan_limit','Status')   


# In[ ]:


#Inference
#Most loans provided are loans with fixed limit.


# In[295]:


#Null Hypothesis--No relation between loan limit and status of loan approval
#Alternate Hypothesis--There is a relation between loan limit and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['business_or_commercial'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
disp_plot('business_or_commercial','Status') 


# # Occupancy Type & Loan Status

# In[293]:


#Null Hypothesis--No relation between Occupancy type and status of loan approval
#Alternate Hypothesis--There is a relation between Occupancy type and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['occupancy_type'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
disp_plot('occupancy_type','Status')    


# In[258]:


df_new.groupby(['Status','occupancy_type'])['loan_amount'].describe()


# In[259]:


pd.crosstab(df_new['Status'],df_new['occupancy_type'])


# In[260]:


sns.boxplot(data=df_new, x='Status', y='loan_amount', hue='occupancy_type')
plt.title('Loan Amount Distribution by Status and Occupancy')
plt.ylabel('Loan Amount')
plt.xlabel('Loan Status')
plt.legend(loc='upper right')
plt.show()


# In[ ]:


#Insight--Most loan number are sanctioned for properties which are self occupied.
#Percentage default rate is lowest for self occupied and highest for mixed occupancy.
#Lowest number of loan allocation is for 'sr' type.
#It is noted that the default percntage for leased out property is higher than that of self occupied property.
#This shows that the property which are occupied by applicants are less likely to get defaulted as it shows an
#emotional connect to the property


# In[261]:


#Null Hypothesis--No relation between age and status of loan approval
#Alternate Hypothesis--There is a relation between gender and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['age'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('failed to reject null hypothesis')
disp_plot('age','Status')    


# In[191]:


df_new.groupby('Status')['age'].describe()


# In[262]:


df_new.groupby(['age','Status'])['loan_amount'].describe()


# In[192]:


pd.crosstab(df_new['Status'],df_new['age'])


# In[230]:


sns.boxplot(data=df_new,x='age',y='Credit_Score',hue='Status')
plt.legend(loc='lower left')
plt.show()


# In[ ]:


#Insight---Most of the applicants falls between age group between 35-44.
#Most number of defaulters are between age 45-54.
#Highest percentage  of defaulters are for age>74.
#Least applicants are for the age group <25 and age group >74.
#'<25' age group has the highest credit score in non default cases and lowest in default cases.
#This age group can be more tapped into for laon applications.
#The upper threshold for loan amount of age group <25 could be raised further


# # Loan type & Loan Status

# In[280]:


#Null Hypothesis--No relation between laon type and status of loan approval
#Alternate Hypothesis--There is a statsical relation between loan type and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['loan_type'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('failed to reject null hypothesis')
disp_plot('loan_type','Status')    


# In[220]:


sns.barplot(data=df_new,x='loan_type',y='loan_amount',hue='Status',estimator='mean',palette=['green','red'])
plt.title('Loan Type Distribution')
plt.show()


# In[194]:


df_new.groupby('Status')['loan_type'].describe()


# In[225]:


loan_type_stat=df_new.groupby(['loan_type','Status'])['loan_amount'].describe()
loan_type_stat


# In[195]:


pd.crosstab(df_new['Status'],df_new['loan_type'])


# In[ ]:


#Insight-
#The approved loans are mostly of type 1 followed by type 2 and 3.It is probable that type 1 may be of
#non commercial nature.
#Highest percentage of deafaulters are for type 2
#Even though loan count is least for type 3 loan disbursed in this category comprises large amount
#It is probable that type 3 may be of commercial type.


# # Loan Purpose & Loan Status

# In[281]:


#Null Hypothesis--No relation between loan purpose and status of loan approval
#Alternate Hypothesis--There is a relation between loan purpose and loan approval status
stat,p_val,dof,exp_frq=chi2_contingency(pd.crosstab(df_new['loan_purpose'],df_new['Status']))
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('failed to reject null hypothesis')
disp_plot('loan_purpose','Status')    


# In[218]:


sns.barplot(data=df_new,x='loan_purpose',y='loan_amount',hue='Status',estimator='mean',palette=['green','red'])
plt.title('Loan Purpose Distibution')
plt.show()


# In[197]:


df_new.groupby('Status')['loan_purpose'].describe()


# In[223]:


loan_purpose_stat=df_new.groupby(['loan_type','Status'])['loan_purpose'].describe()
loan_purpose_stat


# In[224]:


pd.crosstab(df_new['Status'],df_new['loan_purpose'])


# In[ ]:


#Insights
#Loans are mostly applied for purpose 'p3' foloowed by 'p4','p1' and 'p2'.
#The percentage default rate is highest for p2 and least for p1.
#Maximum allocation of loan amount is for  p4
#Lowest allocation of loan amount is for p2


# In[59]:


num_col


# In[183]:


corr_mat=num_col.corr()
plt.figure(figsize=(6,7))
sns.heatmap(corr_mat,cmap='coolwarm',annot=True)


# # Loan Amount & Status

# In[226]:


#Null Hypothesis--There is no statsical correlation between loan amount and status
#Alternate Hypothesis--There is a statsical relation between loan amount and loan status
from scipy.stats import ttest_ind
stat,p_val=ttest_ind(df_new['Status'],df_new['loan_amount'])
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')


# In[227]:


summary_stats = df.groupby('Status')['loan_amount'].describe()
print(summary_stats)
mean_val=df_new.groupby('Status')['loan_amount'].mean()
mean_val.plot(kind='bar')
plt.ylabel('Loan Amount')
plt.xlabel('Loan Status')
plt.title('Loan Amount VS Status')


# In[ ]:


#It is noted that mean value of non defaulters(334990) and defaulters(319275) lies almost close.
#But we have seen the defaulters percentage is  around 25% and this 25 % is contributing to default amount of 319275.


# In[287]:


sns.boxplot(data=df_new,x='Status',y='loan_amount')
plt.title('Loan Amount vs Status')
plt.show()


# # Rate of Interest & Status

# In[263]:


#Null Hypothesis--There is no statsical correlation between rate of interest and status
#Alternate Hypothesis--There is a statsical relation between rate of interest and laon status
from scipy.stats import ttest_ind
stat,p_val=ttest_ind(df_new['Status'],df_new['rate_of_interest'])
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')


# In[202]:


sns.scatterplot(data=df_new,hue='Status',y='rate_of_interest',x='loan_amount')
plt.show()


# Insight-Based on the data and hypothesis tesing it is clear that rate of interst does not have an impact on loan status.
# 4% is the interest charged for most loans

# In[264]:


#Checking the correlation with loan amount and rate of interest for each defaulter and non defaulter
non_default=df_new[df_new['Status']==0]
corr_non_defaulters=non_default['rate_of_interest'].corr(non_default['loan_amount'])

defaulter=df_new[df_new['Status']==1]
corr_defaulters=defaulter['rate_of_interest'].corr(defaulter['loan_amount'])

print(f'Correlation for non defaulters is {corr_non_defaulters}')
print(f'Correlation for non defaulters is {corr_defaulters}')


# #Inference:There is no correaltion between the given variables with the status of the loan

# In[265]:


#Null Hypothesis--There is no statsical correlation between income and status
#Alternate Hypothesis--There is a statsical relation between income and laon status
stat,p_val=ttest_ind(df_new['Status'],df_new['income'])
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('Failed to reject null hypothesis')
    
    
sns.violinplot(data=df_new,x='Status',y='income',palette='coolwarm')
plt.show()    


# In[268]:


sns.scatterplot(data=df_sample,x='loan_amount',y='income',hue='Status')
plt.title('Income vs Loan Amount')
plt.show()


# In[270]:


defaulter=df_new[df_new['Status']==1]
non_defaulter=df_new[df_new['Status']==0]
print('Defaulter')
print(defaulter[['income','loan_amount']].describe())
print('*'*40)
print('Non Defaulter')
print(non_defaulter[['income','loan_amount']].describe())


# Insight-It is obvious from the test and the plot above that income does have a statstical significance on loan status.
# The default cases can be attributed to income of the individual.The mean income of non defaulter comes close around 6151.
# The mean income of default cases is around 5470.

# # Credit Score & Status

# In[207]:


df.groupby('Status')['Credit_Score'].describe()


# In[286]:


sns.boxplot(data=df_new,x='Status',y='Credit_Score')  
plt.title('Credit Score vs Loan Status')
plt.show()


# In[271]:


sns.boxplot(data=df_new, x='Status', y='Credit_Score', hue='credit_type')
plt.title('Credit Score Distribution by Loan Status')
plt.legend(loc='lower left')
plt.show()


# In[ ]:


#Insight_if you consider all the 4 credit score type they have similar perfomace in the loan status as per the given data
#The average score  all the score types comes areound 700
#It is also noted that customers with higher credit scores above 700 also tends to default


# In[ ]:


df_new.groupby('Status')['LTV'].describe()


# In[ ]:


#Null Hypothesis--There is no statsical correlation between LTV and status
#Alternate Hypothesis--There is a statsical relation between LTV and laon status
stat,p_val=ttest_ind(df_new['Status'],df_new['LTV'])
alp=.05
if p_val<alp:
    print('Null Hypothesis is rejected')
else:
    print('failed to reject null hypothesis')
    
sns.boxplot(data=df_new,x='Status',y='LTV')    
plt.show()


# Insights-There is a correlation between LTV and status of the loan.

# In[211]:


sns.scatterplot(data=df_sample,x='LTV',y='loan_amount',hue='Status')
plt.title('Loan Amount and LTV')
plt.show()


# In[289]:


sns.scatterplot(data=df_sample,x='property_value',y='loan_amount',hue='Status')
plt.title('Loan Amount and Property Value')
plt.show()


# # Upfront Charges & Loan Type

# In[83]:


df.groupby('loan_type')['Upfront_charges'].describe()


# In[231]:


sns.barplot(data=df_new,x='loan_type',y='Upfront_charges',estimator='mean',hue='Status')
plt.title('Upfront Charges vs Loan Type')
plt.show()


# In[290]:


sns.barplot(data=df_new,x='loan_purpose',y='Upfront_charges',estimator='mean',hue='Status')
plt.title('Upfront Charges vs Loan Type')
plt.show()


# In[233]:


sns.scatterplot(data=df_sample,x='loan_amount',y='Upfront_charges',hue='Status')
plt.show()


# Insight--
# #Most upfront charges are paid for loan type 1 followed by loan type 3.As default cases are more in type1 it is
# #necessary that upfront charges paid are also higher.
# #Upfront charges paid above 2500 have lower chance of default
# #Upfront charges for type 2 and type 3 could be raised higher for lowering default rates

# The distribution clearly indicates instances of default are higher for income  <4000.

# In[ ]:


#Insights--It is clear that loan application with upfront charges paid have a lower tendency to get defaulted.


# # Property Value & Loan Status

# In[277]:


sns.scatterplot(data=df_new,x='property_value',y='LTV',hue='Status')
plt.show()


# In[278]:


df_new.groupby('Status')['property_value'].describe()


# In[279]:


df_new.groupby('Status')['LTV'].describe()


# In[92]:


corr_loan_features=['loan_amount','LTV','rate_of_interest','Status']
corr_loan_matrix=df_new[corr_loan_features].corr()
sns.heatmap(corr_loan_matrix,annot=True,cmap='coolwarm')
plt.show()


# In[ ]:





# In[ ]:


sns.countplot(data=df_new,x='credit_type',hue='Status')


# 25.	Do applicants with high upfront_charges have lower default rates?
# 

# In[ ]:


df_new['Upfront_charges'].isnull().sum()


# In[ ]:


print(df_new['Upfront_charges']<0)
print(df_new['Upfront_charges']>9725)


# In[ ]:


bins=np.linspace(0,9725,num=6)
labels=['low','mid low','medium','high','Extremely high']
df_new['Upfront_charges_cat']=pd.cut(df_new['Upfront_charges'],bins=bins,labels=labels)
df_new['Upfront_charges_cat'].value_counts().index


# In[ ]:


sns.barplot(x=df_new['Upfront_charges_cat'].index,y=df_new['Upfront_charges_cat'].values)
plt.show()


# FEATURE ENGINEERING

# In[ ]:


#DEBT TO INCOME RATIO
df_new['DTI']=df_new['loan_amount']/df_new['income']

#Interest to Income ratio
df_new['interest_income_ratio']=(df_new['loan_amount']*df_new['rate_of_interest']/100)/df_new['income']


# In[274]:


#RECOMMENDATIONS
#1)The mean loan amount for defaulters and non defaulters differ only by a small margin despite the lower share
#of defaulters in the data set.This is quite alarming and the company needs a greater scrutiny on this aspect.

#2)Gender Impact-The largest number of loan apllicants are for 'Male' category.When the application is coobligant in nature,
#the loan amount sanctioned has higher value and the default percentage is lower for this category.
#The fewer loan application by 'Female' category must be inspected.
#Promote the factor of coobligancy for improving the repayment status.

#3)Commercial nature-Most of the loans are of non commercial in nature

#4)Loan type--
#Focus on Risk Mitigation for Type 2: Tighten eligibility criteria and introduce stricter repayment plans for Type 2 loans, 
#as they have the highest default rate.
#Leverage Type 1 Loans: Promote Type 1 loans further, as they are likely non-commercial and show lower risk. 
#Offer preferential terms to attract more borrowers.
#Optimize Type 3 Allocation: Given the large amounts disbursed for Type 3, likely commercial loans,
#enhance risk assessment and monitoring to prevent high-value defaults.
#Segmentation Analysis: Conduct detailed studies to confirm the commercial/non-commercial nature of loans and tailor policies 
#accordingly for growth and risk management.

#5)Loan Purpose-
#Tighten P2 Policies: Reduce default risks for P2 by stricter eligibility, smaller loan caps, and enhanced monitoring.
#Promote P1 Loans: Expand P1 loans, leveraging their low default rates with incentives like lower interest rates.
#Strengthen P4 Oversight: Ensure profitability for P4, which has the highest loan allocation, 
#by monitoring repayments and offering early payment benefits.
#Optimize P3 Strategy: Address high demand for P3 by tailoring loan products and assessing profitability.
#Educate Borrowers: Reduce defaults through financial literacy programs and personalized consultations, particularly for P2 and P4 categories.

#)Region
#Focus on the North-East: Increase loan allocation in the North-East through tailored loan products and incentives,
#while closely monitoring default risks to ensure sustainable growth.The default percentage is the highest in North East
#Leverage Northern Region Performance: Expand loan offerings in the North, capitalizing on low defaults and high allocations, 
#to maximize profitability.
#Strengthen Central Region Policies: Introduce stricter controls in the Central region to address its relatively high default percentage.
#Regional Risk Segmentation: Tailor lending strategies based on regional performance and default behavior.

#7)Age
#Focus on Age <25: Promote loans to this group due to high credit scores in non-default cases. Increase the loan amount upper threshold to attract more applicants.
#Mitigate Risks for Age >74: Tighten eligibility and introduce collateral-based loans to address high default rates.
#Support Age 45-54: Provide financial counseling and flexible repayment options to reduce defaults in this high-risk group.
#Expand for Age 35-44: Retain this largest applicant base with competitive terms and personalized loan products.
#Age-Based Strategies: Design tailored loan offerings based on age-specific credit behaviors for growth and risk management


#9)#Implementation of Upfront charges for 'type 2' and 'type 3' could reduce the risk of defaulting.
#Most upfront charges are paid for 'type1' loan.

#10)Income level below 6000 is considered as riskier.The capping for loan amout range between 300000 and 400000 is also
#at risk of efault

#11)Credit Score.The normal accepted credit score in the data set is 650-700 range.There are instances of default 
#regardless of high credit score.




# In[ ]:





# In[ ]:




