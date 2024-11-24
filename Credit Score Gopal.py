#!/usr/bin/env python
# coding: utf-8

# In[63]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer


# In[64]:


import warnings
warnings.simplefilter('ignore')


# In[65]:


df=pd.read_csv('Credit_score.csv')


# PROBLEM STATEMENT:We are conducting this case study to formulate a credit score based on the parametres given in the data set

# In[66]:


df.head()


# In[67]:


df.shape


# There are 1 lakhs rows and 27 coloumns in this table

# In[68]:


df.describe()


# In[69]:


df.columns


# Column Description
# 
# ID--Represents a unique identification of an entry
# Customer_ID	--Represents a unique identification of a person 
# Month--Represents the month of the year
# Name--Represents the name of a person
# Age--Represents the age of the person
# SSN--Represents the social security number of a person
# Occupation--Represents the occupation of the person
# Annual_Income--Represents the annual income of the person
# Monthly_Inhand_Salary--Represents the monthly base salary of a person
# Num_Bank_Accounts--Represents the number of bank accounts a person holds
# Num_Credit_Card--Represents the number of other credit cards held by a person
# Interest_Rate--Represents the interest rate on credit card
# Num_of_Loan--Represents the number of loans taken from the bank
# Type_of_Loans--Represents the types of loan taken by a person
# Delay_from_due_date--Represents the average number of days delayed from the payment date
# Num_of_Delayed_Payment--Represents the average number of payments delayed by a person
# Changed_Credit_Limit--Represents the percentage change in credit card limit
# Num_Credit_Inquiries--Represents the number of credit card inquiries
# Credit_Mix--Represents the classification of the mix of credits
# Outstanding_Debt--Represents the remaining debt to be paid (in USD)
# Credit_Utilization_Ratio--Represents the utilization ratio of credit card
# Credit_History_Age--Represents the age of credit history of the person
# Payment_of_Min_Amount--Represents whether only the minimum amount was paid by the person
# Total_EMI_per_month--Represents the monthly EMI payments (in USD)
# Amount_invested_monthly--Represents the monthly amount invested by the customer (in USD)
# Payment_Behaviour--Represents the payment behavior of the customer (in USD)
# Monthly_Balance--Represents the monthly balance amount of the customer (in USD)
# 

# In[70]:


df.dtypes


# Since many of the columns which are supposed to be integer or float type we are converting them using forced conversions as mentioned below.

# In[71]:


col_list=[ 'Age', 'Annual_Income', 'Num_of_Loan', 'Delay_from_due_date',
       'Num_of_Delayed_Payment', 'Changed_Credit_Limit','Outstanding_Debt',
        'Amount_invested_monthly', 'Monthly_Balance']

for col in col_list:
    df[col] = pd.to_numeric(df[col], errors='coerce')


    df[col] = df[col].astype('float64')


# In[72]:


df.dtypes


# In[73]:


df.drop(columns=['SSN','ID'],axis=1,inplace=True)


# In[74]:


df.head()


# DATA CLEANING:Cleaning the coloumns 'Type_of_Loan' and 'Payment_Behaviour'

# In[75]:


df['Type_of_Loan'].head(20)


# In[76]:


df['Type_of_Loan'] = df['Type_of_Loan'].astype(str)
def clean_loans(loans):
    loan_list=loans.replace('and',',').split(',')
    cleaned_loans = [loan.strip() for loan in loan_list if loan.strip()]
    unique_loans=set(cleaned_loans)
    unique_loans.discard('Not Specified')
    return ','.join(unique_loans)
df['Type_of_Loan']=df['Type_of_Loan'].apply(clean_loans)


# In[77]:


df['Type_of_Loan']


# In[78]:


df['Payment_Behaviour']


# In[79]:


import re  ##use regex to filter the words for spent and value
def clean_payment_type(payment):
    if isinstance(payment, str):
        
        cleaned_payment = re.sub(r'[^a-zA-Z0-9_ ]', '', payment)
        
        cleaned_payment = cleaned_payment.strip()
        
        if not cleaned_payment:
            return 'Invalid'  
        return cleaned_payment
    else:
    
        return 'Invalid'
df['Payment_Behaviour']=df['Payment_Behaviour'].apply(clean_payment_type)  


# In[80]:


df['Payment_Behaviour']


# In[81]:


def payment_split(payment):
    if isinstance(payment,str):
        words=payment.replace('_',' ').split()
        if len(words)>2:
            return f'{words[0]}{words[1]} {words[2]}{words[3]}'#combining spent and value
        else:
            return payment
    else:
        return payment


# In[82]:


df['Payment_Behaviour']=df['Payment_Behaviour'].apply(payment_split)


# In[83]:


df['Payment_Behaviour'].value_counts()


# In[84]:


df['Payment_Behaviour'].unique()


# In[85]:


mode_val=df['Payment_Behaviour'].mode()[0]
df['Payment_Behaviour'].replace('98',mode_val,inplace=True)


# Encoding--Using a Label encoder we are mapping a numeric value to the payment behaviour class

# In[86]:


from sklearn.preprocessing import LabelEncoder
label_encoder=LabelEncoder()
df['Payment_Behaviour_encoded']=label_encoder.fit_transform(df['Payment_Behaviour'])


# In[ ]:





# In[152]:


df['Payment_Behaviour_encoded'].head(20),df['Payment_Behaviour'].head(20)


# ENCODING MAPPING
# 0-HIGH SPENT LARGE VALUE
# 1-HIGH SPENT MEDIUM VALUE
# 2-HIGH SPENT SMALL VALUE
# 3-LOW SPENT LARGE VALUE
# 4-LOW SPENT MEDIUM VALUE
# 5-LOW SPENT SMALL VALUE

# custom_mapping = {
#     'Lowspent Smallvalue': 0,
#     'Highspent Mediumvalue':4 ,
#     'Lowspent Mediumvalue': 1,
#     'Highspent Largevalue':5 ,
#     'Highspent Smallvalue': 3,
#     'Lowspent Largevalue': 2
# }
# df['Payment_mapped']=df['Payment_Behaviour_encoded'].map(custom_mapping)

# Lowspent Smallvalue --2
# Highspent Mediumvalue--0
# Lowspent Mediumvalue--3
# Highspent Largevalue--1
# Highspent Smallvalue--2
# Lowspent Largevalue--5

# In[88]:


df['Payment_Behaviour_encoded']


# Convert 'Credit_History_Age' column to string.Fill the null values using mode values.Then convert the column to numerical column

# In[89]:


df['Credit_History_Age'].astype(str)


# In[90]:


mode_hist=df['Credit_History_Age'].mode()[0]
df['Credit_History_Age'].fillna(mode_hist,inplace=True)


# In[91]:


df['Credit_History_Age'].isna().sum()


# Converting Credit_History column to a numerical column

# In[92]:


def convert_history(credit_hist):
    if credit_hist=='NA':
        return np.nan
    part=credit_hist.split('and')
    year=float(part[0].split()[0])
    month=float(part[1].split()[0])
    return year+(month/12)
    


# In[93]:


df['credit_history']=df['Credit_History_Age'].apply(convert_history)


# In[94]:


df['credit_history']
df.drop(columns='Credit_History_Age',axis=1,inplace=True)


# One Hot Encoding on Type of Loan

# In[95]:


df['Type_of_Loan'].value_counts()


# In[96]:


df.dtypes


# FILLING NULL VALUES

# In[97]:


df['Monthly_Salary'] = df.groupby('Customer_ID')['Monthly_Inhand_Salary'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df.drop(columns='Monthly_Inhand_Salary',axis=1,inplace=True)


# In[98]:


df['Age'] = df.groupby('Customer_ID')['Age'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Occupation'] = df.groupby('Customer_ID')['Occupation'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Name'] = df.groupby('Customer_ID')['Name'].transform(lambda x: x.fillna(method='ffill').fillna(method='bfill'))
df['Annual_Income']=df.groupby('Customer_ID')['Annual_Income'].transform(lambda x:x.fillna(method='ffill').fillna(method='bfill'))




# In[99]:


df['Num_of_Loan'].fillna(0,inplace=True)


# In[101]:


df.dtypes


# In[102]:


mean_val_amnt=df['Amount_invested_monthly'].mean()
print(mean_val_amnt)
df['Amount_invested_monthly'].fillna(mean_val_amnt,inplace=True)


# In[103]:


monthly_bal_median=df['Monthly_Balance'].median()
df['Monthly_Balance'].fillna(monthly_bal_median,inplace=True)


# In[104]:


df['Type_of_Loan'].fillna('Not Specified',inplace=True)


# In[105]:


mode_delayed=df['Num_of_Delayed_Payment'].median()
df['Num_of_Delayed_Payment'].fillna(mode_delayed,inplace=True)


# In[106]:


median_changedlimit=df['Changed_Credit_Limit'].median()
df['Changed_Credit_Limit'].fillna(median_changedlimit,inplace=True)


# In[107]:


median_credit_inq=df['Num_Credit_Inquiries'].median()
df['Num_Credit_Inquiries'].fillna(median_credit_inq,inplace=True)


# In[108]:


med_outstand=df['Outstanding_Debt'].median()
df['Outstanding_Debt'].fillna(med_outstand,inplace=True)


# In[109]:


df.isnull().sum()


# OUTLIERS
# 

# In[110]:


num_col=df.select_dtypes(include=['number'])



# The presence of outlier is evident and hence treatment of the same is necessary using IQR method

# In[111]:


num_col.columns


# num_col.head()
# num_col.drop(['Age', 'Annual_Income', 'Num_Bank_Accounts', 'Num_Credit_Card',
#        'Interest_Rate','Num_of_Loan','Num_of_Delayed_Payment','Num_Credit_Inquiries','Total_EMI_per_month',axis=1])

# In[112]:


for col in enumerate(num_col):
    sns.boxplot(x=col[1],data=num_col)
    plt.show()


# Since there are outliers present in the data we need to remove them using a suitable method.Here we will use IQR method to
# clear the outliers

# In[113]:


Q1=num_col.quantile(0.25)
Q3=num_col.quantile(0.75)
IQR=Q3-Q1

mask=~((num_col< (Q1-1.5*IQR))|(num_col > (Q3 + 1.5*IQR))).any(axis=1)
df_new=df[mask]
df_new


# In[114]:


for col in enumerate(num_col):
    sns.boxplot(x=col[1],data=num_col)
    plt.show()


# In[115]:


from sklearn.preprocessing import StandardScaler
ss=StandardScaler()


# UNIVARAIATE ANALYSIS

# In[116]:


df_sample=df_new.sample(n=1000,random_state=42)


# In[117]:


df_sample


# In[118]:


df['Age']


# In[119]:


sns.histplot(df_sample['Age'],bins=10,kde=True)


# In[177]:


plt.figure(figsize=(8,4))
sns.countplot(x='Occupation',data=df_sample)
plt.xticks(rotation=45)
plt.show()


# In[310]:


sns.countplot(x='Num_Bank_Accounts',data=df_sample)


# In[311]:


sns.histplot(df_sample['Monthly_Balance'],bins=10,kde=True)


# In[233]:


sns.histplot(df_sample['Credit_Utilization_Ratio'],bins=10,kde=True)


# In[312]:


sns.histplot(df_sample['Interest_Rate'],bins=10,kde=True)


# BI VARIATE ANALYSIS

# In[120]:


num2_col=df_new.select_dtypes(include=['number'])


# In[121]:


correlation_mat=num2_col.corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation_mat,cmap='coolwarm',annot=True)


# In[122]:


df_new.columns


# In[172]:


sns.scatterplot(data=df_sample,x='Monthly_Balance',y='Outstanding_Debt')


# Users with higher monthly balance has lower outstanding debt

# In[148]:


sns.barplot(data=df_sample,x='Payment_Behaviour_encoded',y='Outstanding_Debt')


# 0-HIGH SPENT LARGE VALUE
# 1-HIGH SPENT MEDIUM VALUE
# 2-HIGH SPENT SMALL VALUE
# 3-LOW SPENT LARGE VALUE
# 4-LOW SPENT MEDIUM VALUE
# 5-LOW SPENT SMALL VALUE 
# 
# It is clear that High spent high value customers have lower outstanding debt may be due to their higher incomes

# In[149]:


df['Payment_Behaviour_encoded']


# In[147]:


sns.scatterplot(data=df_sample,y='Num_of_Delayed_Payment',x='Outstanding_Debt')


# It is clear indicato that as outstanding debt increases delayed payments also increases.Also smaller outsanding debts
# has higher cluster of delayed payments between 10 and 15

# In[146]:


sns.scatterplot(data=df_sample,x='Annual_Income',y='Num_Credit_Card')


# Higher the annual income fewer the subscription for credit cards

# In[145]:


sns.scatterplot(data=df_sample,x='Age',y='Num_of_Loan',hue='Occupation')


# In[123]:


sns.scatterplot(data=df_sample,x='Interest_Rate',y='Num_Bank_Accounts')


# Clearly as number of bank accounts increases intrest rate charged also increases

# In[124]:


sns.scatterplot(data=df_sample,x='Interest_Rate',y='Outstanding_Debt')


# As outstanding debt increases intrest rate also increases

# In[125]:


sns.scatterplot(data=df_sample,x='Num_Credit_Inquiries',y='Outstanding_Debt')


# In[ ]:


sns.


# In[ ]:





# It is clear as outstanding debt increases credit enquiry increases

# Feature Engineering:--Let's select some of the columns presented in our data set to arrive at the
# credit score calculation
# 
# FICO SCORE--A FICO credit score is a numerical representation of a person's creditworthiness, typically ranging from 300 to 850. The calculation of this score is based on five key factors, each weighted differently.
# 
# Payment History (30%)
# This is the most significant factor, reflecting whether a person has paid their credit accounts on time. Late payments, bankruptcies, and collections negatively impact this component.
# 
# Amounts Owed (30%)
# This factor considers the total amount of debt relative to available credit, known as the credit utilization ratio. A lower utilization ratio is preferable, ideally below 30%, as high amounts owed can indicate risk.
# 
# Length of Credit History (15%)
# A longer credit history generally contributes positively to the score. This includes the age of the oldest account, the newest account, and the average age of all accounts.
# 
# Outstanding Debt:It represents the remaning amount to be paid to clear the loan
# 
# Credit Type Score (10%)
# This includes recent credit inquiries and newly opened loan accounts. Frequent applications for new credit and along with the exisisting loan accounts  can be seen as risky behavior and may lower the score.
# 
# Credit Mix (10%)
# A diverse range of credit types (e.g., credit cards, mortgages, installment loans) can positively influence the score. However, it’s not necessary to have one of each type.

# df['Payment_History_score']=df['Num_of_Delayed_Payment'] + (df['Payment_of_Min_Amount'] == 0                                   For arriving at payment hitory we use the coloumns    Num_of_Delayed_Payment' and ['Payment_of_Min_Amount'] == 0 
# These 2 gives a fair idea of track record of the loan repayments

# df['Credit_utilisation_ratio']##It is directly provided in the data set

# df['credit_history']##The column was pre processed at the beginning of the document and onverted to a numerical column
# 

# df['Credit_Type_Score']=df['Num_of_loan']+df['Num_Credit_Card']
# It gives an idea about the amount of liability a person owes currently

# Column--- 'Credit_Mix' is provided in our data set as categorical.We are encoding using the label encoder to 
# process the credit score which is provided below

# In[153]:


df=df_new


# In[154]:


mode_mix=df['Credit_Mix'].mode()[0]
mode_mix
df['Credit_Mix'].replace('_',mode_mix,inplace=True)


# In[155]:


df['Credit_Mix'].unique()


# In[156]:


df['Credit_Mix']=label_encoder.fit_transform(df['Credit_Mix'])


# In[157]:


df['Credit_Mix']


# Monthly Investment Ratio

# In[158]:


from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()


# In[159]:


df[['Payment_History_Score', 'Credit_Utilization_Score', 'Credit_History_Length_Score',
    'Outstanding_Debt_Score', 'Credit_Type_Score', 'Credit_Mix_Score']] = scaler.fit_transform(
    pd.DataFrame([df['Num_of_Delayed_Payment'] + (df['Payment_of_Min_Amount'] == 0),  # 
                  df['Credit_Utilization_Ratio'],  # Credit utilization score
                  df['credit_history'],  # Credit history length score
                  df['Outstanding_Debt'],  # Outstanding debt score
                  df['Num_of_Loan'] + df['Num_Credit_Card'],  # Types of credit score
                  df['Credit_Mix']  # Credit inquiries score
                 ]).T)


# In[160]:


df['Credit_Score'] = (0.30 * df['Payment_History_Score'] +
                      0.30 * df['Credit_Utilization_Score'] +
                      0.15 * df['Credit_History_Length_Score'] +
                      0.10 * df['Outstanding_Debt_Score'] +
                      0.10 * df['Credit_Type_Score'] +
                      0.10 * df['Credit_Mix_Score'])

# Scale the credit score to 5-10 range (adjusted scaling)
df['Credit_Score'] = 5 + 5 * df['Credit_Score']


# In[161]:


df['Credit_Score']


# In[162]:


sns.histplot(df['Credit_Score'],kde=True)


# In[140]:


df_sample2=df.sample(n=5000,random_state=50)


# In[141]:


df_sample2


# In[142]:


sns.histplot(df_sample2['Credit_Score'],kde=True)
plt.show()


# In[ ]:





# In[163]:


df.columns


# In[166]:


sns.scatterplot(data=df_sample2,x='Num_Credit_Card',y='Credit_Score',hue='Outstanding_Debt')


# In[178]:


sns.scatterplot(data=df_sample2,x='Payment_of_Min_Amount',y='Credit_Score',hue='Outstanding_Debt')


# In[170]:


sns.scatterplot(data=df_sample2,x='credit_history',y='Credit_Score',hue='Credit_Mix')


# In[168]:


sns.scatterplot(data=df_sample2,x='Payment_Behaviour_encoded',y='Credit_Score')


# Most of the applicants have a credit score in the range 7-8 which reflects a healthy credit culture

# In[143]:


sns.scatterplot(x='Age',y='Credit_Score',data=df_sample2.head(50))
plt.show()


# Age and credit score does not seems to bear a relation

# INFERENCE:
# 1)The credit score data reveals that most of the scores lies between 7 and 8.
# 2)The age group of 40+ takes fewer credit.The age group 30-40 has highest takers of loans
# 3)Age group 30-40 displays the highest credit scores
# 4)Users with HIGH SPENT HIGH VALUE and LOW SPENT LOW VALUE have higher credit scores
# 5)Users with longer credit history and better credit mix has higher credit score
# 6)Users wo paid minimum amount has better credit scores compared to users who dont pay anything
#     
