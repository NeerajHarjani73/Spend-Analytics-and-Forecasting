#Importing required packages
# pip install pandas-profiling
import numpy as np
import pandas as pd
import random
#import matplotlib.pylab as plt
#import seaborn as sns

df = pd.read_excel('C:/Users/neera/Documents/Incture/Spend_DataSet_Internship.xlsx')
df.profile_report()
df.info()
df.shape
df.info()
unique_vendors=df.VENDOR.value_counts()
len(unique_vendors)

unique_categories=df.CATEGORY.value_counts()
len(unique_categories)
df_detail=df.describe()
df.isnull().sum()

unique_products = pd.DataFrame(df['PRODUCT'].unique())
len(unique_products)
df['PRODUCT']=df['PRODUCT'].replace({'NOT PROVIDED IN FILES': 'Others'},regex=True)
(df['PRODUCT']=='NOT PROVIDED IN FILES').unique()

len(df['PRODUCT'].unique())
x=set(np.arange(0,143020))
x_random = set(random.sample(x,100000))
a=df.loc[x_random,'PRODUCT']
a=pd.DataFrame(a)
a['Index'] = a.index
a['VENDOR']= df.loc[a.loc[:,'Index'],'VENDOR']
df3 = pd.DataFrame((df.PRODUCT.isin(a.PRODUCT))&(df.VENDOR.isin(a.VENDOR)))
df['SOS(SourceOFSupply)'] = df3[0] 

df['INVOICE_DATE'] = np.random.choice(pd.date_range('2018-01-01', '2019-12-31'), len(df['INVOICE_DATE']))

y = set(np.arange(0,688053))
ypart1 = set(random.sample(y,344026))
y = y-ypart1
ypart2 = set(random.sample(y,206415))
y = y - ypart2
ypart3 = set(random.sample(y,68807))
y = y - ypart3
ypart4 = set(random.sample(y,68805))
y = y - ypart4

ypart1 = pd.DataFrame(ypart1)
ypart2 = pd.DataFrame(ypart2)
ypart3 = pd.DataFrame(ypart3)
ypart4 = pd.DataFrame(ypart4)
df.loc[ypart1.loc[:,0],'PO_AMT_USD'] = df.loc[ypart1.loc[:,0],'INV_AMOUNT_USD']
df.loc[ypart2.loc[:,0],'PO_AMT_USD'] = df.loc[ypart2.loc[:,0],'INV_AMOUNT_USD']*0.74
df.loc[ypart3.loc[:,0],'PO_AMT_USD'] = df.loc[ypart3.loc[:,0],'INV_AMOUNT_USD']*0.64
df.loc[ypart4.loc[:,0],'PO_AMT_USD'] = df.loc[ypart4.loc[:,0],'INV_AMOUNT_USD'] + (df.loc[ypart4.loc[:,0],'INV_AMOUNT_USD']*0.10)

#Calculating 'Product Unit Price' for each product
df['Invoice Qty'] = df['PO Qty']
df['Product Unit Price'] = df['PO_AMT_USD']/df['PO Qty']

x = set(np.arange(0,688053))
xpart1 = set(random.sample(x,412831))
x = x - xpart1
xpart2 = set(random.sample(x,172015))
x = x - xpart2
xpart3 = set(random.sample(x,80000))
x = x - xpart3
xpart4 = set(random.sample(x,23207))
x = x - xpart4
xpart1 = pd.DataFrame(xpart1)
xpart2 = pd.DataFrame(xpart2)
xpart3 = pd.DataFrame(xpart3)
xpart4 = pd.DataFrame(xpart4)
df.loc[xpart1.loc[:,0],'Received Qty'] = df.loc[xpart1.loc[:,0],'PO Qty']
df.loc[xpart2.loc[:,0],'Received Qty'] = (df.loc[xpart2.loc[:,0],'PO Qty'] - (df.loc[xpart2.loc[:,0],'PO Qty']*0.10)).round(0)
df.loc[xpart3.loc[:,0],'Received Qty'] = (df.loc[xpart3.loc[:,0],'PO Qty'] - (df.loc[xpart3.loc[:,0],'PO Qty']*0.20)).round(0)
df.loc[xpart4.loc[:,0],'Received Qty'] = (df.loc[xpart4.loc[:,0],'PO Qty'] + (df.loc[xpart4.loc[:,0],'PO Qty']*0.10)).round(0)

df.to_csv('C:/Users/neera/Documents/Incture/Spend_DataSet_Internship_Enriched.csv')

df2 = df.copy()
#Converting the product and category level as indirect spends
df.loc[df['CATEGORY']=='NOT ASSIGNED', 'SOS(SourceOFSupply)'] = False
df.loc[df['PRODUCT']=='NOT PROVIDED IN FILES', 'SOS(SourceOFSupply)'] = False

#Vendor Competitive Analysis
vend_comp=df2.groupby(['PRODUCT','VENDOR'])['INV_AMOUNT_USD'].sum().reset_index()
vend_comp_invoiceqty = (df2.groupby(['PRODUCT','VENDOR'])['Invoice Qty'].sum()).reset_index()
vend_comp['Invoice Qty'] = vend_comp_invoiceqty['Invoice Qty'].copy()
vend_comp['Unit Price']=vend_comp['INV_AMOUNT_USD']/vend_comp['Invoice Qty']
vend_comp_sample=vend_comp[(vend_comp.PRODUCT == '1,3-DIISOPROPANOLBENZENE -FIBC')]

#Purchase price fluctuation
price_fluctuate=df2.groupby(['VENDOR','PRODUCT','INVOICE_DATE'])['INV_AMOUNT_USD'].sum().reset_index()
price_fluctuate_invoiceqty = (df2.groupby(['VENDOR','PRODUCT','INVOICE_DATE'])['Invoice Qty'].sum()).reset_index()
price_fluctuate['Invoice Qty'] = price_fluctuate_invoiceqty['Invoice Qty'].copy()
price_fluctuate['Unit Price']=price_fluctuate['INV_AMOUNT_USD']/price_fluctuate['Invoice Qty']
price_fluctuate_sample=price_fluctuate[(price_fluctuate.VENDOR == 'A 1 WHEEL')&(price_fluctuate.PRODUCT == 'DISPOSAL FEE')]

df2.to_csv('C:/Users/neera/Documents/Incture/Spend_DataSet_Internship_Enriched_Version2.csv')

df['INV_AMOUNT_USD'].sum()
df.groupby(df['INVOICE_DATE'].dt.year)['INV_AMOUNT_USD'].sum()
df.loc[df.PRODUCT=='25367     DISPERSION','INV_AMOUNT_USD'].sum()


df2['diff']=(df['Received Qty']/df['Invoice Qty'])*100
Greater_than_80 = df2[(df2['diff'] > 80) & (df2['diff'] < 101)][['VENDOR','PRODUCT','Received Qty','Invoice Qty','diff']]
Greater_than_50 = df2[(df2['diff'] > 50) & (df2['diff'] < 80)][['VENDOR','PRODUCT','Received Qty','Invoice Qty','diff']]
Greater_than_25 = df2[(df2['diff'] > 25) & (df2['diff'] < 50)][['VENDOR','PRODUCT','Received Qty','Invoice Qty','diff']]
Less_than_25 = df2[(df2['diff'] < 25)][['VENDOR','PRODUCT','Received Qty','Invoice Qty','diff']]


n = set(not_in_files.index)
npart1 = set(random.sample(n,15614))
n = n - npart1
npart2 = set(random.sample(n,15614))
n = n - npart2
npart3 = set(random.sample(n,15614))
n = n - npart3
npart4 = set(random.sample(n,15614))
n = n - npart4
npart5 = set(random.sample(n,15614))
n = n - npart5
npart6 = set(random.sample(n,15614))
n = n - npart6
npart7 = set(random.sample(n,15614))
n = n - npart7
npart8 = set(random.sample(n,15614))
n = n - npart8
npart9 = set(random.sample(n,15614))
n = n - npart9
npart10 = set(random.sample(n,15614))
n = n - npart10
len(n)
npart1 = pd.DataFrame(npart1)
npart2 = pd.DataFrame(npart2)
npart3 = pd.DataFrame(npart3)
npart4 = pd.DataFrame(npart4)
npart5 = pd.DataFrame(npart5)
npart6 = pd.DataFrame(npart6)
npart7 = pd.DataFrame(npart7)
npart8 = pd.DataFrame(npart8)
npart9 = pd.DataFrame(npart9)
npart10 = pd.DataFrame(npart10)

#BOBINA SOL. 220V 60HZ 099216001B ASCO
#TRAVAUX CELLULES SEM 2
#PLATFORM ROLLERS
#BRASNOX AAP 25KG DRUM UNG
#KROMASIL 100-5-PHENYL 3.9X150 MM
#PIT-9303&3757
#HERTEL: STEIGER AANPASSING CA423C
#SERV. ANDAMIOS 1° FEB'19
#PREVENTIVE MAINTENANCE.
#2019 CYLINDER RENTAL LAB MAINTNEACE

df2.loc[npart1.loc[:,0],'PRODUCT'] = 'BOBINA SOL. 220V 60HZ 099216001B ASCO'
df2.loc[npart2.loc[:,0],'PRODUCT'] = 'TRAVAUX CELLULES SEM 2'
df2.loc[npart3.loc[:,0],'PRODUCT'] = 'PLATFORM ROLLERS'
df2.loc[npart4.loc[:,0],'PRODUCT'] = 'BRASNOX AAP 25KG DRUM UNG'
df2.loc[npart5.loc[:,0],'PRODUCT'] = 'KROMASIL 100-5-PHENYL 3.9X150 MM'
df2.loc[npart6.loc[:,0],'PRODUCT'] = 'PIT-9303&3757'
df2.loc[npart7.loc[:,0],'PRODUCT'] = 'HERTEL: STEIGER AANPASSING CA423C'
df2.loc[npart8.loc[:,0],'PRODUCT'] = "SERV. ANDAMIOS 1° FEB'19"
df2.loc[npart9.loc[:,0],'PRODUCT'] = 'PREVENTIVE MAINTENANCE.'
df2.loc[npart10.loc[:,0],'PRODUCT'] = '2019 CYLINDER RENTAL LAB MAINTNEACE'

n_60= pd.concat([npart1,npart2,npart3,npart4,npart5,npart6,npart7,npart8,npart9,npart10], ignore_index=True)

df2.loc[n_60_random.loc[:,0],'SOS(SourceOFSupply)'] = True

df2.to_csv('C:/Users/neera/Documents/Incture/Spend_DataSet_Internship_Enriched_Version2.csv')