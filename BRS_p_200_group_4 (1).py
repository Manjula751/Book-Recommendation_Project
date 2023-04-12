#!/usr/bin/env python
# coding: utf-8

# # Business Objective:
# Generate the features from the dataset and use them to recommend the books accordingly to the users. Content The Book-Crossing dataset comprises 3 files.
# ● Users
# Contains the users. Note that user IDs (User-ID) have been anonymized and map to integers. Demographic data is provided (Location, Age) if available. Otherwise, these fields contain NULL-values.
# ● Books
# Books are identified by their respective ISBN. Invalid ISBNs have already been removed from the dataset. Moreover, some content-based information is given (Book-Title, Book-Author, Year-Of-Publication, Publisher), obtained from Amazon Web Services. Note that in case of several authors, only the first is provided. URLs linking to cover images are also given, appearing in three different flavours (Image-URL-S, Image-URL-M, Image-URL-L), i.e., small, medium, large. These URLs point to the Amazon web site.
# ● Ratings
# Contains the book rating information. Ratings (Book-Rating) are either explicit, expressed on a scale from 1-10 (higher values denoting higher appreciation), or implicit, expressed by

# In[774]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import streamlit as st
st.title('Model Deployment: Book Recommender System')


# In[775]:


books=pd.read_csv(r"C:\Users\srima\Desktop\madhu\Books.csv")

# In[776]:


books.head()


# In[777]:


print(books.shape)


# In[778]:


books.describe()


# In[779]:


books.info()


# In[780]:


books.isnull().sum()


# In[781]:


#there is 1-null vakue in the book-author
book_author_NA = pd.isnull(books["Book-Author"])
books[book_author_NA]


# In[782]:


#there are two null rows in the Publisher
books["Book-Author"].fillna("SOPHIE PYOTT", inplace = True)


# In[783]:


book_author_NA


# In[784]:


books.isnull().sum()


# In[785]:


books.loc[187689]


# In[786]:


publisher_NA = pd.isnull(books["Publisher"])
books[publisher_NA]


# In[787]:


books["Publisher"].fillna("NovelBooks, Inc.", limit = 1, inplace = True)


# In[788]:


books.loc[(books['ISBN'] == '193169656X'),'Publisher'] = 'NovelBooks, Inc.'
books.loc[(books['ISBN'] == '1931696993'),'Publisher'] = 'CreateSpace Independent Publishing Platform'


# In[789]:


books.isnull().sum()


# In[790]:


books.loc[128890]


# In[791]:


books.loc[129037]


# In[792]:


#there are three null rows in the Image-URL-L
image_L_NA = pd.isnull(books["Image-URL-L"])
books[image_L_NA]


# In[793]:


books.loc[(books['ISBN'] == '078946697X'),'Image-URL-L'] = 'http://images.amazon.com/images/P/2070426769.0....'
books.loc[(books['ISBN'] == '078946697X'),'Publisher'] = 'DK Children'

books.loc[(books['ISBN'] == '2070426769'),'Image-URL-L'] = 'http://images.amazon.com/images/P/2070426769.0....'
books.loc[(books['ISBN'] == '2070426769'),'Publisher'] = 'Gallimard Education'

books.loc[(books['ISBN'] == '0789466953'),'Image-URL-L'] = 'http://images.amazon.com/images/P/0789466953.0....'
books.loc[(books['ISBN'] == '0789466953'),'Publisher'] = 'DK Children'


# In[794]:


books.loc[209538]


# In[795]:


books.loc[220731]


# In[796]:


books.loc[221678]


# In[797]:


books.isnull().sum()


# In[798]:


print(books.duplicated().sum())


# In[799]:


#hence in the books data are no null values are duplicated data
books.head()


# In[800]:


books


# In[801]:


# getting unique value from 'year_of_publication' in data 
books['Year-Of-Publication'].unique()


# In[802]:


# Extracting mismatch in feature 'year_of_publication', 'publisher', 'book_author', 'book_title'
books[books['Year-Of-Publication'] == 'DK Publishing Inc'] 


# In[803]:


books[books['Year-Of-Publication'] == 'Gallimard']


# In[804]:


books.loc[221678]


# In[805]:


books.loc[221678]


# In[806]:


books.loc[220731]


# In[807]:


def replace_df_value(df, idx, col_name, val):
    df.loc[idx, col_name] = val
    return df


# In[808]:


replace_df_value(books, 220731, 'Book-Title', "Peuple du ciel, suivi de 'Les Bergers")
replace_df_value(books, 220731, 'Book-Author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(books, 220731, 'Year-Of-Publication', 2003)
replace_df_value(books, 220731, 'Publisher', 'Gallimard')


# In[809]:


books.loc[220731]


# In[810]:


replace_df_value(books, 209538, 'Book-Title', 'DK Readers: Creating the X-Men, How It All Began (Level 4: Proficient Readers)')
replace_df_value(books, 209538, 'Book-Author', 'Michael Teitelbaum')
replace_df_value(books, 209538, 'Year-Of-Publication', 2000)
replace_df_value(books, 209538, 'Publisher', 'DK Publishing Inc')


# In[811]:


books.loc[209538]


# In[812]:


replace_df_value(books, 221678, 'Book-Title', "Peuple du ciel, suivi de 'Les Bergers")
replace_df_value(books, 221678, 'Book-Author', 'Jean-Marie Gustave Le ClÃ?Â©zio')
replace_df_value(books, 221678, 'Year-Of-Publication', 2003)
replace_df_value(books, 221678, 'Publisher', 'Gallimard')


# In[813]:


books.loc[221678]


# In[814]:


books.info()


# In[815]:


#all the columns are of the data type objects


# In[816]:


books.describe()


# In[817]:


s=books[books.duplicated('Image-URL-S')]
m=books[books.duplicated('Image-URL-M')]
l=books[books.duplicated('Image-URL-L')]
print("Image-S",s.shape)
print("Image-M",m.shape)
print("Image-L",l.shape)


# In[818]:


y=books[books["Image-URL-S"] == "http://images.amazon.com/images/P/042511774X.01.THUMBZZZ.jpg"]
a=books[books["Image-URL-S"] == "http://images.amazon.com/images/P/038572179X.01.THUMBZZZ.jpg"]
b=books[books["Image-URL-S"] == "http://images.amazon.com/images/P/044651747X.01.THUMBZZZ.jpg"]
y


# In[819]:


a


# In[820]:


b


# In[821]:


#this duplicates is unable to be found because of the upper and lower case differnce in the ISBN numbers


# In[822]:


books['ISBN'] = books['ISBN'].apply(str.upper)


# In[823]:


books[books["Image-URL-S"] == "http://images.amazon.com/images/P/044651747X.01.THUMBZZZ.jpg"]


# In[824]:


books.describe()


# In[825]:


print("the count of the duplicated values",books.duplicated().sum())
print("the shape of the books",books.shape)


# In[826]:


#now we can see the duplicated values and we need to drop the duplicated values


# In[827]:


books=books.drop_duplicates()
books.shape


# In[828]:


books.duplicated().sum()


# In[829]:


books.describe()


# In[830]:


s=books[books.duplicated('Image-URL-S')]
print("the shape of duplicated values",s.shape)
print("the shape of the dataset",books.shape)


# In[831]:


books=books.drop_duplicates(subset=['Image-URL-S'])
books.shape


# In[832]:


books.describe()


# In[833]:


books["Year-Of-Publication"].dtype


# In[834]:


books['Year-Of-Publication'] = books['Year-Of-Publication'].astype(int)


# In[835]:


books["Year-Of-Publication"].dtype


# In[836]:


books.info()


# In[837]:


books["Year-Of-Publication"].describe()


# In[838]:


books = books.rename(columns={'Book-Title': 'Title', 'Book-Author': 'Author','Year-Of-Publication':'Year','Image-URL-S':'Image_URL_S','Image-URL-M':'Image_URL_M','Image-URL-L':'Image_URL_L'})


# In[839]:


books.head()


# In[840]:


books.info()


# In[841]:


#here in the year column the min year is 0 and the maximum year is 2050
x=books.Year.unique()
x.sort()


# In[842]:


x


# In[843]:


y = books['Year'].value_counts(ascending=False)
y.head(60)


# In[844]:


#totally 4628 years are missing in the column and hence we are gonna impute it with the mode values
books['Year']=books['Year'].fillna(books['Year'].median())


# In[845]:


y = books['Year'].value_counts(ascending=False)
y.head()


# In[846]:


books.isnull().sum()


# In[847]:


books


# from this book dataset All the missing values and the duplicated preprocessed and also all of the columns are sensible

# ## Rating

# In[848]:


rating=pd.read_csv(r"C:\Users\srima\Desktop\madhu\Ratings.csv")


# In[849]:


rating.head()


# In[850]:


rating.info()


# In[851]:


rating.duplicated().sum()


# In[852]:


rating.isnull().sum()


# In[853]:


rating=rating.rename(columns={"User-ID":"User","Book-Rating":"Rating"})


# In[854]:


rating['ISBN'] = rating['ISBN'].apply(str.upper)


# In[855]:


rating


# In[856]:


rating.duplicated().sum()


# In[857]:


sns.histplot(data = rating["Rating"])


# In[858]:


y = rating['Rating'].value_counts(ascending=False).reset_index()
y.head(12)


# In[859]:


print("books rows ",books.shape)
print("ratings rows",rating.shape)


# In[860]:


# the book rating with "0" indicates that these books are not rated and 
#also we sholud see that does we have the rated books details in books dataset


# In[861]:


rating_new = rating[rating['ISBN'].isin(books['ISBN'])]
print("rating_new dimension",rating_new.shape)
rating_new


# In[862]:


print("rating_new dimension",rating_new.shape)
print("rating dimension",rating.shape)


# In[863]:


x=1149780-1031128
print("the no ratings that we dont have the books data is",x)


# In[864]:


# now we have to find the books that have no ratings i.e) Zero rating=647291


# In[865]:


y = rating_new['Rating'].value_counts(ascending=False).reset_index()
y.head(12)


# the Zero rating are considered to be implicit rating which has no influence in the model                                
# hence we are splitting the dataset into implicit and explicit dataset

# In[866]:


# Explicit Ratings Dataset
exp_rating = rating_new[rating_new['Rating'] != 0]
exp_rating = exp_rating.reset_index(drop = True)
exp_rating.shape


# In[867]:


# Implicit Ratings Dataset
imp_rating = rating_new[rating_new['Rating'] == 0]
imp_rating = imp_rating.reset_index(drop = True)
imp_rating.shape


# In[868]:


# only explicit dataset is to used for recomendation system


# In[869]:


plt.style.use('fivethirtyeight')
plt.figure(figsize=(12, 8))
sns.countplot(data=exp_rating , x='Rating', palette='rocket_r')


# In[870]:


# Create column Rating average 
exp_rating['Avg_Rating']=round(exp_rating.groupby('ISBN')['Rating'].transform('mean'),3)
# Create column Rating sum
exp_rating['Total_No_Of_Users_Rated']=exp_rating.groupby('ISBN')['Rating'].transform('count')


# In[871]:


exp_rating


# ## User

# In[872]:


user=pd.read_csv(r"C:\Users\srima\Desktop\madhu\Users.csv")
user


# In[873]:


def missing_values(user):
    mis_val=user.isnull().sum()
    mis_val_percent=round(user.isnull().mean().mul(100),2)
    mz_table=pd.concat([mis_val,mis_val_percent],axis=1)
    mz_table=mz_table.rename(
    columns={user.index.name:'col_name',0:'Missing Values',1:'% of Total Values'})
    mz_table['Data_type']=user.dtypes
    mz_table=mz_table.sort_values('% of Total Values',ascending=False)
    return mz_table.reset_index()


# In[874]:


missing_values(user)


# Age have around 39% missing values

# #### Age distribution

# In[875]:


user.Age.hist(bins=[0, 10, 20, 30, 40, 50, 100])
plt.title('Age Distribution\n')
plt.xlabel('Age')
plt.ylabel('Count')
plt.show()


# from this it is seen that more active users are in the age range of 20-40

# to check the outliers in the columns

# In[876]:


sns.boxplot(y='Age', data=user)
plt.title('Find outlier data in Age column')


# In[877]:


print(sorted(user.Age.unique()))


# In[878]:


median_age = user['Age'].median()
median_age


# In[879]:


# Replacing all null values with median
user['Age'] = user['Age'].fillna(median_age)


# In[880]:


user['Age'].isnull().sum()


# Now there are no null values

# In[881]:


def age_group(age):
    
    if age<15:
        x='Children'
    elif age>=15 and age<35:
        x='Youth'
    elif age>=35 and age<65:
        x='Adults'
    else:
        x='Senior Citizens'
    return x


# In[882]:


user['Age_group']=user['Age'].apply(lambda x: age_group(x))


# In[883]:


plt.figure(figsize=(15,7))
sns.countplot(y='Age_group',data=user)
plt.title('Age Distribution')


# In[884]:


user[user.duplicated()]


# In[885]:


item_counts = user["Location"].value_counts()
print(item_counts)


# In[886]:


user.Location.nunique()


# In[887]:


for i in user:
    user['Country']=user.Location.str.extract(r'\,+\s?(\w*\s?\w*)\"*$') 


# In[888]:


#drop location column
user.drop('Location',axis=1,inplace=True)


# In[889]:


user.isnull().sum()


# In[890]:


user['Country']=user['Country'].astype('str')


# In[891]:


a=list(user.Country.unique())
a=set(a)
a = [x for x in a if x is not None]
a.sort()

print(a)


# In[892]:


user['Country'].replace(['','01776','02458','19104','23232','30064','85021','87510','alachua','america','austria','autralia','cananda','geermany','italia','united kindgonm','united sates','united staes','united state','united states','us'],
                               ['other','usa','usa','usa','usa','usa','usa','usa','usa','usa','australia','australia','canada','germany','italy','united kingdom','usa','usa','usa','usa','usa'],inplace=True)


# many of the data has misspelling values 

# In[893]:


plt.figure(figsize=(15,7))
sns.countplot(y='Country',data=user,order=pd.value_counts(user['Country']).iloc[:10].index)
plt.title('Count of users Country wise')


# In[894]:


user=user.rename(columns={"User-ID":"User"})
user.head()


# In[895]:


user.isnull().sum()


# ## EDA

# In[896]:


plt.figure(figsize=(20,15))
sns.countplot(y="Author", data=books, order=books['Author'].value_counts().index[0:50], palette='deep')
plt.title("Top 50 Authors with most published number of books")


# In[897]:


plt.figure(figsize=(20,15))
sns.countplot(y="Publisher", palette = 'pastel', data=books, order=books['Publisher'].value_counts().index[0:50])
plt.title("Top 50 Publishers with number of books published")


# In[898]:


yr = books['Year'].value_counts().reset_index()
yr.columns = ['value', 'count']
yr['year'] = yr['value'].astype(str) + ' year '

plt.figure(figsize=(20,15))
plt.title('Top 25 years published number of books')
sns.barplot(data = yr.head(25), x='count', y="year")
print("in this graph the year 2002 is found to be maximum because this value has been imputed as mode")


# In[899]:


plt.figure(figsize=(20,15))
sns.countplot(y="Country", palette = 'pastel', data=user, order=user['Country'].value_counts().index[0:15])
plt.title("Top 15 Country with highest user")


# In[900]:


plt.figure(figsize=(6,2))
sns.countplot(x="Rating", data=exp_rating)
plt.title("Explicit Ratings Count")


# merging of the datasets

# In[901]:


books.head()


# In[902]:


exp_rating.head()


# In[903]:


user.head()


# In[904]:


print("user_dimension  ",user.shape)
print("Rating_dimension",exp_rating.shape)
print("books_dimension ",books.shape)


# In[905]:


df_rec = pd.merge(books, exp_rating, on='ISBN', how='inner')
df_rec = pd.merge(df_rec, user, on='User', how='inner')
df_rec


# In[906]:


df_rec["Year"] = df_rec["Year"].astype(int)


# In[907]:


df_rec.info()


# In[908]:


df_rec.isnull().sum()


# In[909]:


df_rec.duplicated().sum()


# In[910]:


plt.figure(figsize=(20,10))
sns.countplot(y=df_rec["Title"], data=df_rec, order=df_rec['Title'].value_counts().index[0:50])
plt.title("Top 50 books that are most rated")


# In[911]:


rating_count = pd.DataFrame(df_rec.groupby('ISBN')['Rating'].count())
rating_count.sort_values('Rating', ascending=False).head()


# In[912]:


most_rated_books = pd.DataFrame(['0316666343', '0971880107', '0385504209', '0312195516', '0060928336'], index=np.arange(5), columns = ['ISBN'])
most_rated_books_summary = pd.merge(most_rated_books, books, on='ISBN')
most_rated_books_summary


# In[913]:


Final_Dataset=user.copy()
Final_Dataset=pd.merge(Final_Dataset,exp_rating,on='User')
Final_Dataset=pd.merge(Final_Dataset,books,on='ISBN')


# In[914]:


Final_Dataset.head()


# In[915]:


missing_values(Final_Dataset)


# In[916]:


Final_Dataset.shape


# ### Popularity Based recommendation sys
# 

#  Popularity based recommendation system works with the trend. It basically uses the items which are in trend right now. For example, if any book which is usually bought by every new user then there are chances that it may suggest that book

# In[917]:


books = books.rename(columns={'Title':'Book-Title','Author':'Book-Author','Rating':'Book-Rating','Year':'Year-Of-Publication','Image_URL_S':'Image-URL-S','Image_URL_M':'Image-URL-M','Image_URL_L':'Image-URL-L'})


# In[918]:


books


# In[919]:


ratings_with_name = rating.merge(books,on='ISBN')
ratings_with_name.head()


# In[920]:


ratings_with_name.shape


# In[936]:


rating=rating.rename(columns={'User':'User-ID',"Rating":"Book-Rating"})


# In[937]:


rating


# In[938]:


num_rating_df = ratings_with_name.groupby('Book-Title').count()['Rating'].reset_index()
num_rating_df.rename(columns={'Rating':'num_ratings'},inplace=True)
num_rating_df


# In[939]:


avg_rating_df = ratings_with_name.groupby('Book-Title').mean()['Rating'].reset_index()
avg_rating_df.rename(columns={'Rating':'avg_rating'},inplace=True)
avg_rating_df


# In[940]:


popular_df = num_rating_df.merge(avg_rating_df,on='Book-Title')
popular_df


# In[941]:


popular_df = popular_df[popular_df['num_ratings']>=250].sort_values('avg_rating',ascending=False).head(50)


# In[942]:


popular_df = popular_df.merge(books,on='Book-Title').drop_duplicates('Book-Title')[['Book-Title','Book-Author','Image-URL-M','num_ratings','avg_rating']]


# In[943]:


popular_df['Image-URL-M'][0]


# ### Collaborative Filtering Based Recommender System

# In[947]:


x = ratings_with_name.groupby('User').count()['Rating'] > 200
padhe_likhe_users = x[x].index


# In[948]:


filtered_rating = ratings_with_name[ratings_with_name['User'].isin(padhe_likhe_users)]


# In[950]:


y = filtered_rating.groupby('Book-Title').count()['Rating']>=50
famous_books = y[y].index


# In[951]:


final_ratings = filtered_rating[filtered_rating['Book-Title'].isin(famous_books)]


# In[952]:


pt = final_ratings.pivot_table(index='Book-Title',columns='User',values='Rating')


# In[953]:


pt.fillna(0,inplace=True)


# In[954]:


pt


# In[955]:


pt


# In[956]:


from sklearn.metrics.pairwise import cosine_similarity


# In[957]:


similarity_score = cosine_similarity(pt)


# In[958]:


similarity_score.shape


# In[959]:


similarity_score[0]


# In[960]:


def recommend_book(book):
    index = np.where(pt.index==book)[0][0]
    similar_books =sorted(list(enumerate(similarity_score[index])),key=lambda x:x[1],reverse=True)[1:6]
    
    data = []
    for i in similar_books:
        item = []
        temp_df = books[books['Book-Title'] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Title'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Book-Author'].values))
        item.extend(list(temp_df.drop_duplicates('Book-Title')['Image-URL-M'].values))
        
        data.append(item)
    return data


# In[961]:


recommend_book('Angels')


# In[963]:


import pickle
pickle.dump(popular_df,open('popular.pkl','wb'))


# In[964]:


popular_df


# In[970]:


pickle.dump(pt,open("pt.pkl","wb"))
pickle.dump(similarity_score,open("similarity_score.pkl","wb"))
pickle.dump(books,open("books.pkl","wb"))


# In[971]:


books.drop_duplicates('Book-Title')


# In[972]:


books


# In[973]:


pt.index.values


# In[974]:


pt


# In[ ]:




