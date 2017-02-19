---
layout: post
title: Building A Simple Recommender System With MovieLens DataSet
author: Samriddhi Sinha
---
We will try to build an extremely simple recommender system with the help of just **Pandas**. We will be using the **MovieLens dataset**.

-----
## Introduction 


One of the most common datasets that is available on the internet for building a **Recommender System** is the **MovieLens DataSet**. Do a simple google search and see how many GitHub projects pop up. The data is obtained from the [MovieLens](https://movielens.org/) website during the seven-month period from September 19th, 1997 through April 22nd, 1998.

What is a recommender system then?
In ideology it is a machine learning prototype that learns how users’ choice of products (in this case movies)  vary with the users characteristics and recommends a product accordingly. For example a 15 year old kid is more likely to watch the upcoming The Lego Batman Movie than someone who is say 35 years old. Or maybe girls would prefer to watch The Notebook more than guys.

But simple recommender systems do exist. If I pull out a list of Movies from IMDB along with their ratings then all I need to do is sort the movies first according to ratings and then according to the number of people who saw the movie. This would enable me to give people a generalized recommendation but not a personalised recommendation. I will use two terms users and product extensively throughout this post for comfort. Products are the items to be recommended in this case it is movies. Users are the people the recommendation is being made to.

I would be building the recommender system in Python. There are three types of recommender systems possible.

1. **The Simple Recommender**: Just filter the movies based on their popularity/ratings and we are good to go. All we need to do is load them up on Pandas and sort them.
2. **Recommendation Based on Collaborative Filtering**: The basic difference between this method and the next method is pretty simple. Collaborative Filtering is based on either
   * the similarity in preferences, tastes and choices of two users. It analyses how similar the tastes of one user is to another and makes recommendations on the basis of that.
   * the similarity in between two items. In this case factors like genre come into play.
3. **Recommendation Based on Content Based Filtering**: Content based filtering is based on the users choice of products. If you like the TV Series Game of Thrones thn you might like the series Vikings

## Requirements

1. **SciPy**
2. **NumPy**
3. **matplotlib**
4. **Pandas**

## The Dataset

This is the popular [MovieLens](https://grouplens.org/datasets/movielens/100k/) dataset. It has multiple CSV  files zipped into a folder. We shall be working with these files

1. u.data: A  consolidated data about users and the movie ratings. This is a tab separated list of **user id-item id-rating-timestamp**
2. u.item: This contains information about the item (movies) . This is a tab separated list of **movie id  movie title  release date  video release date  IMDb URL  unknown  Action  Adventure  Animation  Children’s  Comedy  Crime  Documentary  Drama  Fantasy  Film-Noir  Horror  Musical  Mystery  Romance  Sci-Fi  Thriller  War  Western**. The last 19 are genres with values 0 or 1.
3. u.user: Demographic information about the users; this is a tab separated list of **user id-age-gender-occupation-zip code**

## Loading the Dataset

We will load the dataset with Pandas onto Dataframes data, item and user

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#column headers for the dataset
data_cols = ['user id','movie id','rating','timestamp']
item_cols = ['movie id','movie title','release date',
'video release date','IMDb URL','unknown','Action',
'Adventure','Animation','Childrens','Comedy','Crime',
'Documentary','Drama','Fantasy','Film-Noir','Horror',
'Musical','Mystery','Romance ','Sci-Fi','Thriller',
'War' ,'Western']
user_cols = ['user id','age','gender','occupation',
'zip code']

#importing the data files onto dataframes
users = pd.read_csv('Desktop/ml-100k/u.user', sep='|',
names=user_cols, encoding='latin-1')
item = pd.read_csv('Desktop/ml-100k/u.item', sep='|',
names=item_cols, encoding='latin-1')
data = pd.read_csv('Desktop/ml-100k/u.data', sep='\t',
names=data_cols, encoding='latin-1')
```

Let us go and check out the heads of these files

```python 
#printing the head of these dataframes
print(users.head())
print(item.head())
print(data.head())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-15-49-591.png){:class="img-responsive"}

A look at the basic details of these data files

```python
print(users.info())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-15-51-48.png){:class="img-responsive"}

```python
print(item.info())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-15-52-06.png){:class="img-responsive"}

```python
print(data.info())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-15-52-25.png){:class="img-responsive"}


## Creating A Simple Recommendation Engine with Pandas

First we merge the three dataframes into one single dataframe

```python
#Create one data frame from the three
dataset = pd.merge(pd.merge(item, data),users)
print(dataset.head())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-16-01-50.png?w=740){:class="img-responsive"}

Next we use groupby to group the movies by their titles. Then we use the size function to returns the total number of entries under each movie title. This will help us get the number of people who rated the movie/ the number of ratings.
```python
ratings_total = dataset.groupby('movie title').size()
print(ratings_total.head())
```

![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-25-23-57-45.png){:class="img-responsive"}

Next we try to take the mean ratings of each movie using the mean function. First we groupby movie title. From the resulting dataframe we select only the movie title and the rating headers. Then we use the mean function on them.
```python
ratings_mean = (dataset.groupby('movie title'))['movie title','rating'].mean()
print(ratings_mean.head())
```

Now if you check ratings_total then you will find its a Series and not a Data Frame. So we will convert that into a dataframe. In the ratings_mean we will see that the movie title has been converted from a column to an index. So we make that a column again.
```python
#modify the dataframes so that we can merge the two
ratings_total = pd.DataFrame({'movie title':ratings_total.index,
'total ratings': ratings_total.values})
ratings_mean['movie title'] = ratings_mean.index
```

Now we head for the merging part. Now we sort the values by the total rating and this helps us sort the data frame by the number of people who viewed the movie
```python
final = pd.merge(ratings_mean, ratings_total).sort_values(by = 'total ratings',
ascending= False)
print(final.head())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-26-11-52-36.png){:class="img-responsive"}

We need to look at the basic characteristics of the data to determine the minimum cutoff of total ratings. Because its not reliable to recommend a movie with a high mean rating that has been rated by only 10 people.
```python
print(final.describe())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-26-12-01-40.png){:class="img-responsive"}

I see the 75th percentile is at around 80.I decide to set the cutoff at 100. With a bit of slicing I am able to ascertain that the 340th element has a total rating of approximately 100. So next try to cut off the remaining data. Then we sort the new Data frame with respect to the mean ratings. And we are done building the recommender system. Print out the head of the data frame to give the top 5 recommendations.
```python
final = final[:300].sort_values(by = 'rating',
ascending = False)
print(final.head())
```

![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-26-12-13-23.png){:class="img-responsive"}

So there is your Simple Recommender! 

For the source code you can visit this [Repository](https://github.com/djokester/RecommenderSystemMovieLens)

