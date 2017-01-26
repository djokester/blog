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


