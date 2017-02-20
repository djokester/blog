---
layout: post
title: Building A Recommender System on User-User Collaborative Filtering (MovieLens DataSet)
author: Samriddhi Sinha
---
Previously I built a very simple data set based on just Pandas manipulation. Now I am looking to build a Collaborative Filtering Recommender System based on the similarity of the user.

We can use many similarity models for this purpose like the Pearson, Cosine etc. But we will just stick to the Eucledian Distance model for this one.

For any information about the Dataset and/or Recommender Systems please refer back to the previous post.

-----

## Getting Started

We will load the data sets firsts.
```python
#column headers for the dataset
data_cols = ['user id','movie id','rating','timestamp']
item_cols = ['movie id','movie title','release date','video release date','IMDb URL','unknown','Action','Adventure','Animation','Childrens','Comedy','Crime','Documentary','Drama','Fantasy','Film-Noir','Horror','Musical','Mystery','Romance ','Sci-Fi','Thriller','War' ,'Western']
user_cols = ['user id','age','gender','occupation','zip code']

#importing the data files onto dataframes 
users = pd.read_csv('Desktop/ml-100k/u.user', sep='|', names=user_cols, encoding='latin-1')
item = pd.read_csv('Desktop/ml-100k/u.item', sep='|', names=item_cols, encoding='latin-1')
data = pd.read_csv('Desktop/ml-100k/u.data', sep='\t', names=data_cols, encoding='latin-1')
```
We will use the file **u.data** first as it contains User ID, Movie IDs and Ratings. These three elements are all we need for determining the similarity of the users based on their ratings for a particular movie. I will first sort the DataFrame by User ID and then we are going to split the data-set into a training set and a test set (I just need one user for the training).

```python
utrain = (data.sort_values('user id'))[:99832]
print(utra
print(utrain.tail())
utest = (data.sort_values('user id'))[99833:]
print(utest.head())
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-29-11-50-13.png){:class="img-responsive"}

We convert them to a NumPy Array for ease of iteration!
```python
utrain = utrain.as_matrix(columns = ['user id', 'movie id', 'rating'])
utest = utest.as_matrix(columns = ['user id', 'movie id', 'rating'])
```
Create a **users_list** which is a **list of users** that contains a **list of movies** rated by him. This part is going to greatly compromise on the program time unfortunately!

```python
users_list = []
for i in range(1,943):
    list = []
    for j in range(0,len(utrain)):
        if utrain[j][0] == i:
            list.append(utrain[j])    
        else:
            break
    utrain = utrain[j:]
    users_list.append(list) 
```

## Similarity Scores
Define a Function by the Name of **EucledianScore**. The purpose of the EucledianScore is to measure the similarity between two users based on their ratings given to movies that they have both in common. But what if the users have just one movie in common? In my opinion having more movies in common is a great sign of similarity. So if users have less than 4 movies in common then we assign them a high **EucledianScore**.

```python
def EucledianScore(train_user, test_user):
    sum = 0
    count = 0
    for i in test_user:
        score = 0
        for j in train_user:
            if(int(i[1]) == int(j[1])):
                score= ((float(i[2])-float(j[2]))*(float(i[2])-float(j[2])))
                count= count + 1        
            sum = sum + score
    if(count<4):
        sum = 1000000           
    return(math.sqrt(sum))
```
Now we will iterate over **users_list** and find the similarity of the users to the **test_user** by means of this function and append the **EucledianScore** along with the **User ID** to a separate list **score_list**. We then convert it first to a DataFrame, sort it by the EucledianScore and finally convert it to a NumPy Array **score_matrix** for the ease of iteration.

```python
score_list = []               
for i in range(0,942):
    score_list.append([i+1,EucledianScore(users_list[i], utest)])

score = pd.DataFrame(score_list, columns = ['user id','Eucledian Score'])
score = score.sort_values(by = 'Eucledian Score')
print(score)
score_matrix = score.as_matrix()
```

![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-29-14-13-031.png){:class="img-responsive"}

Now we see that the user with ID 310 has the lowest **Eucledian score** and hence the highest similarity. So now we need to obtain the list of movies that are **not common** between the two users. Make two lists. Get the full list of movies which are there on USER_ID 310. And then the list of common movies. Convert these lists into sets and get the list of movies to be recommended.

```python
user= int(score_matrix[0][0])
common_list = []
full_list = []
for i in utest:
    for j in users_list[user-1]:
        if(int(i[1])== int(j[1])):
            common_list.append(int(j[1]))
        full_list.append(j[1])

common_list = set(common_list)  
full_list = set(full_list)
recommendation = full_list.difference(common_list)
```
## Stringing them Together

Now we need to create a compiled list of the movies along with their mean ratings. Merge the item and data files.Then **groupby** movie titles, select the columns you need and then find the mean ratings of each movie. Then express the dataframe as a NumPy Array.

```python
item_list = (((pd.merge(item,data).sort_values(by = 'movie id')).groupby('movie title')))['movie id', 'movie title', 'rating']
item_list = item_list.mean()
item_list['movie title'] = item_list.index
item_list = item_list.as_matrix()
```
Now we find the movies on **item_list** by IDs from recommendation. Then append them to a separate list.

```python
recommendation_list = []
for i in recommendation:
    recommendation_list.append(item_list[i-1])
    
recommendation = (pd.DataFrame(recommendation_list,columns = ['movie id','mean rating' ,'movie title'])).sort_values(by = 'mean rating', ascending = False)
print(recommendation[['mean rating','movie title']])
```
![image-title-here](https://acodeforthought.files.wordpress.com/2016/12/screenshot-from-2016-12-29-16-41-54.png?w=740){:class="img-responsive"}

Print them out and your recommendations are ready!