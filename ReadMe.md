

```python
# Dependencies
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import tweepy
import time
import seaborn as sns
from datetime import datetime
import itertools as iter
from config import consumer_key, API_secret, access_token, token_secret
```


```python
# Import and Initialize Sentiment Analyzer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
analyzer = SentimentIntensityAnalyzer()
```


```python
# Twitter API Keys
consumer_key = consumer_key
API_secret = API_secret
access_token = access_token
token_secret = token_secret

# Setup Tweepy API Authentication
auth = tweepy.OAuthHandler(consumer_key, API_secret)
auth.set_access_token(access_token, token_secret)
api = tweepy.API(auth, parser=tweepy.parsers.JSONParser())
```


```python
# Target Account
target_users = ["@CNN", "@BBC", "@CBS", "@FoxNews", "@nytimes"]
sentiments = []


for target_user in target_users:
    counter = 1

#loop through 5 pages of tweets - 100 total for each news outlet
    for x in range(5):
        
        public_tweets = api.user_timeline(target_user, page=x+1)
        
        #loop though tweets
        for tweet in public_tweets:
            
            #run vader analysis on each tweet
            compound = analyzer.polarity_scores(tweet["text"])["compound"]
            pos = analyzer.polarity_scores(tweet["text"])["pos"]
            neu = analyzer.polarity_scores(tweet["text"])["neu"]
            neg = analyzer.polarity_scores(tweet["text"])["neg"]
            tweets_ago = counter
            
            sentiments.append({"Date": tweet["created_at"], 
                           "Compound": compound,
                           "Positive": pos,
                           "Negative": neu,
                           "Neutral": neg,
                           "Tweets Ago": counter,
                            "Media Source" : target_user})
            
            counter = counter+1
```


```python
sentiments_pd = pd.DataFrame.from_dict(sentiments)
sentiments_pd.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Compound</th>
      <th>Date</th>
      <th>Media Source</th>
      <th>Negative</th>
      <th>Neutral</th>
      <th>Positive</th>
      <th>Tweets Ago</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.2732</td>
      <td>Mon Apr 09 22:08:33 +0000 2018</td>
      <td>@CNN</td>
      <td>0.913</td>
      <td>0.000</td>
      <td>0.087</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.5267</td>
      <td>Mon Apr 09 22:07:14 +0000 2018</td>
      <td>@CNN</td>
      <td>0.833</td>
      <td>0.000</td>
      <td>0.167</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.8176</td>
      <td>Mon Apr 09 22:04:38 +0000 2018</td>
      <td>@CNN</td>
      <td>0.615</td>
      <td>0.000</td>
      <td>0.385</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0000</td>
      <td>Mon Apr 09 21:54:58 +0000 2018</td>
      <td>@CNN</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>0.000</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-0.5574</td>
      <td>Mon Apr 09 21:39:04 +0000 2018</td>
      <td>@CNN</td>
      <td>0.854</td>
      <td>0.146</td>
      <td>0.000</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>




```python
#create scatterplot of tweet semtiment 
sns.set()
sns.lmplot( x="Tweets Ago", y="Compound", data=sentiments_pd, fit_reg=False, markers="*", 
           size=10, hue="Media Source", legend=True, scatter_kws={"s": 500})
plt.title("Tweet Sentiment over past 100 Tweets")
```




    Text(0.5,1,'Tweet Sentiment over past 100 Tweets')




![png](output_5_1.png)



```python
#create dataframes with each news outlet (used for the aggregate compound sentiment scores in the bar chart)
cnn_sentiments_df = sentiments_pd.loc[sentiments_pd["Media Source"] == "@CNN"]
bbc_sentiments_df = sentiments_pd.loc[sentiments_pd["Media Source"] == "@BBC"]
cbs_sentiments_df = sentiments_pd.loc[sentiments_pd["Media Source"] == "@CBS"]
foxnews_sentiments_df = sentiments_pd.loc[sentiments_pd["Media Source"] == "@FoxNews"]
nytimes_sentiments_df = sentiments_pd.loc[sentiments_pd["Media Source"] == "@nytimes"]
```


```python
#identify aggregate compound sentiment score for each news outlet
cnn_agg = round((cnn_sentiments_df["Compound"].sum()/20),2)   
bbc_agg = round((bbc_sentiments_df["Compound"].sum()/20),2)   
cbs_agg = round((cbs_sentiments_df["Compound"].sum()/20),2)   
nytimes_agg = round((nytimes_sentiments_df["Compound"].sum()/20),2)   
foxnews_agg = round((foxnews_sentiments_df["Compound"].sum()/20),2)
```


```python
#Create bar graph showing compound sentiment scores
sns.set()

x = sentiments_pd["Media Source"].unique()
#x = np.arange(len(y))
#y = [cnn_agg, bbc_agg, cbs_agg, nytimes_agg, foxnews_agg]
y = round((sentiments_pd.groupby(["Media Source"])["Compound"].mean()),2)

plt.title("Sentiment Analysis of Media Tweets (March 2018)")
plt.xlabel("Media Sources")
plt.ylabel("Compound Sentiment Score")

plt.bar(x, y, color=["lightskyblue", "aqua", "violet", "orange", "blue"], alpha=0.5, align="center")
```




    <Container object of 5 artists>




![png](output_8_1.png)

