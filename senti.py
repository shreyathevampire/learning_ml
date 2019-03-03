import tweepy
from textblob import TextBlob

consumer_key = '1M6JYiIcuUzpSzOKq5UZ4Ln8B'
consumer_secret_key = '8L1kYRaaUpZ3eAIQAuG5UVOCFiGcwIqLs1aFXDnBhhU7OkXMh6'

access_token = '718123737833287680-fxbgBH2nhec5zugRHOc7KGCnSENDz4I'
access_secret_key = 'POI4NjtDrBW3cTmhPLaAgwTPC9cZX4oEFApIW3EuQnrJO'

auth = tweepy.OAuthHandler(consumer_key,consumer_secret_key)

auth.set_access_token(access_token,access_secret_key)

apis = tweepy.API(auth)

filename = "/home/user/ml/data.txt"
file = open(filename,"w")

public_tweets = apis.search('Our hero is home. #Abhinandan')
for tweet in public_tweets:
    print (tweet.text)
    file.write(tweet.text + '\n')
    analysis = TextBlob(tweet.text)
    print(analysis.sentiment)
    string = analysis.sentiment
    file.write(str(analysis.sentiment) + '\n\n\n')

file.close()


