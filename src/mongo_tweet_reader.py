from unidecode import unidecode
from pymongo import MongoClient
import scraper_example as screx
import tweet_scraper as twtsc
import tweepy
import pandas as pd


def get_username_list_from_mongo(dbname, collectionname):
    username_list = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        username_list.append(unidecode(document['user']['screen_name']))
    return list(set(username_list))


def get_tweets(dbname, collectionname):
    textlist = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        textlist.append(unidecode(document['text']))
    return textlist

def get_tweets_whole_db(dbname):
    textlist = []
    client = MongoClient()
    db = client[dbname]
    for collection in db.collection_names():
        tab = db[collection].find()
        for document in tab:
            textlist.append(unidecode(document['text']))
    return textlist

def get_full_tweet_info(dbname, collectionname):
    textlist = []
    datelist = []
    client = MongoClient()
    db = client[dbname]
    tab = db[collectionname].find()
    for document in tab:
        textlist.append(unidecode(document['text']))
        datelist.append(document['created_at'])
    result = { 'text':textlist, 'date': datelist}
    return pd.DataFrame(result)


if __name__ == "__main__":
    print("getting total user info")
    username_list_total = set(get_username_list_from_mongo('trumpmillion', 'topictweets'))
    print("getting downloaded user info")
    username_downloaded = set(get_username_list_from_mongo('trumpmillion', 'timelinetweets'))
    print("getting list left to download")
    to_download = list(username_list_total - username_downloaded)
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = screx.access_credentials()
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    twtsc.get_user_tweets_sequentially(api, to_download, 200, 'trumpmillion', 'timelinetweets')
