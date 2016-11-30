import tweepy
import sys
import jsonpickle
import os
import numpy as np
import botornot
from unidecode import unidecode
from pymongo import MongoClient
import time
from datetime import datetime
import mongo_tweet_reader as mtr
import json


def access_credentials():
    with open('../Data/twitter_keys.json') as f:
        data = json.load(f)
    API_KEY = data['API_KEY']
    API_SECRET = data['API_SECRET']
    ACCESS_TOKEN = data['ACCESS_TOKEN']
    ACCESS_TOKEN_SECRET = data['ACCESS_TOKEN_SECRET']
    return API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET


def download_tweets_to_mongo(tweetsPerQry, maxTweets, api, dbname, collection):
    '''
    this function downloads tweets into a mongodb
    searchQuery = this is the search query, it should follow the twitter api query structure
    tweetsPerQry = this is how many tweets you download per query to the website
    maxTweets = this is the maximum number of tweets you want to download
    api = this is your api object, you create this with:
        API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = access_credentials()
        auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
        api = tweepy.API(auth, wait_on_rate_limit=True,
                         wait_on_rate_limit_notify=True)
    dbname = this is the name of the mongodb you will store things in
    collection = this is the name of the collection inside the mongodb you want to store it in
    '''
    client = MongoClient()
    db = client[dbname]
    tab = db[collection]
    tweet_list = []
    username_list = []
    sinceId = None
    max_id = -1L
    tweetCount = 0
    place_id = '96683cc9126741d1'
    print("Downloading max {0} tweets".format(maxTweets))
    while tweetCount < maxTweets:
        try:
            # if (max_id <= 0):
            #     if not sinceId:
            #         new_tweets = api.search(q=searchQuery,
            #                                 count=tweetsPerQry)
            #     else:
            #         new_tweets = api.search(q=searchQuery,
            #                                 count=tweetsPerQry,
            #                                 since_id=sinceId)
            # else:
            #     if not sinceId:
            #         new_tweets = api.search(q=searchQuery,
            #                                 count=tweetsPerQry,
            #                                 max_id=str(max_id - 1))
            #     else:
            #         new_tweets = api.search(q=searchQuery,
            #                                 count=tweetsPerQry,
            #                                 max_id=str(max_id - 1),
            #                                 since_id=sinceId)
            new_tweets = api.search(q='place:%s' % place_id, count = tweetsPerQry, lang='en')
            if not new_tweets:
                print("No more tweets found")
                break
            for tweet in new_tweets:
                tab.insert_one(tweet._json)
                tweet_list.append(tweet._json)
                username_list.append(tweet._json['user']['screen_name'])
            tweetCount += len(new_tweets)
            print("Downloaded {0} tweets".format(tweetCount))
            print("API calls remaining: {}".format(api.rate_limit_status()['resources']['search']['/search/tweets']['remaining']))
            max_id = new_tweets[-1].id
        except:
            print("some error : ")
            break
    print("Downloaded {0} tweets".format(tweetCount))
    return tweet_list, list(set(username_list))


def get_next_reset(api):
    timestamp_int = api.rate_limit_status()['resources']['search']['/search/tweets']['reset']
    date = datetime.fromtimestamp(timestamp_int)
    return date


def check_username_validity(twitter_app_auth, username_list):
    client = MongoClient()
    db = client['country_info']
    tab = db['user_check_results']
    bon = botornot.BotOrNot(**twitter_app_auth)
    total = len(username_list)
    print("We are going to classify {} accounts as real or fake".format(total))
    result_list = []
    for i, username in enumerate(username_list):
        try:
            result = bon.check_account('@{}'.format(username))
            print("We have classified {0} account(s), and we have {1} percent to go".format(i, 1-round(float(i)/total, 2)))
            result_list.append(result)
            tab.insert_one(result)
        except:
            print("{} has protected tweets, we will continue".format(username))
            continue
    return result_list


def get_results_for_top_list(twitter_app_auth):
    top_list = ['CowboyNewsBot', 'Onlucyme', 'macaspacxx', 'AngeloYnax',
                'YnaAngelox', 'lepitens2', 'DXNXXLFXRD', 'Daez17',
                'lepitennicell1', 'lepitennicell69', 'kbdpftcristwo2',
                'JadineTQ', 'kbdpftcristhre', 'lepitennicell2',
                'knsolidified1', 'kbdpftcris9', 'lepitennicell10',
                'kbdpftcris8', 'lepitennicell4', 'kbdpftcris10']
    bon = botornot.BotOrNot(**twitter_app_auth)
    total = len(top_list)
    print("We are going to classify {} accounts as real or fake".format(total))
    result_list = []
    for i, username in enumerate(top_list):
        try:
            result = bon.check_account('@{}'.format(username))
            print("We have classified {0} account(s), and we have {1} percent to go".format(i, 1-round(float(i)/total, 2)))
            result_list.append(result)
            tab.insert_one(result)
        except:
            print("{} has protected tweets, we will continue".format(username))
            result_list.append('failed')
            continue
    return result_list


if __name__ == "__main__":
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = access_credentials()
    auth = tweepy.AppAuthHandler(API_KEY, API_SECRET)
    # auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    searchQuery = 'nigger'  # this is what we're searching for
    # places = api.geo_search(query='Philippines', granularity="country")
    # place_id = places[0].id
    # place_id = 'fb151ef38fa2ac0d'
    # searchQuery = "place:%s" % place_id
    maxTweets = 500  # Some arbitrary large number
    tweetsPerQry = 100
    tweet_list, username_list = download_tweets_to_mongo(searchQuery, tweetsPerQry, maxTweets, api, 'example', 'example')
    username_list = mtr.get_username_list_from_mongo()
    # result_list = check_username_validity(twitter_app_auth, username_list)
    # result_list = get_results_for_top_list(twitter_app_auth)
