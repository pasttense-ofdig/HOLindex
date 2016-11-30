import tweepy
from pymongo import MongoClient
import scraper_example as screx
import mongo_tweet_reader as mtr
import multiprocessing
import threading
import pandas as pd
import numpy as np


'''
There are two objectives inside this script
1. To get as many tweets as possible for a specific topic, everynight
(and make sure I don't get repeated tweets from the previous night)
2. To get the unique users for these different topics
3. To crawl the pages of these different users and get their tweets
(and make sure I don't download repeated tweets from the previous night)
'''


client = MongoClient()
db_usertweets = client['hate']
tab_usertweets = db_usertweets['timelinetweets']
n_tweets = 200


def get_user_tweets_sequentially(api, username_list, n_tweets, dbname, collection_name):
    client = MongoClient()
    db = client[dbname]
    tab = db[collection_name]
    print("Downloading tweets from {} users".format(len(username_list)))
    for username in username_list:
        try:
            tweetlist = api.user_timeline(screen_name=username, count=n_tweets)
            for tweet in tweetlist:
                tab.insert_one(tweet._json)
            print("Downloaded {} tweets from {}".format(len(tweetlist), username))
            print(api.rate_limit_status()['resources']['search']['/search/tweets']['remaining'])
        except:
            print("{} has protected tweets, we will continue".format(username))
            continue

def get_tweets_from_ids(id_list, api, dbname, collection):
    client = MongoClient()
    db = client[dbname]
    tab = db[collection]
    for sub_list in id_list:
        new_tweets = api.statuses_lookup(sub_list)
        for tweet in new_tweets:
            tab.insert_one(tweet._json)
        print "Downloaded {} tweets.".format(len(new_tweets))
    print "Finished getting all tweets."

def divide_username_list_info_equal_groups(username_list, n_groups):
    avg_list_length = len(username_list)/float(n_groups)
    grouped_list = []
    last = 0.0
    while last < len(username_list):
        grouped_list.append(username_list[int(last):int(last+avg_list_length)])
        last += avg_list_length
    return grouped_list


def get_one_users_tweets(username):
    try:
        tweetlist = api.user_timeline(screen_name=username, count=n_tweets)
        for tweet in tweetlist:
            tab_usertweets.insert_one(tweet._json)
        print("Downloaded {} tweets from {}".format(len(tweetlist), username))
        print(api.rate_limit_status()['resources']['search']['/search/tweets']['remaining'])
    except:
        print("{} has protected tweets, we will continue".format(username))
        print(api.rate_limit_status()['resources']['search']['/search/tweets']['remaining'])


def tweet_scraper_parallel(username_subset_list):
    thread_list = []
    for username in username_subset_list:
        thread_list.append(threading.Thread(target=get_one_users_tweets, args=(username, )))
        thread_list[-1].start()
    for thread in thread_list:
        thread.join()


def get_user_tweets_concurrent(api, username_list, n_groups):
    print("Downloading tweets from {} users".format(len(username_list)))
    grouped_list = divide_username_list_info_equal_groups(username_list, n_groups)
    pool = multiprocessing.Pool(processes=n_groups)
    outputs = pool.map(func=tweet_scraper_parallel, iterable=grouped_list)

def get_location_list(filename,n_miles=50):
    df = pd.read_csv(filename)
    df['n_miles'] = str(n_miles) + 'mi'
    return [(row[1].state,str(row[1].latitude) + ',' + str(row[1].longitude) + ',' + row[1].n_miles) for row in df.iterrows()]

def get_tweet_id_lists(filename):
    df = pd.read_table(filename,sep=',',dtype=np.str_)
    tweet_list = list(df['twitter_id'])
    return [tweet_list[x:x+100] for x in xrange(0,len(tweet_list),100)]

def concatenate_tweet_ids(id_list):
    result = []
    for sub_list in id_list:
        result.append(','.join(str(x) for x in sub_list))
    return result

if __name__ == "__main__":
    API_KEY, API_SECRET, ACCESS_TOKEN, ACCESS_TOKEN_SECRET = screx.access_credentials()
    auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    auth.secure = True
    auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True,
                     wait_on_rate_limit_notify=True)
    # searchQuery = 'you until:2016-11-18'  # this is what we're searching for
    # maxTweets = 10000000  # Some arbitrary large number
    # tweetsPerQry = 100

    # tweet_ids = get_tweet_id_lists('../Data/recent_sightings.sql')
    # # concatenated_tweets = concatenate_tweet_ids(tweet_ids)
    # get_tweets_from_ids(tweet_ids,api,'hate','HateBrain Sightings')

    # location_list = get_location_list('../Data/state_latlon.csv')
    # finished_locations = [u'AR',u'DE',u'CO',u'GA',u'ID',u'IN',u'AL',u'FL',u'AZ',u'IA',u'CA',u'CT',u'DC',u'HI',u'IL']
    # new_locations = [state for state in location_list if state[0] not in finished_locations]
    # for location in new_locations:
    #     tweet_list, username_list = screx.download_tweets_to_mongo(searchQuery, tweetsPerQry, maxTweets, api, location[1], '2016-11-08', location[0])
    #


    # tweet_list, username_list = screx.download_tweets_to_mongo(tweetsPerQry, maxTweets, api, 'USTweets', 'initial_tweets')
    username_list = mtr.get_username_list_from_mongo('USTweets', 'initial_tweets')
    already_done_list = mtr.get_username_list_from_mongo('USTweets','timelinetweets')
    list_to_do = list(set(username_list) - set(already_done_list))
    # auth = tweepy.OAuthHandler(API_KEY, API_SECRET)
    # auth.set_access_token(ACCESS_TOKEN, ACCESS_TOKEN_SECRET)
    # api = tweepy.API(auth, wait_on_rate_limit=True,
    #                  wait_on_rate_limit_notify=True)
    get_user_tweets_sequentially(api, list_to_do, 200, 'USTweets', 'timelinetweets')
    # get_user_tweets_concurrent(api, username_list, 4)
