# setting up spotipy api
username = "223scnieqibmx3optwxzd3zai" #enter Spotify username here (if you used Facebook Login, it should be your Facebook User)
my_playlist_name = "Fund V" #this is one of my playlists, replace this with your own (I would recommend using Discover Weekly, if it is avaliable)


#importing packages
import pandas as pd
import spotipy
import spotipy.oauth2 as oauth2
from spotipy.oauth2 import SpotifyOAuth
import spotipy.util as util
from datetime import datetime
import matplotlib.pyplot as plt

import seaborn as sns

from spotipy.oauth2 import SpotifyClientCredentials

client_id = 'e887105295de4ed2826b0d1d8450a944'
client_secret = 'b935dbc08d62408c82aa2e3b2c9f0fb6'

scope = "user-library-read"
redirect_uri='https://www.google.com'


sp = spotipy.Spotify(auth_manager=SpotifyOAuth(client_id=client_id, client_secret=client_secret,
                                               redirect_uri=redirect_uri,
                                               scope=scope, username=username))

results = sp.current_user_saved_tracks()
for idx, item in enumerate(results['items']):
    track = item['track']
    print(idx, track['artists'][0]['name'], " – ", track['name'])