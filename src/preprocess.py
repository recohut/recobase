import os
from pathlib import Path
import pandas as pd

from src.config import *

parent_path = Path(__file__).parent.parent
data_path = os.path.join(parent_path, RAW_DATASET_ROOT_FOLDER, 'ml-1m')
save_path = os.path.join(parent_path, REFINED_DATASET_ROOT_FOLDER, 'ml-1m')

if not Path(save_path).is_dir():
    Path(save_path).mkdir(parents=True)

USER_DATA_FILE = 'users.dat'
MOVIE_DATA_FILE = 'movies.dat'
RATING_DATA_FILE = 'ratings.dat'

AGES = { 1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+" }
OCCUPATIONS = { 0: "other or not specified", 1: "academic/educator", 2: "artist", 3: "clerical/admin",
                4: "college/grad student", 5: "customer service", 6: "doctor/health care",
                7: "executive/managerial", 8: "farmer", 9: "homemaker", 10: "K-12 student", 11: "lawyer",
                12: "programmer", 13: "retired", 14: "sales/marketing", 15: "scientist", 16: "self-employed",
                17: "technician/engineer", 18: "tradesman/craftsman", 19: "unemployed", 20: "writer" }
                
RATINGS_CSV_FILE = 'ratings.csv'
USERS_CSV_FILE = 'users.csv'
MOVIES_CSV_FILE = 'movies.csv'

def preprocess():
    ratings = pd.read_csv(os.path.join(data_path, RATING_DATA_FILE), 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names=['userid', 'movieid', 'rating', 'timestamp'])

    max_userid = ratings['userid'].drop_duplicates().max()
    max_movieid = ratings['movieid'].drop_duplicates().max()

    ratings['user_emb_id'] = ratings['userid'] - 1
    ratings['movie_emb_id'] = ratings['movieid'] - 1

    print(len(ratings), 'ratings loaded')

    ratings.to_csv(os.path.join(save_path, RATINGS_CSV_FILE), 
                sep='\t', 
                header=True, 
                encoding='latin-1', 
                columns=['user_emb_id', 'movie_emb_id', 'rating', 'timestamp'])

    print('Saved to', RATINGS_CSV_FILE)

    users = pd.read_csv(os.path.join(data_path, USER_DATA_FILE), 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names=['userid', 'gender', 'age', 'occupation', 'zipcode'])

    users['age_desc'] = users['age'].apply(lambda x: AGES[x])
    users['occ_desc'] = users['occupation'].apply(lambda x: OCCUPATIONS[x])

    print(len(users), 'descriptions of', max_userid, 'users loaded.')

    users['user_emb_id'] = users['userid'] - 1

    users.to_csv(os.path.join(save_path, USERS_CSV_FILE), 
                sep='\t', 
                header=True, 
                encoding='latin-1',
                columns=['user_emb_id', 'gender', 'age', 'occupation', 'zipcode', 'age_desc', 'occ_desc'])

    print('Saved to', USERS_CSV_FILE)

    movies = pd.read_csv(os.path.join(data_path, MOVIE_DATA_FILE), 
                        sep='::', 
                        engine='python', 
                        encoding='latin-1',
                        names=['movieid', 'title', 'genre'])

    print(len(movies), 'descriptions of', max_movieid, 'movies loaded.')

    movies.to_csv(os.path.join(save_path, MOVIES_CSV_FILE), 
                header=True, 
                columns=['movieid', 'title', 'genre'],
                index=False)

    print('Saved to', MOVIES_CSV_FILE)