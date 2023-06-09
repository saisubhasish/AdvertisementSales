import json
import pymongo
from dask import dataframe as dd
from advertisement.config import mongo_client

DATA_FILE_PATH="D:\\FSDS-iNeuron\\10.Projects-DS\\AdvertisementSales\\advertising.csv"
DATABASE_NAME="advertisementSales"
COLLECTION_NAME="advertisement"

if __name__=="__main__":
    df = dd.read_csv(DATA_FILE_PATH).head(n=200)
    print(f"Rows and columns: {df.shape}")

    #Convert dataframe to json so that we can dump these record in mongo db
    df.reset_index(drop=True,inplace=True)

    # Each record will represent one row
    json_record = list(json.loads(df.T.to_json()).values())
    print(json_record[1])
    #insert converted json record to mongo db
    mongo_client[DATABASE_NAME][COLLECTION_NAME].insert_many(json_record)
