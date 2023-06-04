import sys
from advertisement.exception import AdvertisementException
from advertisement.pipeline.training_pipeline import start_training_pipeline




if __name__ == "__main__":
     try:
         start_training_pipeline()

     except Exception as e:
          raise AdvertisementException(error_message=e, error_detail=sys)