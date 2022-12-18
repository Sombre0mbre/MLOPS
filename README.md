For information we use the following dataset for our training : https://www.kaggle.com/datasets/pranavraikokte/covid19-image-dataset

## SETUP
docker build -t flask-app .

## RUN
docker run -d -p 5001:5001 flask-app

go to http://localhost:5001
 
Then you can submit chest xray to predict Covid, Viral Pneumonia or if it is normal.

You can also re train the model with a dataset with the same architecture as the one mentionned before
