import boto3
import pandas as pd

aws_id = 'AKIA2N6OE5X4S6JEGC4Y'
aws_key = '5DD1Y8mtEeg5Z3bbCLtOxeeLMSl9WMvKPNq5q4Fa'

s3 = boto3.client(
    's3',
    region_name='us-east-1',
    aws_access_key_id = aws_id,
    aws_secret_access_key = aws_key
    )

s3.upload_file(
    Filename='scooter_report.csv',
    Bucket='my-data-images',
    Key='scooter_report.csv',
    ExtraArgs={'ACL':'public-read'}
    )

df = pd.read_csv('https://my-data-images.s3.amazonaws.com/scooter_report.csv')

rekog = boto3.client(
    'rekognition',
    region_name='us-east-1',
    aws_access_key_id = aws_id,
    aws_secret_access_key = aws_key
    )

comprehend = boto3.client(
    'comprehend',
    region_name='us-east-1',
    aws_access_key_id = aws_id,
    aws_secret_access_key = aws_key
    )

translate = boto3.client(
    'translate',
    region_name='us-east-1',
    aws_access_key_id = aws_id,
    aws_secret_access_key = aws_key
    )

#Translating all descriptions to English
for index, row in df.iterrows():
    desc = df.loc[index, 'public_description']
    if desc != '':
        resp = translate.translate_text(
            Text=desc,
            SourceLanguageCode='auto',
            TargetLanguageCode='en',
            )
        df.loc[index, 'public_description'] = resp['TranslatedText']

#Detect text sentiment
for index, row in df.iterrows():
    desc = df.loc[index, 'public_description']
    if desc != '':
        resp = comprehend.detect_sentiment(
            Text=desc,
            LanguageCode='en',
            )
        df.loc[index, 'sentiment'] = resp['Sentiment']

#Detect scooter in image
df['img_scooter'] = 0
for index, row in df.iterrows():
    image = df.loc[index, 'image']
    response = rekog.detect_labels(
        Image={'S3Object': {'Bucket': 'my-data-images', 'Name': image}}
        )
    for label in response['Labels']:
        if label['Name'] == 'Scooter':
            df.loc[index, 'img_scooter'] =1
            break

pickups = df[((df.img_scooter == 1) & (df.sentiment == 'NEGATIVE'))]

num_pickups = len(pickups)


        
