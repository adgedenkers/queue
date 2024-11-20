curl -X POST "https://api.denkers.co/queue/" -H "Content-Type: application/json" -H "x-token: $env:OW_TKN" 
     -d '{
           "user_id": 1,
           "raw_text": "DV Elaine in Rose Gold size 11",
           "options": {
             "flag1": true,
             "setting2": "value"
           },
           "properties": {
             "required_key": "required_value"
           },
           "images": [
             {
               "base64_data": "base64_encoded_image_data_here",
               "filename": "example_image.jpg"
             }
           ]
         }' 



curl -X 'POST' 'https://api.denkers.co/queue/' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'Content-Type: application/json' -d '{ "properties": { "special": "shoes-for-sale" }, "raw_text": "DV Elaine size 11 in Rose Gold", "user_id": 1 }'

# BASE POST COMMAND
curl -X 'POST' 'https://api.denkers.co/queue/' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'Content-Type: application/json'

curl -X POST "https://api.denkers.co/queue/" -H "Content-Type: application/json" -H "x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b" -d '{ "user_id": 1, "raw_text": "DV by dolche vida Elaine in Rose Gold size 11" }'


# WORKING
curl -X 'POST' 'https://api.denkers.co/queue/' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'Content-Type: application/json' -d '{ "user_id": 1, "raw_text": "DV Elaine size 11 in Rose Gold", "options": {"object_type": "shoes-for-sale"}, "properties": {}, "images": [] }'

# this worked b/c we were asking for user_id as a form field or just as a parameter - below we provide it in the header
curl -X 'GET' 'https://api.denkers.co/queue/1?user_id=1' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'user_id: 1' -H 'Content-Type: application/json'
# this is the preferred way
curl -X 'GET' 'https://api.denkers.co/queue/' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'user_id: 1' -H 'Content-Type: application/json'
curl -X 'GET' 'https://api.denkers.co/queue/2' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'user_id: 1' -H 'Content-Type: application/json'


# PS C:\Users\adged> curl -X 'POST' 'https://api.denkers.co/queue/' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'Content-Type: application/json' -d '{ "user_id": 1, "raw_text": "DV Elaine size 11 in Rose Gold", "options": {"object_type": "shoes-for-sale"}, "properties": {}, "images": [] }'
# {"id":1,"status":"success","created":"2024-11-15T18:32:09.605851","image_count":0}


curl -X GET "https://api.denkers.co/shoes/" -H "Content-Type: application/json" -H "x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b" -H "user-id: 1"

curl -X POST "https://api.denkers.co/queue/" -H "Content-Type: application/json" -H "x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b" -d '{ "user_id": 1, "raw_text": "DV by dolche vida Elaine in Rose Gold size 11" }'

# WORKING
curl -X 'GET' 'https://api.denkers.co/queue/2?user_id=1' -H 'accept: application/json' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'user_id: 1' -H 'Content-Type: application/json'

{
  "id":2,
  "user_id":1,
  "raw_text":"DV Elaine size 11 in Rose Gold",
  "status":"pending",
  "options":
  {
    "object_type":"shoes-for-sale"
  },
  "properties":{},
  "created":"2024-11-14T23:27:05.776687",
  "updated":"2024-11-14T23:27:05.776690",
  "images":[]
}


curl -X 'GET' 'https://api.denkers.co/queue/2?user_id=1' -H 'user_id: 1' -H 'x-token: 6b1f9eeb7e3a1f6e3b3f1a2c5a7d9e1b' -H 'accept: application/json' -H 'Content-Type: application/json'

{
  "id":2,
  "user_id":1,
  "raw_text":"DV Elaine size 11 in Rose Gold",
  "status":"pending",
  "options":{"object_type":"shoes-for-sale"},
  "properties":{},
  "created":"2024-11-14T23:27:05.776687",
  "updated":"2024-11-14T23:27:05.776690",
  "images":[]
}