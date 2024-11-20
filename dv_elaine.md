# Process

## Dictated Text from User

`DV by dolche vida Elaine in Rose Gold size 11`

## Generate Shoe-related JSON from the raw text

The following JSON data object is returned from ChatGPT

```json
{
    "id": 1,
    "user_id": 4,
    "raw_text": "DV by dolche vida Elaine in Rose Gold size 11",
    "status": None,
    "options": {"object_type": "shoes-for-sale"},
    "properties": {},
    "created": "2024-11-19T21:04:37.800041",
    "updated": "2024-11-19T21:04:37.800044",
    "images": []
}
```


## Enhance Shoe Data

Request additional data for the shoes, now that they've been generally identified. The json data is sent with a prompt, which then returns the following data json data object for the shoes - this object, to be saved to the 'shoes' table.

When item is saved to a permanent table, that item should stay connected to its original queue item. So I need to add a field into the shoes column for `queue_id`.

```json
{
    "queue_id": "1",
    "brand": "DV by Dolce Vita",
    "model": "Elaine",
    "size": "11",
    "gender": "Womens",
    "color": "Rose Gold",
    "condition": "New",
    "shoe_type": "Sandals",
    "style": "Strappy Sandals",
    "material": "Synthetic",
    "fit_type": "Standard Fit",
    "pattern": "Solid",
    "occasion": "Casual, Outdoor",
    "season": "Spring, Summer",
    "length": None,
    "upc": "888133729889",
    "msrp": 35.99,
    "asking_price": 29.99,
    "sale_price": None,
    "description": "Stand out in style with the DV by Dolce Vita Elaine sandals in Rose Gold. They are designed to leave a lasting impression wherever you go. They offer a comfortable fit and a stylish design that is perfect for any occasion.",
    "quantity": 1,
    "condition_id": 1000,
    "dispatch_time": 1,
    "return_accepted": False,
    "return_policy": None,
    "shipping_type": "Flat",
    "shipping_cost": 0.0,
    "international_shipping": False,
    "item_location": "Norwich, NY",
    "listing_status": "Active",
    "views": 0,
    "watchers": 0,
    "last_updated": "22024-11-19T21:04:37.800044",
    "created_at": "2024-11-19T21:04:37.800044"
```