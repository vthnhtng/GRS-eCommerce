import pymongo
import csv


client = pymongo.MongoClient("mongodb://localhost:27017/")
db_name = 'mydb'
collection_name = 'products'


try:
    # Get the database object
    db = client[db_name]

    # Create the collection
    collection = db[collection_name]

    # Open the CSV file
    with open('../database_files/database_meta.csv', mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        next(csv_reader)
        # Iterate over each row in the CSV file
        count = 0
        for row in csv_reader:
            count += 1
            product_dict = {
                'asin': row[6],
                'title': row[1],
                'avg_rating': row[2],
                'rating_number': row[3],
                'price': row[4],
                'category': row[7],
                'image_links': row[8],
                'store': row[5],
                'description': row[9]
            }

            result = collection.insert_one(product_dict)
            if result.inserted_id:
                print(f"{count}. Inserted {result.inserted_id}")
            else:
                print("An error occurred while inserting the item.")

except Exception as e:
    print("Connection failed:", e)

finally:
    client.close()
