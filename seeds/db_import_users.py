import pymongo
import csv


client = pymongo.MongoClient("mongodb://localhost:27017/")
db_name = 'mydb'
collection_name = 'ratings'


try:
    # Get the database object
    db = client[db_name]

    # Create the collection
    collection = db[collection_name]

    # Open the CSV file
    with open('../database_files/database_users.csv', mode='r') as file:
        # Create a CSV reader object
        csv_reader = csv.reader(file)
        next(csv_reader)
        # Iterate over each row in the CSV file
        count = 0
        for row in csv_reader:
            count += 1
            user_dict = {
                'user_id': row[0],
                'password': '123456'
            }

            result = collection.insert_one(user_dict)
            if result.inserted_id:
                print(f"{count}. Inserted {result.inserted_id}")
            else:
                print("An error occurred while inserting the item.")

except Exception as e:
    print("Connection failed:", e)

finally:
    client.close()
