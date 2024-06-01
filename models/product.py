import torch
import pickle
import pymongo
import ast
import math
import random
import re

def knn_search(query_embedding, node_embeddings, k=12):
    # Compute pairwise Euclidean distances
    distances = torch.cdist(query_embedding.unsqueeze(0), node_embeddings)

    # Sort the distances and get the indices of the K nearest neighbors
    _, indices = torch.topk(distances, k+1, largest=False)
    return indices.squeeze()

def split_and_remove_special_chars(text):
    # Split the text by spaces
    words = text.split()

    # Remove special characters using regex
    cleaned_words = [re.sub(r'[^a-zA-Z0-9]', '', word) for word in words]

    return cleaned_words

class Product:
    db_collection = None

    def __init__(self, asin, title, avg_rating, rating_number, price, category, image_links, store, description):
        self.asin = asin
        self.title = title
        self.avg_rating = avg_rating
        self.rating_number = rating_number
        self.price = price
        self.category = category
        self.image_links = image_links
        self.store = store
        self.description = description

    def get_recommended_products(self, embeddings_path="embeddings/price_niche_rating_store.pkl"):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        asin_list = list(embeddings['asin_list'])
        embeddings = embeddings['embeddings']

        query_index = asin_list.index(self.asin)
        query_embedding = embeddings[query_index]

        recommendations = knn_search(query_embedding, embeddings)
        recommended_asins = []

        for i in list(recommendations):
            recommended_asins.append(asin_list[i])

        recommended_products = []
        for asin in recommended_asins[1:]:
            recommended_products.append(Product.get_product_by_asin(asin))

        return recommended_products

    def to_dict(self):
        return {
            'asin': self.asin,
            'title': self.title,
            'avg_rating': self.avg_rating,
            'rating_number': self.rating_number,
            'price': self.price,
            'category': self.category,
            'image_links': self.image_links,
            'store': self.store,
            'description': self.description
        }

    @classmethod
    def get_product_by_asin(cls, asin):
        try:
            product_record = cls.db_collection.find_one({"asin": asin})
            # Preprocess image links
            product_record['image_links'] = product_record['image_links'].strip(
                "[]").split(", ")
            product_record['image_links'] = [item.replace(
                "'", "") for item in product_record['image_links']]

            # Preprocess description
            if product_record['description'] == "[]":
                product_record['description'] = "No information"
            product_record['description'] = product_record['description'].strip(
                "[]").split(", ")
            product_record['description'] = [item.replace(
                "'", "") for item in product_record['description']]

            price = float(product_record['price'])
            return Product(
                product_record['asin'],
                product_record['title'],
                float(product_record['avg_rating']),
                int(product_record['rating_number']),
                price,
                product_record['category'],
                product_record['image_links'],
                product_record['store'],
                product_record['description'],
            )
        except Exception as e:
            raise e

    @classmethod
    def get_products_by_query(cls, query_statement, query_type, page):
        try:
            if query_type == "category":
                query = {'category': query_statement}
                product_records = cls.db_collection.find(query).skip(8 * (int(page)-1)).limit(8)
            elif query_type == "store":
                query = {'store': query_statement}
                product_records = cls.db_collection.find(query).skip(8 * (int(page)-1)).limit(8)
            else:
                words = split_and_remove_special_chars(query_statement)
                
                max_docs_count = -1
                max_index = -1
                for i in range(0, len(words)):
                    regex = re.compile(words[i], re.IGNORECASE)
                    query = {"title": {"$regex": regex}}
                    docs_count = cls.db_collection.count_documents(query)

                    if docs_count > max_docs_count:
                        max_docs_count = docs_count
                        max_index = i

        
                regex = re.compile(words[max_index], re.IGNORECASE)
                query = {"title": {"$regex": regex}}
                product_records = cls.db_collection.find(query).skip(8 * (int(page)-1)).limit(8)
                

            record_count = cls.db_collection.count_documents(query)
            products = []
            for product_record in product_records:
                # Preprocess image links
                product_record['image_links'] = product_record['image_links'].strip(
                    "[]").split(", ")
                product_record['image_links'] = [item.replace(
                    "'", "") for item in product_record['image_links']]

                # Preprocess description
                if product_record['description'] == "[]":
                    product_record['description'] = "No information"
                product_record['description'] = product_record['description'].strip(
                    "[]").split(", ")
                product_record['description'] = [item.replace(
                    "'", "") for item in product_record['description']]

                price = float(product_record['price'])

                products.append(Product(
                    product_record['asin'],
                    product_record['title'],
                    float(product_record['avg_rating']),
                    int(product_record['rating_number']),
                    price,
                    product_record['category'],
                    product_record['image_links'],
                    product_record['store'],
                    product_record['description'],
                ))
            page_number = math.ceil(record_count / 8)
            return products, page_number
        except Exception as e:
            
            return [], 0
        
    @classmethod
    def get_categories(cls, category, page):
        try:
            if category:
                words = split_and_remove_special_chars(category)     
                max_docs_count = -1
                max_index = -1
                for i in range(0, len(words)):
                    regex = re.compile(words[i], re.IGNORECASE)
                    query = {"title": {"$regex": regex}}
                    docs_count = cls.db_collection.count_documents(query)

                    if docs_count > max_docs_count:
                        max_docs_count = docs_count
                        max_index = i

        
                regex = re.compile(words[max_index], re.IGNORECASE)
                cursor = cls.db_collection.find({"category": {"$regex": regex}})
                categories = [document['category'] for document in cursor]
                categories = list(set(categories))
            else:
                categories = cls.db_collection.distinct("category")

            random.shuffle(categories)
            index = 10 * (int(page) - 1)
            len_categories = len(categories)
            categories = categories[index:index+12]
            thumb_images = []
            for c in categories:
                img_link = cls.db_collection.find_one({"category": c})['image_links'].strip("[]").split(", ")[0].replace("'", '')
                thumb_images.append(img_link)
            page_number = math.ceil(len_categories / 12)
            return categories, thumb_images, page_number
        except Exception as e:
            raise e
        
    @classmethod
    def get_stores(cls, store, page):
        try:
            if store:
                words = split_and_remove_special_chars(store)     
                max_docs_count = -1
                max_index = -1
                for i in range(0, len(words)):
                    regex = re.compile(words[i], re.IGNORECASE)
                    query = {"title": {"$regex": regex}}
                    docs_count = cls.db_collection.count_documents(query)

                    if docs_count > max_docs_count:
                        max_docs_count = docs_count
                        max_index = i

        
                regex = re.compile(words[max_index], re.IGNORECASE)
                cursor = cls.db_collection.find({"store": {"$regex": regex}})
                stores = [document['store'] for document in cursor]
                stores = list(set(stores))
                
            else:
                stores = cls.db_collection.distinct("store")
            random.shuffle(stores)
            index = 10 * (int(page) - 1)
            len_stores = len(stores)
            stores = stores[index:index+12]
            thumb_images = []
            for s in stores:
                img_link = cls.db_collection.find_one({"store": s})['image_links'].strip("[]").split(", ")[0].replace("'", '')
                thumb_images.append(img_link)
            page_number = math.ceil(len_stores / 12)
            return stores, thumb_images, page_number
        except Exception as e:
            raise e