import torch
import pickle
import pymongo
import ast
import math
import random
from .product import Product


def knn_search(query_embedding, node_embeddings, k=12):
    # Compute pairwise Euclidean distances
    distances = torch.cdist(query_embedding.unsqueeze(0), node_embeddings)

    # Sort the distances and get the indices of the K nearest neighbors
    _, indices = torch.topk(distances, k+1, largest=False)
    return indices.squeeze()


class User:
    db_collection = None
    db_ratings_collection = None
    def __init__(self, id, name):
        self.id = id
        self.name = name


    @classmethod
    def get_user_by_id(cls, id):
        try:
            if not cls.db_collection:
                return None
            
            user_record = cls.db_collection.find_one({"user_id": id})

            if not user_record:
                return None
            
            return User(
                user_record['user_id'],
                "User" + user_record['user_id']
            )
            
        except Exception as e:
            raise e
        
    
    def get_recommendations(self, embeddings_path="embeddings/price_niche_rating_store.pkl"):
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)

        asin_list = list(embeddings['asin_list'])
        embeddings = embeddings['embeddings']

        records = self.db_ratings_collection.find({"user_id": self.id})
        recommended_products = []

        for r in records:
            asin = r['asin']
            timestamp = r['timestamp']
            
            rated_products = [(asin, timestamp) for r in records]
            latest_rated_asins = sorted(rated_products, key=lambda item: item[1], reverse=True)[:5]

            for asin, _ in latest_rated_asins:
                query_index = asin_list.index(asin)
                query_embedding = embeddings[query_index]

                recommendations = knn_search(query_embedding, embeddings)
                recommended_asins = [asin_list[i] for i in list(recommendations)]

                for asin in recommended_asins[1:]:
                    recommended_products.append(Product.get_product_by_asin(asin))

        random.shuffle(recommended_products)
        return recommended_products