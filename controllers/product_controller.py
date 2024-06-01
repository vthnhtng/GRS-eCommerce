from models.product import Product
import json
import pymongo


COLLECTION = 'products'
def show(asin, db):
	# Get product by asin
	Product.db_collection = db[COLLECTION]
	return Product.get_product_by_asin(asin)


def get_products_by_query(query_statement, query_type, page, db):
	Product.db_collection = db[COLLECTION]
	return Product.get_products_by_query(query_statement, query_type, page)

def get_categories(category, page, db):
	Product.db_collection = db[COLLECTION]
	return Product.get_categories(category, page)

def get_stores(store, page, db):
	Product.db_collection = db[COLLECTION]
	return Product.get_stores(store, page)

def search(keyword, page, db):
	Product.db_collection = db[COLLECTION]
	return Product.get_products_by_query(keyword, "search", page)
