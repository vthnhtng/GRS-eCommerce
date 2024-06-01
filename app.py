from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify

from controllers import product_controller
from controllers import session_controller
from views import product_view, session_view

from pymongo import MongoClient

app = Flask(__name__)
app.secret_key = 'f97c007d2c642eb1e794bb5e03edfd39'
# MongoDB connection
client = MongoClient("mongodb://localhost:27017/")
db = client["mydb"]


@app.route('/', defaults={'page': 1}, methods=['GET'])
@app.route('/categories/', defaults={'page': 1},  methods=['GET'])
@app.route('/<int:page>', methods=['GET'])
@app.route('/categories/<int:page>',  methods=['GET'])
def get_categories(page):
    categories, thumb_images, page_number = product_controller.get_categories(None, page, db)
    current_url = request.url
    return product_view.categories(categories, thumb_images, page_number, current_url)

@app.route('/stores/', defaults={'page': 1},  methods=['GET'])
@app.route('/stores/<int:page>',  methods=['GET'])
def get_stores(page):
    stores, thumb_images, page_number = product_controller.get_stores(None, page, db)
    current_url = request.url
    return product_view.stores(stores, thumb_images, page_number, current_url)


@app.route('/products/<string:asin>', methods=['GET'])
def get_product_detail(asin):
    product = product_controller.show(asin, db)

    if product:
        return product_view.product_detail(product)
    else:
        return jsonify({'error': 'Product not found'}), 404

@app.route('/category/<string:category>/', defaults={'page': 1}, methods=['GET'])
@app.route('/category/<string:category>/<int:page>', methods=['GET'])
def get_products_by_category(category, page):
    user = session_controller.index(session)
    products, page_number = product_controller.get_products_by_query(
        query_statement=category, query_type="category", page=page, db=db)

    current_url = request.url
    if products:
        return product_view.list_product(products, page_number, current_url, user)
    else:
        return product_view.list_product([], 1, current_url, user)

@app.route('/store/<string:store>/', defaults={'page': 1}, methods=['GET'])
@app.route('/store/<string:store>/<int:page>', methods=['GET'])
def get_products_by_store(store, page):
    user = session_controller.index(session)
    products, page_number = product_controller.get_products_by_query(
        query_statement=store, query_type="store", page=page, db=db)

    current_url = request.url
    if products:
        return product_view.list_product(products, page_number, current_url, user)
    else:
        return product_view.list_product([], 1, current_url, user)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        return session_controller.create(session, username, password, db)
    return session_view.login()
    
@app.route('/logout')
def logout():
    return session_controller.delete(session)


@app.route('/search/<string:keyword>/', defaults={'page': 1}, methods=['GET'])
@app.route('/search/<string:keyword>/<int:page>', methods=['GET'])
def search(keyword, page):
    user = session_controller.index(session)
    products, page_number = product_controller.get_products_by_query(
        query_statement=keyword, query_type="search", page=page, db=db)

    current_url = request.url
    if products:
        return product_view.list_product(products, page_number, current_url, user)
    else:
        return product_view.list_product([], 1, current_url, user)
    

    
@app.route('/search_categories/<string:keyword>/', defaults={'page': 1}, methods=['GET'])
@app.route('/search_categories/<string:keyword>/<int:page>', methods=['GET'])
def search_categories(keyword, page):
    categories, thumb_images, page_number = product_controller.get_categories(keyword, page, db)
    current_url = request.url
    return product_view.categories(categories, thumb_images, page_number, current_url)

@app.route('/search_stores/<string:keyword>/', defaults={'page': 1}, methods=['GET'])
@app.route('/search_stores/<string:keyword>/<int:page>', methods=['GET'])
def search_stores(keyword, page):
    stores, thumb_images, page_number = product_controller.get_stores(keyword, page, db)
    current_url = request.url
    return product_view.stores(stores, thumb_images, page_number, current_url)


@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404



if __name__ == '__main__':
    app.run(debug=True)
