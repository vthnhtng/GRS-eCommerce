from flask import render_template
from models.product import Product


def categories(categories, thumb_images, page_number, current_url):
    return render_template('product/categories.html', categories=categories, thumb_images=thumb_images, page_number=page_number, current_url=current_url)

def stores(stores, thumb_images, page_number, current_url):
    return render_template('product/stores.html', stores=stores, thumb_images=thumb_images, page_number=page_number, current_url=current_url)

def product_detail(product):
    return render_template('product/product_detail.html', product=product)


def list_product(products, page_number, current_url, user):
    return render_template('product/product_list.html', products=products, page_number=page_number, current_url=current_url, user=user)

