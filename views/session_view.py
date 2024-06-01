from flask import render_template
from models.user import User


def login():
    return render_template('session/login.html')

