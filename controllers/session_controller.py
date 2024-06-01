from models.user import User
from flask import redirect, url_for, flash


COLLECTION = 'users'
RATINGS_COLLECTION = 'ratings'
def create(session, username, pwd, db):
    User.db_collection = db[COLLECTION]
    User.db_ratings_collection = db[RATINGS_COLLECTION]
    user = User.get_user_by_id(username)

    if user and pwd == '123456':
        if pwd :
            session['user_id'] = username
            return redirect(url_for('get_categories'))
    
    return redirect(url_for('login'))

def new(arg):
    pass


def index(session):
    return User.get_user_by_id(session.get('user_id'))

def delete(session):
    session.pop('user_id', None)
    return redirect(url_for('get_categories'))



