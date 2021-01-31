from flask import Flask
from flask_sqlalchemy import SQLAlchemy
#import registrstion and login function from forms.py
from flask_bcrypt import Bcrypt
from flask_login import LoginManager


# Flask constructor takes the name of 
# current module (__name__) as argument.
app = Flask(__name__)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

#setting secret key(set of random characters) for security against forgery attacks or crosssite attacks
#generate secret key as:import secrets ,secrets.token_hex(16)
app.config['SECRET_KEY']='S'
app.config['SQLALCHEMY_DATABASE_URI']='sqlite:///site.db'
db=SQLAlchemy(app)#db instance
bcrypt=Bcrypt(app)#used for hashing of password to prevent hacking
login_manager=LoginManager(app)
login_manager.login_view='login'#if user try to access account route withour being logged in login page opens
login_manager.login_message_category='info'#warns to login before accessing account and category is set to info

from flask_pack import routes