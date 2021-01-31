from itsdangerous import TimedJSONWebSignatureSerializer as Serializer
from flask_pack import db,login_manager,app#app import to get secret key
from flask_login import UserMixin
#we want someone who has user's email can change paasword

@login_manager.user_loader #loginmanager is an extension
def load_user(user_id):
	return User.query.get(int(user_id))#return user of that user id

class User(db.Model,UserMixin):
	id=db.Column(db.Integer,primary_key=True)
	username=db.Column(db.String(20),unique=True,nullable=False)
	email=db.Column(db.String(120),unique=True,nullable=False)
	image_file=db.Column(db.String(20),nullable=False,default='static/images/images.png')#profile image
	password=db.Column(db.String(60),nullable=False)

	def get_reset_token(self,expires_sec=1800):#expire sec is time after which token expires
		s=Serializer(app.config['SECRET_KEY'],expires_sec)#create serializer object
		return s.dumps({'user_id':self.id}).decode('utf-8')#return token combined with user id
		
	@staticmethod#tells python not to expect self as argument
	def verify_reset_token(token):
		s=Serializer(app.config['SECRET_KEY'])
		try:
			user_id=s.loads(token)['user_id']#s.loads(token)loads the token form the token extracr userid
		except:
			return None
		return User.query.get(user_id)	#if token valid return userid	

	def __repr__(self):#on quering the table these 3 fileds will be displayed
		return f"User('{self.username}','{self.email}','{self.image_file}')"


