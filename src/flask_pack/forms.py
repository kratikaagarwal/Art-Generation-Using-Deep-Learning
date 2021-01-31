from flask_wtf import FlaskForm
from flask_login import current_user
from flask_wtf.file import FileField,FileAllowed
from wtforms import StringField,PasswordField,SubmitField,BooleanField,Field
from wtforms.validators import DataRequired,Length,Email,EqualTo,ValidationError
from flask_pack.models import User

class RegistrationForm(FlaskForm):
	#username field is required and username length should be between 8 and 20 chars
	username=StringField('Username',validators=[DataRequired(),Length(min=8,max=20)])
	email=StringField('Email',validators=[DataRequired(),Email()])
	password= PasswordField('Password',validators=[DataRequired()])
	confirm_password= PasswordField('Confirm Password',validators=[DataRequired(),EqualTo('password')])
	submit=SubmitField('Sign Up')

	def validate_username(self,username):#check if userrname and email is uniquw
		user=User.query.filter_by(username=username.data).first()
		if user:
			raise ValidationError('That username is taken.Please chose another username.')

	def validate_email(self,email):#check if userrname and email is uniquw
		user=User.query.filter_by(email=email.data).first()
		if user:
			raise ValidationError('That email is taken.Please chose another email.')
		
class LoginForm(FlaskForm):
	email=StringField('Email',validators=[DataRequired(),Email()])
	password= PasswordField('Password',validators=[DataRequired()])
	remember=BooleanField('Remember Me')
	submit=SubmitField('Login')

class ResetPasswordForm(FlaskForm):
	email=StringField('Email',validators=[DataRequired(),Email()])
	password= PasswordField('Password',validators=[DataRequired()])
	confirm_password= PasswordField('Confirm Password',validators=[DataRequired(),EqualTo('password')])
	submit=SubmitField('Reset Password')

	def validate_email(self,email):#check if user email exists or not
		user=User.query.filter_by(email=email.data).first()
		if user is None:
			raise ValidationError('There is no account with that email.You must register first')

class UpdateAccountForm(FlaskForm):
	#username field is required and username length should be between 8 and 20 chars
	username=StringField('Username',validators=[DataRequired(),Length(min=8,max=20)])
	email=StringField('Email',validators=[DataRequired(),Email()])
	picture=FileField('Update Profile Picture',validators=[FileAllowed(['jpg','png'])])
	submit=SubmitField('Update')

	def validate_username(self,username):#if username and email is dffren from already exisitng usernames
		if username.data!=current_user.username:#last username ot = to newone
			user=User.query.filter_by(username=username.data).first()
			if user:
				raise ValidationError('That username is taken.Please chose another username.')

	def validate_email(self,email):#check if userrname and email is uniquw
		if email.data!=current_user.email:
			user=User.query.filter_by(email=email.data).first()
			if user:
				raise ValidationError('That email is taken.Please chose another email.')

class ImageStyleForm(FlaskForm):
	content=FileField('Content',validators=[DataRequired(),FileAllowed(['jpg','png'])])
	style=FileField('Style',validators=[DataRequired(),FileAllowed(['jpg','png'])])
	submit=SubmitField('MERGE')

class VideoStyleForm(FlaskForm):
	input_video=FileField('Content',validators=[DataRequired(),FileAllowed(['mp4','avi'])])
	submit=SubmitField('MERGE')
	