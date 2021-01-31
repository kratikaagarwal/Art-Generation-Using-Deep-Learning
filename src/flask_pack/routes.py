import os
import secrets
from flask_pack.main import receive
from flask_pack.main1 import process
from PIL import Image#to resize images
from flask import render_template,flash,redirect,url_for,request
from flask_pack import app,db,bcrypt
from flask_pack.models import User
from flask_pack.forms import RegistrationForm,LoginForm,ResetPasswordForm,UpdateAccountForm,ImageStyleForm,VideoStyleForm
from flask_login import login_user,current_user,logout_user,login_required
# The route() function of the Flask class is a decorator, which tells the application which URL should call  the associated function.
from flask_mail import Message

from werkzeug.utils import secure_filename
import urllib.request
@app.route('/')
@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/about')
def about():
 	return render_template('about.html')  

@app.route('/contact')
def contact():
 	return render_template('contact.html') 

@app.route('/service')
def service():
 	return render_template('service.html') 

@app.route('/gallery')
def gallery():
 	return render_template('gallery.html')  

@app.route('/output1')
def output1():
 	return render_template('output1.html') 
 	
@app.route('/feedback')
def feedback():
 	return render_template('feedback.html')   	 	

@app.route('/register',methods=['GET','POST'])
def register():
	if current_user.is_authenticated:#if user larady logged in and tries to access login page redirect it to service page
		return redirect(url_for('service'))
	form=RegistrationForm()#instance of registration form to pass it down in form=form
	if form.validate_on_submit():#if form validate successfully then display flash msg and redirect to hom epage
		hashed_pw=bcrypt.generate_password_hash(form.password.data).decode('utf-8')#decode is used to get string instaed of bytes
		user=User(username=form.username.data,email=form.email.data,password=hashed_pw)
		db.session.add(user)
		db.session.commit()
		flash('Account Created! You can now Login..','success')#flash msg is a one time alert msg
		return redirect(url_for('login'))
	return render_template('register.html',title='Register',form=form)

@app.route('/login',methods=['GET','POST'])
def login():
	if current_user.is_authenticated:
		return redirect(url_for('service'))
	form=LoginForm()
	if form.validate_on_submit():
		user=User.query.filter_by(email=form.email.data).first()
		if user and bcrypt.check_password_hash(user.password,form.password.data):#check user exitst and password entered is valid
			login_user(user,remember=form.remember.data)
			next_page=request.args.get('next') #if login page is opened when user accessed acount route then ater logging in next page 
			#that should be opened should be my account and not services page
			flash('Logged in Successfully','success')
			#so if next parameteris not none return next page else return service page
			return redirect(next_page) if next_page else redirect(url_for('service'))
		else:
			flash('Login Unsuccessful! Please check email and password','danger')	
			return redirect(url_for('login'))
	return render_template('login.html',title='Login',form=form)


@app.route('/logout')
def logout():
	logout_user()
	return redirect(url_for('home'))

@app.route("/reset_token",methods=['GET','POST'])#actually reset password with token generate dfor themusing itsdangerous serilaixer
def reset_token():
	if current_user.is_authenticated: #if user is already login then no need to reset password
		return redirect(url_for('home'))
	#user=User.verify_reset_token(token)#verifying user
	#if user is None:# token is invalid or expired then no user id returned
		#flash('That is an invalid or expired token','warning')
		#return redirect(url_for('reset_request'))
	form=ResetPasswordForm()	
	if form.validate_on_submit():#if form validate successfully then display flash msg and redirect to hom epage
		user=User.query.filter_by(email=form.email.data).first()
		hashed_pw=bcrypt.generate_password_hash(form.password.data).decode('utf-8')#decode is used to get string instaed of bytes
		user.password=hashed_pw
		db.session.commit()
		flash('Your Password Has Been Updated! You can now Login..','success')#flash msg is a one time alert msg
		return redirect(url_for('login'))
	return render_template('reset_token.html',title='Reset Password',form=form)	

def save_picture(form_picture):
	#get a random name for picture of user using secrets moduel
	random_hex=secrets.token_hex(8)
	#the image shoud be saved with the same extension user uploaded it
	#os.split return filename without extension and extension
	#f_name,f_ext=os.apth.splittext(form_picture.filename)but we'll not use f_name so use _(unused variable)
	_,f_ext=os.path.splitext(form_picture.filename) 
	picture_fn=random_hex+ f_ext #picture file name with extension
	#image path :app.root_path gives path of app till falsk_app package direcotry then
	picture_path=os.path.join(app.root_path,'static/profile_pics',picture_fn)
	#before saving image resize it it will save space on file system and speed up our website
	output_size=(125,125)
	i=Image.open(form_picture)#newimage created
	i.thumbnail(output_size)#resized i
	#save picture in db
	i.save(picture_path)#save resized image
	return picture_fn
	

@app.route('/account',methods=['GET','POST'])
@login_required
def account():
	form=UpdateAccountForm()
	if form.validate_on_submit():
		if form.picture.data:#picture is not mandaotry so check if picture data is there 
			picture_file=save_picture(form.picture.data)
			current_user.image_file=picture_file
		current_user.username=form.username.data
		current_user.email=form.email.data
		db.session.commit()
		flash('Your Account Has Been Updated','success')
		return redirect(url_for('account'))
	elif request.method=='GET':#placeholder has curent user data
		form.username.data=current_user.username
		form.email.data=current_user.email
	image_file=current_user.image_file
	return render_template('account.html',image_file=image_file,form=form)

@app.route('/output')
def output():
	return render_template('output.html')

@app.route('/nstimage',methods=['GET','POST'])
@login_required
def nstimage():
	form=ImageStyleForm()
	if request.method=='POST' and form.validate_on_submit():
		file1=form.content.data
		file2=form.style.data
		filename1=secure_filename(file1.filename)
		filename2=secure_filename(file2.filename)
		file1.save(os.path.join(app.root_path,'static/nstimages',filename1))
		file2.save(os.path.join(app.root_path,'static/nstimages',filename2))
		rname=receive(filename1,filename2)
		return redirect(url_for('output'))
	return render_template('nstimage.html',form=form)

@app.route('/nstvideo',methods=['GET','POST'])
@login_required
def nstvideo():
	form=VideoStyleForm()
	if form.validate_on_submit():
		return redirect(url_for('output1'))
	return render_template('nstvideo.html',form=form)

# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response
    