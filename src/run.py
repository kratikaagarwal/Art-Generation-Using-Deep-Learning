from flask_pack import app

# main driver function
if __name__ == '__main__':
 	#app.run(debug=True)
    # run() method of Flask class runs the application 
    # on the local development server.debug mode on helps to reload webpage when changes made without running app again.
  	app.run(threaded=False)#debug is only for development not on production otherwirse its risky to show errors to all

