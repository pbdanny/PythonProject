from flask import Flask
from flask import render_template
from datetime import datetime
import re

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/about/')
def about():
    return render_template('about.html')

@app.route('/contact/')
def contact():
    return render_template('contact.html')

# Without template
# @app.route('/hello/<name>')
# def hello_there(name):
#     now = datetime.now()
#     formatted_now = now.strftime('%a, %d %b, %Y at %X')
#     match_object = re.match('[a-zA-Z]+', name)

#     if match_object:
#         clean_name = match_object.group(0)
#     else:
#         clean_name = 'Friend'
    
#     content = f'Hello there, {clean_name}! it is {formatted_now}'
#     return content

@app.route('/hello/')
@app.route('/hello/<name>')
def hello_there(name = None):
    return render_template(
        'hello_there.html', 
        name=name,
        date=datetime.now()
    )

@app.route('/api/data')
def get_data():
    return app.send_static_file('data.json')
    