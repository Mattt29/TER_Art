import os

from flask import Flask, render_template

def create_app():

    app = Flask(__name__)

    @app.route('/')
    @app.route('/home')
    def home():
        img_folder = os.path.join('static', 'img', 'carousel')
        img_files = [f for f in os.listdir(img_folder) if f.endswith(('.png', '.jpg', '.jpeg', '.gif'))]
        return render_template('home.html', img_files=img_files)

    @app.route('/ourProject')
    def ourProject():
        return render_template('ourProject.html')

    @app.route('/yourExploration')
    def yourExploration():
        return render_template('yourExploration.html')

    @app.route('/aboutUs')
    def aboutUs():
        return render_template('aboutUs.html')
    
    @app.route('/contact')
    def contact():
        return render_template('contact.html')

    return app