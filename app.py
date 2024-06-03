from dotenv import load_dotenv
from flask import Flask, render_template
from flask_wtf import FlaskForm
from wtforms import FileField, SubmitField
from wtforms.validators import DataRequired
import pdfplumber
import os
import _pickle as cPickle
from pathlib import Path
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
import pandas as pd
from IPython.display import Markdown, display, clear_output
from werkzeug.utils import secure_filename
from flask import Flask,render_template,flash, redirect,url_for,session,logging,request,Response,jsonify
from flask_sqlalchemy import SQLAlchemy
import json
import os
import gensim
from gensim.test.utils import datapath, get_tmpfile
from gensim.models import KeyedVectors
import re
import gensim
from gensim.parsing.preprocessing import remove_stopwords
from gensim.models import Word2Vec
import gensim.downloader as api
import pprint
import nltk
from flask.globals import request
from werkzeug.utils import secure_filename
from QA.workers import pdf2text, txt2questions
import time
import nltk
from qa_model import word2vec_drive, glove_drive 
nltk.download('stopwords')
app = Flask(__name__)

basedir = os.path.abspath(os.path.dirname(__file__))
db_path = os.path.join(basedir, 'site.db')
app.config['SECRET_KEY'] = "supersecretkey"
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
db = SQLAlchemy(app)
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(80), nullable=False)

# Create the database schema
with app.app_context():
    db.create_all()

class UploadFileForm(FlaskForm): 
    file = FileField("File", validators=[DataRequired()])
    submit = SubmitField("Upload File") 



@app.route('/')
def index():
    return render_template("index.html")

@app.route("/login",methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form["username"]
        password = request.form["password"]
        
        login = User.query.filter_by(username=username, password=password).first()
        if login is not None:
            return redirect(url_for("index"))
    return render_template("login.html")

@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']

        register = User(username=username, email=email, password=password)
        db.session.add(register)
        db.session.commit()
        return redirect(url_for("login"))
    return render_template("register.html")
@app.route("/logout")
def logout():
    session.pop('user_id', None)
    return redirect(url_for('index'))

@app.route('/delete_account', methods=['POST'])
def delete_account():
    if 'user_id' in session:
        user_id = session['user_id']
        user = User.query.get(user_id)
        if user:
            db.session.delete(user)
            db.session.commit()
            session.pop('user_id', None)
            return redirect(url_for('home'))
    return redirect(url_for('login'))





@app.route('/upload', methods=['GET', 'POST'])
def upload():
    form = UploadFileForm()
    if form.validate_on_submit():
        file = form.file.data
        filename = secure_filename(file.filename)
        unique_filename = f"{int(time.time())}_{filename}"  # Append timestamp to filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(file_path)
        print(f"File saved to {file_path}")  # Debug statement

        # Process the file and generate questions
        file_exten = filename.rsplit('.', 1)[1].lower()
        try:
            uploaded_content = pdf2text(file_path, file_exten)
            print(f'Uploaded Content: {uploaded_content[:500]}...')  # Debug statement

            questions = txt2questions(uploaded_content)
            print(f'Generated Questions: {questions}')  # Debug statement
        except Exception as e:
            print(f"Error processing file: {e}")
            flash(f"Error processing file: {e}", "danger")
            return redirect(url_for('upload'))

        # Redirect to quiz page with generated questions
        return render_template('quiz.html', uploaded=True, questions=questions, size=len(questions))
    else:
        print("Form not valid")  # Debug statement
    return render_template('upload.html', form=form)


def pdf_extract(file_name):
    directory = app.config['UPLOAD_FOLDER']
    pdf_txt = ""
    for file in os.listdir(directory):
        filename = os.fsdecode(file)
        if filename == file_name:
            pdf_txt = ''  # Initialize
            with pdfplumber.open(directory + '/' + filename) as pdf:
                for pdf_page in pdf.pages:
                    single_page_text = pdf_page.extract_text()
                    pdf_txt += single_page_text
    return pdf_txt


@app.route('/quiz', methods=['GET', 'POST'])
def quiz():
    """ Handle upload and conversion of file + other stuff """
    UPLOAD_STATUS = False
    questions = dict()

    # Make directory to store uploaded files, if not exists
    if not os.path.isdir('./uploads'):
        os.mkdir('./uploads')

    if request.method == 'POST':
        try:
            # Check if file is uploaded
            if 'file' in request.files:
                # Retrieve file from request
                uploaded_file = request.files['file']
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(uploaded_file.filename))
                file_exten = uploaded_file.filename.rsplit('.', 1)[1].lower()

                # Save uploaded file
                uploaded_file.save(file_path)

                # Get contents of file
                uploaded_content = pdf2text(file_path, file_exten)
                print(f'Uploaded Content: {uploaded_content[:500]}...')  # Debug statement to print the first 500 characters of the uploaded content

                questions = txt2questions(uploaded_content)
                print(f'Generated Questions: {questions}')  # Debug statement to print the generated questions

                # File upload + convert success
                if uploaded_content is not None:
                    UPLOAD_STATUS = True
        except Exception as e:
            print(e)

    return render_template('quiz.html', uploaded=UPLOAD_STATUS, questions=questions, size=len(questions))



@app.route('/result', methods=['POST', 'GET'])
def result():
    correct_q = 0
    for k, v in request.form.items():
        correct_q += 1
    return render_template('result.html', total=5, correct=correct_q)


from flask import request

@app.route('/uploadqa/<model>', methods=['GET', 'POST'])
def uploadqa(model):
    form = UploadFileForm()
    if form.validate_on_submit():
        try:
            file = form.file.data
            filename = secure_filename(file.filename)
            unique_filename = f"{int(time.time())}_{filename}"
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
            file.save(file_path)
            print(f"File saved to {file_path}")  # Debug statement

            # Redirect to the question page with the file path and model
            return redirect(url_for('ask_question', model=model, file_path=unique_filename))
        except Exception as e:
            print(f"Error during file upload or redirection: {e}")
            flash(f"Error during file upload or redirection: {e}", "danger")
    return render_template('uploadqa.html', form=form, model=model)



@app.route('/ask_question/<model>/<file_path>', methods=['GET', 'POST'])
def ask_question(model, file_path):
    answer = ""
    question = request.form.get('question', '')
    full_file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_path)

    if request.method == 'POST' and question:
        if os.path.exists(full_file_path):
            if model == 'word2vec':
                answer = word2vec_drive(full_file_path, question)
                print(answer)
            elif model == 'glove':
                answer = glove_drive(full_file_path, question)
            elif model == 'bert':
                answer = "BERT model not implemented yet"
        else:
            answer = f"Error: The file {file_path} does not exist in the directory {app.config['UPLOAD_FOLDER']}."

    return render_template('qa.html', question=question, answer=answer, model=model, file_path=file_path)



if __name__ == '__main__':
    app.run(debug=True)