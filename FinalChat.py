import os
from flask import Flask, request, jsonify, render_template, send_file, session
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import speech_recognition as sr
import pyttsx3
import PyPDF2
import docx
from datetime import datetime
import json
import torch
from pathlib import Path
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import logging
from werkzeug.utils import secure_filename
import threading

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize text-to-speech and speech-to-text engines
engine = pyttsx3.init()
recognizer = sr.Recognizer()

# Load medical knowledge base
try:
    with open('medical_knowledge.json', 'r') as f:
        medical_knowledge = json.load(f)
    logger.info("Successfully loaded medical knowledge base")
except Exception as e:
    logger.error(f"Failed to load medical knowledge base: {str(e)}")
    raise

# Initialize the transformer model for symptom classification
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased")

ALLOWED_EXTENSIONS = {'pdf', 'docx', 'wav', 'mp3'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_text_input(text):
    try:
        # Tokenize and clean input
        tokens = word_tokenize(text.lower())
        stop_words = set(stopwords.words('english'))
        symptoms = [word for word in tokens if word not in stop_words]
        
        # Match symptoms with medical knowledge base
        matches = []
        for condition in medical_knowledge['conditions']:
            symptom_match_count = sum(1 for symptom in symptoms if any(s in condition['symptoms'] for s in [symptom]))
            if symptom_match_count > 0:
                matches.append((condition, symptom_match_count))
        
        if not matches:
            return {
                'diagnosis': 'Unable to determine specific condition',
                'solution': 'Please consult a healthcare professional for accurate diagnosis',
                'medicines': [],
                'doctor_info': 'Please visit your nearest healthcare facility',
                'confidence': 0
            }
        
        # Sort matches by number of matching symptoms
        matches.sort(key=lambda x: x[1], reverse=True)
        best_match = matches[0][0]
        confidence = matches[0][1] / len(best_match['symptoms'])
        
        return {
            'diagnosis': best_match['condition'],
            'solution': best_match['solution'],
            'medicines': best_match['medicines'],
            'doctor_info': best_match['doctor_type'],
            'confidence': confidence
        }
    except Exception as e:
        logger.error(f"Error processing text input: {str(e)}")
        return None

def text_to_speech(text):
    try:
        speech_file = os.path.join(app.config['UPLOAD_FOLDER'], 'response.mp3')
        engine.save_to_file(text, speech_file)
        engine.runAndWait()
        return speech_file
    except Exception as e:
        logger.error(f"Error converting text to speech: {str(e)}")
        return None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    try:
        text = None
        
        if 'voice' in request.files:
            voice_file = request.files['voice']
            if voice_file and allowed_file(voice_file.filename):
                filename = secure_filename(voice_file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                voice_file.save(filepath)
                
                with sr.AudioFile(filepath) as source:
                    audio = recognizer.record(source)
                    try:
                        text = recognizer.recognize_google(audio)
                    finally:
                        os.remove(filepath)
        
        elif 'file' in request.files:
            file = request.files['file']
            if file and allowed_file(file.filename):
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                if filename.endswith('.pdf'):
                    with open(filepath, 'rb') as f:
                        pdf_reader = PyPDF2.PdfReader(f)
                        text = ''
                        for page in pdf_reader.pages:
                            text += page.extract_text()
                
                elif filename.endswith('.docx'):
                    doc = docx.Document(filepath)
                    text = '\n'.join([paragraph.text for paragraph in doc.paragraphs])
                
                os.remove(filepath)
        
        else:
            data = request.get_json()
            text = data.get('text', '')
        
        if not text:
            return jsonify({'error': 'No input provided'}), 400
        
        # Process the input and get response
        response = process_text_input(text)
        if not response:
            return jsonify({'error': 'Error processing input'}), 500
        
        # Generate voice response
        response_text = (
            f"Diagnosis: {response['diagnosis']}. "
            f"Solution: {response['solution']}. "
            f"Recommended medicines: {', '.join(response['medicines'])}. "
            f"Doctor recommendation: {response['doctor_info']}"
        )
        
        speech_file = text_to_speech(response_text)
        if not speech_file:
            return jsonify({'error': 'Error generating voice response'}), 500
        
        return jsonify({
            'text_response': response,
            'voice_response': os.path.basename(speech_file)
        }), 200
    
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/get_voice_response/<filename>')
def get_voice_response(filename):
    try:
        return send_file(
            os.path.join(app.config['UPLOAD_FOLDER'], filename),
            mimetype='audio/mp3'
        )
    except Exception as e:
        logger.error(f"Error serving voice response: {str(e)}")
        return jsonify({'error': 'Error serving voice response'}), 500

if __name__ == '__main__':
    print("Starting Medical Chatbot server...")
    print("Access the application at: http://127.0.0.1:5001")
    app.run(host='127.0.0.1', port=5001, debug=True)