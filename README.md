# AI Medical Chatbot

An AI-powered medical chatbot for physical and mental wellbeing that provides diagnosis, solutions, medicine recommendations, and doctor consultation information based on user symptoms.

## Features

- User authentication and secure login
- Multiple input methods:
  - Text input
  - Voice input
  - PDF document upload
  - Word document upload
- Multiple output formats:
  - Text responses
  - Voice responses
- Secure data storage using MongoDB
- Chat history tracking
- Real-time voice recording and playback

## Prerequisites

- Python 3.8 or higher
- MongoDB installed and running locally
- Required Python packages (install using `pip install -r requirements.txt`)

## Installation

1. Clone the repository
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Ensure MongoDB is running locally on the default port (27017)
4. Create the uploads directory:
   ```bash
   mkdir uploads
   ```

## Running the Application

1. Start the Flask application:
   ```bash
   python FinalChat.py
   ```
2. Open a web browser and navigate to:
   ```
   https://localhost:5000
   ```

## Usage

1. Register a new account or login with existing credentials
2. Input your symptoms using any of the following methods:
   - Type text in the input field
   - Click the microphone button to use voice input
   - Upload PDF or Word documents containing symptom descriptions
3. Receive comprehensive responses including:
   - Diagnosis
   - Recommended solutions
   - Suggested medicines
   - Doctor consultation information
4. Listen to voice responses or read the text output
5. View your chat history

## Security Features

- Password hashing using bcrypt
- Secure session management
- HTTPS support
- File upload validation
- MongoDB security best practices

## Note

This is a demonstration project and should not be used as a replacement for professional medical advice. Always consult with healthcare professionals for medical concerns.
