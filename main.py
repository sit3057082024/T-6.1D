import socket
import argparse
from flask import Flask, request, jsonify
from datetime import datetime
import os
from gradientai import Gradient

token = 'LXkZBp5a1nGCP16xM7QieYKNz2Ns0tOW'
workspace_id = 'b5f958d9-ffd4-41bb-a492-704114e02c8e_workspace'

os.environ['GRADIENT_ACCESS_TOKEN'] = token
os.environ['GRADIENT_WORKSPACE_ID'] = workspace_id
format_question="""
    QUESTION: What is the purpose of a web server in web development?
    OPTION A: To store website files
    OPTION B: To manage database connections
    OPTION C: To handle HTTP requests and responses
    OPTION D: To provide security for the website
    ANS: A
    QUESTION: What is the difference between a static and a dynamic website?
    OPTION A: A static website is created using HTML and CSS, while a dynamic website is created using JavaScript and CSS.
    OPTION B: A static website is created using JavaScript and CSS, while a dynamic website is created using HTML and CSS.
    OPTION C: A static website is updated frequently, while a dynamic website is updated infrequently.
    OPTION D: A static website is designed for a specific device, while a dynamic website is designed for multiple devices.
    ANS: B
    QUESTION: What is the role of a web developer in the development of a website?
    OPTION A: To design the website's layout and user interface
    OPTION B: To write the website's code and implement its functionality
    OPTION C: To test the website for bugs and errors
    OPTION D: To launch the website and make it available to the public.
    ANS: C
    """

app = Flask(__name__)

# Global variable for new_model_adapter
new_model_adapter = None

import re
import json

def process_quiz(quiz_text):
    print(quiz_text)
    questions = []
    pattern = re.compile(r'QUESTION: (.+?)\n(?:OPTION A: (.+?)\n)+(?:OPTION B: (.+?)\n)+(?:OPTION C: (.+?)\n)+(?:OPTION D: (.+?)\n)+ANS: (.+?)', re.DOTALL)
    matches = pattern.findall(quiz_text)
    print(matches)
    for match in matches:
        question = match[0].strip()
        options = match[1].strip(), match[2].strip(), match[3].strip(), match[4].strip()
        correct_ans = match[-1].strip()
        
        question_data = {
            "question": question,
            "options": options,
            "correct_answer": correct_ans
        }
        questions.append(question_data)
    
    return  questions

@app.route('/')
def index():
    return "Welcome to the Flask API!"

def fetchQuizFromLlama(student_topic, new_model_adapter):
    query = f"<s>[INST] <<SYS>>\n Generate a quiz 3 questions to test the ability of the students on the topic provided by them. Generate 4 options after each question where only one of the options is correct. Each question should start with QUES: and options should start with OPTIION: . The format of your response should strictly match this example {format_question} \n<</SYS>>\n\n Student topic {student_topic} [/INST]  </s>"
    response = new_model_adapter.complete(query=query, max_generated_token_count=500).generated_output
    return response

@app.route('/getQuiz', methods=['GET'])
def get_quiz():
    global new_model_adapter
    if new_model_adapter is None:
        return jsonify({'error': 'Model adapter not initialized'}), 500
    student_topic = request.args.get('topic')
    if student_topic is None:
        return jsonify({'error': 'Missing topic parameter'}), 400
    quiz = fetchQuizFromLlama(student_topic, new_model_adapter)
    return jsonify({'quiz': process_quiz(quiz)}), 200

def prepareLlamaBot(name):
    gradient = Gradient()
    base_model = gradient.get_base_model(base_model_slug="llama2-7b-chat")
    global new_model_adapter
    new_model_adapter = base_model.create_model_adapter(name=name)



if __name__ == '__main__':
    default_name = f"Llama_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{socket.gethostname()}"
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=default_name, help='Specify a name')
    args = parser.parse_args()

    port_num = 5000
    print(f"Starting Llama bot with name {args.name}...\n This may take a while.")
    prepareLlamaBot(args.name)
    print(f"App running on port {port_num}")
    app.run(port=port_num)
