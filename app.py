import logging

from flask import Flask, render_template, jsonify, request

from common.logger import set_log_conf
from processor import ChatProcessor
from common.config import load_config

# Log File configuration
set_log_conf()
config = load_config()

LANGUAGE_EN = config["model"]["language"]["english"]
LANGUAGE_JP = config["model"]["language"]["japan"]
chat_processor_en = ChatProcessor(config, LANGUAGE_EN)
chat_processor_jp = ChatProcessor(config, LANGUAGE_JP)

app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chatbot', methods=["POST"])
def chatbot_response():
    the_question = request.form["question"]
    language = request.form["language"]
    logging.info(f"start get chatbot response. input: {the_question}, lang: {language}")
    if language.lower() == LANGUAGE_JP:
        response = chat_processor_jp.get_chatbot_response(the_question)
    else:
        response = chat_processor_en.get_chatbot_response(the_question)
    logging.info(f"finish get chatbot response. input: {the_question}, lang: {language}, resp: {response}")
    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
