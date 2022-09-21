from flask import Flask, render_template, jsonify, request

from processor import ChatProcessor

# Log File configuration
# logging.basicConfig(filename=constant.LOG_FILE_PATH,
#                      format='%(asctime)s %(levelname)-8s %(message)s',
#                      level=logging.DEBUG,
#                      datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger('matplotlib.font_manager').disabled = True

LANGUAGE_EN = "en"
LANGUAGE_JP = "jp"
chat_processor_en = ChatProcessor(LANGUAGE_EN)
chat_processor_jp = ChatProcessor(LANGUAGE_JP)

app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chatbot', methods=["POST"])
def chatbotResponse():
    the_question = request.form["question"]
    language = request.form["language"]
    #logging.info("[app.py] get chatbot response >>>>>")
    if language.lower() == LANGUAGE_JP:
        response = chat_processor_jp.get_chatbot_response(the_question)
    else:
        response = chat_processor_en.get_chatbot_response(the_question)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)
