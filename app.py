from flask import Flask, render_template, jsonify, request
import processor
import processor_jp
import logging
import constant
# Log File configuration
# logging.basicConfig(filename=constant.LOG_FILE_PATH,
#                      format='%(asctime)s %(levelname)-8s %(message)s',
#                      level=logging.DEBUG,
#                      datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger('matplotlib.font_manager').disabled = True

app = Flask(__name__)

app.config['SECRET_KEY'] = 'enter-a-very-secretive-key-3479373'


@app.route('/', methods=["GET", "POST"])
def index():
    return render_template('index.html', **locals())


@app.route('/chatbot', methods=["POST"])
def chatbotResponse():
    the_question = request.form['question']
    language = request.form["language"]
    #logging.info("[app.py] get chatbot response >>>>>")
    if language.lower() == 'jp':
        response = processor_jp.chat(the_question)
    else:
        response = processor.chatbot_response(the_question)

    return jsonify({"response": response})


if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
