from flask import Flask, render_template, jsonify, request
import processor
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



@app.route('/chatbot', methods=["GET", "POST"])
def chatbotResponse():

    if request.method == 'POST':
        the_question = request.form['question']
        #logging.info("[app.py] get chatbot response >>>>>")
        response = processor.chatbot_response(the_question)

    return jsonify({"response": response })



if __name__ == '__main__':
    app.run(host='0.0.0.0', port='8888', debug=True)
