# AIRIBOT
![alt text](images/logo/company_logo.png)

## 環境準備（Enviornment Setup）
To get started follow the steps below:

1. Install a virtual environment by runnning the following
```
virtualenv chatbotenv
source chatbotenv/bin/activate
```

2. Install all the required libraries 
```
pip install -r requirements.txt
```

Run the chatbot.py file to create the english model
```
python chatbot.py
```

Run the chatbot_jp.py file to create the japan model
```
python chatbot_jp.py
```

Run the APP to create a Flask front end on port 8888 (or any port the app is pointing to)
```
python app.py
```

## Project Reference
● >https://github.com/tatiblockchain/python-deep-learning-chatbot
