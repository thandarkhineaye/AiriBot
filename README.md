# AIRIBOT
![alt text](images/logo/company_logo.png)

## 環境準備（Enviornment Setup）
To get started follow the steps below:

1. Install a virtual environment by runnning the following
```
virtualenv chatbotenv
```
※　if virtualenv is not installed yet, run the following command and after that run the above install command again
```
pip install virtualenv
```
then, activate the virtual environment
```
source chatbotenv/bin/activate
```
※if file not found error occurred, run the following command to give permission for long file paths
```
set-executionpolicy remotesigned
```

2. Install all the required libraries with requirement.txt
```
pip install -r requirements.txt
```

3. Environment Setup

4. Run the chatbot.py file to create English version model
```
python chatbot.py
```

5. Run the chatbot.py file to create English version model
```
python chatbot.py
```

6. Run the chatbot_jp.py file to create the japan model
```
python chatbot_jp.py
```

Run the APP to create a Flask front end on port 8888 (or any port the app is pointing to)
```
python app.py
```

## Project Reference
● >https://github.com/tatiblockchain/python-deep-learning-chatbot
