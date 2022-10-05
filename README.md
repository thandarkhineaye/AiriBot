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

3. Create models (both English version model and Japanese version model)

For Windows,
```
cd bin
./create_models_{ENV}.bat
```

For Linux,
```
cd bin
./create_models_{ENV}.sh
```
※　If execution permission error occurred, run `chmod +x create_models_{ENV}.sh`.

※　ENV is desired environment (develop/production/staging).

4. Run the APP to create a Flask front end on port 8888 (or any port the app is pointing to)

For Windows,
```
cd bin
./app_{ENV}.bat
```

For Linux,
```
cd bin
./app_{ENV}.sh
```
※　If execution permission error occurred, run `chmod +x app_{ENV}.sh`.

※　ENV is desired environment (develop/production/staging).

## Project Reference
● >https://github.com/tatiblockchain/python-deep-learning-chatbot
