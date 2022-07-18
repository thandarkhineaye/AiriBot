# AIRIBOT
![alt text](images/logo/company_logo.png)
## Executive Summary:
・チャットボットに聞いた質問からテキストを分析し、それに関する商品やサービスをおすすめする。自然言語処理を使用してテキストの類似性、感情分析で商品検索し、レコメンデーションシステムを構築する。
\
・When ask the chatbot, analyze the text from the questions from end-users and recommend products and services related to it from Airitech. Use natural language processing and sentiment analysis to search for products by text similarity, build a recommendation system.


## プロジェクト目標 : Project Goal
・レコメンデーションシステムの勉強、チャットボットを元にしてQ/A開発を実施できるように頑張る。\
メンバーも実際のプロジェクトを実施ながら新しい経験を貰い、６ヶ月でデモ成果物（AiriBot）とそれに対して記事作成できるまで頑張る。\
・Create Q/A development system which based on recommendation systems and chatbots.\
All members will also get new experiences while carrying out the actual project and will do their best until they can create a demo product (AiriBot) and an article for it in 6 months.

## 成果物 : Process
● フレームワーク検討し、テストデータ準備、チャットボット環境準備(Review framework, Prepare Data and environment）\
● メンバー募集と開発環境準備(Recruiting other members and preparing the development environment)\
● 各メンバーのタスクを分担(Sharing Tasks)\
● タスク登録と進捗管理(Define Task and Progress Management)

## ビジネスケース / 背景 : Business & background

● 第二ML勉強会の終了後、学んだ事を使って一つの新成果物を作成したい事について相談がきました。\
        After the second ML Study session, We are planned to create a new product by using what we learned.\
● チャットボットから聞いた質問を基にして自社の商品やサービスをおすすめするような開発を試してみたいですが、テキスト検索で分析でき、テストの意味からデータを検索し、結果をもらい、それからチャットボットから商品を返すようなことをテストしたいことです。\
        We would like to try development that recommends our products or services based on the questions from the chatbot. Analyze by searching text and data from the meaning of the questions, get results.After that the chatbot like to be return usefule product.\
● 勉強会で皆の力を纏めて、今回の活動でQ/A研究みたいな製品をチャレンジして見たいと思いました。\
        At Our Study session, we wanted to put together everyone's strengths and take on the challenge of creating products such as Q/A research.\
● 社内プロジェクトとして実施、メンバーたちも新実際経験を貰えるためです。\
        Firstly, we will implemented as an in-house project and also the members can get new actual experiences from this.

## リスク (Risks)
● 参加者メンバーの開発時間が取れない、開発できない可能もあり\ 
        Participant members may not have time to develop or may not be able to develop\
● 開発で時間がかかる場合プロジェクト締め切りに間に合わない可能もあり\
        If development takes time or getting some problems, it may not be possible to finish in project deadline\
● 環境構築準備に時間がかかる場合もあり\
        It may take time to prepare for development environment preparation.
## プロジェクトチーム (Team Member)
● Project Sponsor: Airitech\
● Project Lead: ニャンリンさん(Nyan Linn)、タンダーさん(Thandar)\
● Project Team: タンルインウーさん(Than Lwin Oo)、ネイトゥッさん(Nay Htut), カインカインジョー(Khine Khine Kyaw)\
● Additional Stakeholders:　落合さん(Mr.Ochiai)、カイザーさん(Khine Zar)

## 環境準備（Enviornment Setup）
To get started follow the steps below:

1. Install a virtual environment by runnning the following
```
virtualenv chatbotenv
source chatbotenv/bin/activate
```

2. Install all the required libraries 
```
pip install nltk
pip install numpy
pip install keras
pip install tensorflow
pip install flask
```

Run the chatbot.py file to create the model
```
python chatbot.py
```

Run the APP to create a Flask front end on port 8888 (or any port the app is pointing to)
```
python app.py
```

## Project Reference
● >https://github.com/tatiblockchain/python-deep-learning-chatbot
