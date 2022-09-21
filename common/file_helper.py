import json
import os
import pickle
from typing import Optional, Dict, List

from keras.models import load_model
from tensorflow.python.keras.engine.sequential import Sequential


class FileHelper:
    def __init__(self):
        # TODO later constants should be replaced with config values
        self.MODEL_PATH = os.path.join(os.path.dirname(__file__), '../data/models')
        self.INTENTS_PATH = os.path.join(os.path.dirname(__file__), '../data/intents')
        self.MODEL_SAVE_FILE = "chatbot_model.h5"
        self.WORDS_SAVE_FILE = "words.pkl"
        self.CLASSES_SAVE_FILE = "classes.pkl"
        self.INTENTS_FILE = "Common.json"

    def load_pickle_file(self, file_name: str, language: str = "en"):
        """
        load pickle file
        :param file_name:
        :param language:
        :return:
        """
        file_path = os.path.join(self.MODEL_PATH, language.lower(), file_name)
        if not os.path.isfile(file_path):
            return None
        return pickle.load(open(file_path, 'rb'))

    def load_words_file(self, language: str = "en") -> Optional[List[str]]:
        """
        教師データに出現した単語の目録リスト
        :param language:
        :return:
        """
        return self.load_pickle_file(self.WORDS_SAVE_FILE, language)

    def load_classes_file(self, language: str = "en") -> Optional[List[str]]:
        """
        教師データに出現した意図ラベルの目録リスト
        :param language:
        :return:
        """
        return self.load_pickle_file(self.CLASSES_SAVE_FILE, language)

    def load_model(self, language: str = "en") -> Optional[Sequential]:
        """
        学習したモデルを読み込む
        :param language:
        :return:
        """
        file_path = os.path.join(self.MODEL_PATH, language.lower(), self.MODEL_SAVE_FILE)
        if not os.path.isfile(file_path):
            return None
        return load_model(file_path)

    def load_intents(self, language: str = "en") -> Optional[Dict[str, list]]:
        """
        意図ファイルを読み込む
        :param language:
        :return:
        """
        file_path = os.path.join(self.INTENTS_PATH, language, self.INTENTS_FILE)
        if not os.path.isfile(file_path):
            return None
        return json.loads(open(file_path, encoding='utf-8').read())
