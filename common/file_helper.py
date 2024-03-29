import json
import logging
import os
import pickle
from typing import Optional, Dict, List, Any

import nltk
from keras.models import load_model
from omegaconf import DictConfig
from tensorflow.python.keras.engine.sequential import Sequential


class FileHelper:
    def __init__(self, conf: DictConfig):
        self.MODEL_PATH = os.path.join(os.path.dirname(__file__), conf["model"]["models_path"])
        self.INTENTS_PATH = os.path.join(os.path.dirname(__file__), conf["model"]["intents_path"])
        self.MODEL_SAVE_FILE = conf["model"]["model_file"]
        self.WORDS_SAVE_FILE = conf["model"]["words_file"]
        self.CLASSES_SAVE_FILE = conf["model"]["classes_file"]
        self.INTENTS_FILE_EXTENSION = conf["model"]["intents_file"]["file_extension"]
        self.INTENTS_FILE_ENCODING = conf["model"]["intents_file"]["file_encoding"]
        self.NLTK_TARGETS = conf["eng_model"]["nltk_download_targets"]

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
        intents = {"intents": []}
        file_path = os.path.join(self.INTENTS_PATH, language)
        files = self.list_files(file_path, self.INTENTS_FILE_EXTENSION)
        for file in files:
            sub_intents = json.loads(open(file, encoding=self.INTENTS_FILE_ENCODING).read())
            if "intents" not in sub_intents:
                logging.info(f"No intents in {file}")
                continue
            intents["intents"].extend(sub_intents["intents"])
        return intents

    @staticmethod
    def list_files(file_path: str, target_file_extension: str = None) -> List[str]:
        """
        list of files inside directory
        :param file_path:
        :param target_file_extension:
        :return:
        """
        result = []
        for (root, dirs, files) in os.walk(file_path):
            for file in files:
                if target_file_extension and not file.lower().endswith(target_file_extension):
                    continue
                result.append(os.path.join(root, file))
        return result

    def dump_pickle_file(self, data: Any, file_name: str, language: str = "en") -> None:
        """
        save pickle file
        :param data:
        :param file_name:
        :param language:
        :return:
        """
        file_path = os.path.join(self.MODEL_PATH, language.lower(), file_name)
        pickle.dump(data, open(file_path, 'wb'))

    def dump_words_file(self, data: List[str], language: str = "en") -> None:
        """
        教師データに出現した単語の目録リストを保存する。
        :param data:
        :param language:
        :return:
        """
        return self.dump_pickle_file(data, self.WORDS_SAVE_FILE, language)

    def dump_classes_file(self, data: List[str], language: str = "en") -> None:
        """
        教師データに出現した意図ラベルの目録リストを保存する。
        :param data:
        :param language:
        :return:
        """
        return self.dump_pickle_file(data, self.CLASSES_SAVE_FILE, language)

    def save_model(self, model: Sequential, data: Any, language: str = "en") -> None:
        """
        学習したモデルを保存する。
        :param model:
        :param data:
        :param language:
        :return:
        """
        file_path = os.path.join(self.MODEL_PATH, language.lower(), self.MODEL_SAVE_FILE)
        return model.save(file_path, data)

    def download_nltk_targets(self) -> None:
        """
        download target nltk data
        :return:
        """
        for target in self.NLTK_TARGETS:
            nltk.download(target)
