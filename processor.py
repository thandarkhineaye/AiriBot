import logging
import random
from typing import List

import numpy as np

from common.file_helper import FileHelper
from common.sentence_preprocessor import SentencePreProcessor


class ChatProcessor:
    def __init__(self, conf: dict, language: str):
        self.language = language
        self.file_helper = FileHelper(conf)
        self.model = self.file_helper.load_model(language)
        self.intents = self.file_helper.load_intents(language)
        self.words = self.file_helper.load_words_file(language)
        self.classes = self.file_helper.load_classes_file(language)
        self.sentence_preprocessor = SentencePreProcessor(conf, language)
        self.ERROR_THRESHOLD = conf["model"]["error_threshold"]
        self.INVALID_QUESTION = conf["model"]["invalid_question"][self.language.lower()]

    def bow(self, sentence: str, show_details=True) -> List[int]:
        """
        return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        :param sentence:
        :param show_details:
        :return:
        """
        # tokenize the pattern
        sentence_words = self.sentence_preprocessor.clean_up_sentence(sentence)
        logging.info(f"input question clean up. input {sentence}, after clean up: {sentence_words}")
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(self.words)
        for s in sentence_words:
            for index, word in enumerate(self.words):
                if word == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[index] = 1
                    if show_details:
                        logging.info(f"found in bag: {word}")
        return bag

    def predict_class(self, sentence) -> List[dict]:
        """
        predict the intent for input sentence
        :param sentence:
        :return:
        """
        if self.words is None or self.classes is None or self.model is None or self.intents is None:
            logging.warning("Fail to predict the chat sentence.")
            return []
        # filter out predictions below a threshold
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        logging.info(f"Model prediction. input: {sentence}, result: {res}")
        results = [[i, r] for i, r in enumerate(res) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        logging.info(f"Predicted intents. input: {sentence}, prediction: {return_list}")
        return return_list

    def get_chatbot_response(self, sentence: str) -> str:
        """
        get chatbot's predicted answer for the input sentence
        :param sentence:
        :return:
        """
        predicted_intents = self.predict_class(sentence)
        if len(predicted_intents) == 0:
            logging.warning(f"No predicted result.Invalid question. input: {sentence}")
            return self.INVALID_QUESTION
        tag = predicted_intents[0]['intent']
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                result = random.choice(intent['responses'])
                logging.info(f"Matched intent. input {sentence}, intent tag: {intent['tag']}, resp: {result}")
                return result
        logging.warning(f"Invalid question. input: {sentence}")
        return self.INVALID_QUESTION
