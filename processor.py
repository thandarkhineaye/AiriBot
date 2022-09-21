import random
from typing import List

import numpy as np

from common.file_helper import FileHelper
from common.sentence_preprocessor import SentencePreProcessor


# logging.basicConfig(filename=constant.LOG_FILE_PATH,
#                      format='%(asctime)s %(levelname)-8s %(message)s',
#                      level=logging.DEBUG,
#                      datefmt='%Y-%m-%d %H:%M:%S')
# logging.getLogger('matplotlib.font_manager').disabled = True


class ChatProcessor:
    def __init__(self, language: str):
        self.language = language
        self.file_helper = FileHelper()
        self.model = self.file_helper.load_model(language)
        self.intents = self.file_helper.load_intents(language)
        self.words = self.file_helper.load_words_file(language)
        self.classes = self.file_helper.load_classes_file(language)
        self.sentence_preprocessor = SentencePreProcessor(language)
        self.ERROR_THRESHOLD = 0.25
        self.INVALID_QUESTION = "You must ask the right questions" if self.language == "en" else "うまくお答えできません"

    def bow(self, sentence: str, show_details=True) -> List[int]:
        """
        return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
        :param sentence:
        :param show_details:
        :return:
        """
        # tokenize the pattern
        sentence_words = self.sentence_preprocessor.clean_up_sentence(sentence)
        # bag of words - matrix of N words, vocabulary matrix
        bag = [0]*len(self.words)
        for s in sentence_words:
            for index, word in enumerate(self.words):
                if word == s:
                    # assign 1 if current word is in the vocabulary position
                    bag[index] = 1
                    if show_details:
                        print("found in bag: %s" % word)
        return bag

    def predict_class(self, sentence) -> List[dict]:
        """
        predict the intent for input sentence
        :param sentence:
        :return:
        """
        if self.words is None or self.classes is None or self.model is None or self.intents is None:
            return []
        # filter out predictions below a threshold
        p = self.bow(sentence, show_details=False)
        res = self.model.predict(np.array([p]))[0]
        results = [[i, r] for i, r in enumerate(res) if r > self.ERROR_THRESHOLD]
        # sort by strength of probability
        results.sort(key=lambda x: x[1], reverse=True)
        return_list = []
        for r in results:
            return_list.append({"intent": self.classes[r[0]], "probability": str(r[1])})
        return return_list

    def get_chatbot_response(self, sentence: str) -> str:
        """
        get chatbot's predicted answer for the input sentence
        :param sentence:
        :return:
        """
        predicted_intents = self.predict_class(sentence)
        if len(predicted_intents) == 0:
            return self.INVALID_QUESTION
        tag = predicted_intents[0]['intent']
        for intent in self.intents['intents']:
            if intent['tag'] == tag:
                return random.choice(intent['responses'])
        return self.INVALID_QUESTION
