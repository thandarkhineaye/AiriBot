import logging
from typing import List, Dict

import nltk
from janome.analyzer import Analyzer
from janome.tokenfilter import POSStopFilter
from janome.tokenizer import Tokenizer
from nltk.stem import WordNetLemmatizer
from omegaconf import DictConfig


class SentencePreProcessor:
    def __init__(self, conf: DictConfig, language: str = "en"):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = Tokenizer()
        # 読み捨てるトークンの品詞を指定する
        self.token_filters = [POSStopFilter(conf["jp_model"]["token_filters"])]
        self.analyzer = Analyzer(tokenizer=self.tokenizer, token_filters=self.token_filters)
        self.language = language
        self.DEFAULT_LANGUAGE = conf["model"]["language"]["english"]
        self.IGNORE_WORDS = conf["model"]["ignore_words"]

    def clean_up_sentence_en(self, sentence: str) -> List[str]:
        """
        対象英語テキスト文字列を基本形の分かち書きリストに変換する
        Lemmatize word using WordNet's built-in morphy function.
         Returns the input word unchanged if it cannot be found in WordNet- Ignore words.
        :param sentence:
        :return:
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words
                          if word not in self.IGNORE_WORDS]
        return sentence_words

    def clean_up_sentence_jp(self, sentence: str) -> List[str]:
        """
        対象日本語テキスト文字列を基本形の分かち書きリストに変換する
        :param sentence:
        :return:
        """
        tokens = self.analyzer.analyze(sentence)
        sentence_words = [t.base_form for t in tokens]
        return sentence_words

    def clean_up_sentence(self, sentence: str) -> List[str]:
        """
        対象テキスト文字列を基本形の分かち書きリストに変換する
        :param sentence:
        :return:
        """
        return self.clean_up_sentence_en(sentence) if self.language.lower() == self.DEFAULT_LANGUAGE\
            else self.clean_up_sentence_jp(sentence)

    def tokenize_data(self, intents: Dict[str, list]) -> dict:
        """
        Data Tokenization and Preparation
        :param intents:
        :return:
        """
        logging.info("Data Tokenization and Preparation")
        words = []
        classes = []
        documents = []
        for intent in intents['intents']:
            # 各々の教師文(pattern)について繰り返す
            for pattern in intent['patterns']:
                # 教師文を分かち書きリストにする
                result = self.clean_up_sentence(pattern)
                # 分かち書きにした教師文のリスト
                words.extend(result)
                # リスト化した教師文とラベルの対
                documents.append((result, intent['tag']))
                # 教師文に存在する意図ラベルの目録リスト
                if intent['tag'] not in classes:
                    classes.append(intent['tag'])
        logging.info("Sorting words and classes array")
        words = sorted(list(set(words)))
        classes = sorted(list(set(classes)))
        logging.info(f"{len(documents)} documents, {documents}")
        logging.info(f"{len(classes)} classes, {classes}")
        logging.info(f"{len(words)} unique lemmatized words, {words}")
        return {"words": words, "classes": classes, "documents": documents}
