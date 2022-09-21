from typing import List

import nltk
from janome.analyzer import Analyzer
from janome.tokenfilter import POSStopFilter
from janome.tokenizer import Tokenizer
from nltk.stem import WordNetLemmatizer


class SentencePreProcessor:
    def __init__(self, language: str):
        self.lemmatizer = WordNetLemmatizer()
        self.tokenizer = Tokenizer()
        self.token_filters = [POSStopFilter(['記号', '助詞', '助動詞'])]
        self.analyzer = Analyzer(tokenizer=self.tokenizer, token_filters=self.token_filters)
        self.language = language
        self.DEFAULT_LANGUAGE = "en"

    def clean_up_sentence_en(self, sentence: str) -> List[str]:
        """
        対象英語テキスト文字列を基本形の分かち書きリストに変換する
        :param sentence:
        :return:
        """
        sentence_words = nltk.word_tokenize(sentence)
        sentence_words = [self.lemmatizer.lemmatize(word.lower()) for word in sentence_words]
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
