# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 16:06:38 2019

@author: vinee
"""

import nltk


text = "This is Andrew's text, isn't it?"

tokenizer = nltk.tokenize.WhitespaceTokenizer()
tokenizer.tokenize(text)

tokenizer = nltk.tokenize.TreebankWordTokenizer()
tokenizer.tokenize(text)

tokenizer = nltk.tokenize.WordPunctTokenizer()
tokenizer.tokenize(text)