#imports
import numpy as np
import pandas as pd

import nltk
from nltk.tokenize import TreebankWordTokenizer
from nltk.tokenize import sent_tokenize
from nltk.stem import LancasterStemmer, PorterStemmer
from nltk.corpus import stopwords
from wordcloud import WordCloud, STOPWORDS

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

import re

import matplotlib.pyplot as plt
import seaborn as sns


def sentiments_desc():
    return {0: 'Negative', 1: 'Neutral', 2: 'Positive', 3: 'Cannot Tell'}


# Remove & Cleaning Functions
def remove_hash(text):
    hash_pat = r'#'
    return re.sub(hash_pat, ' ', text)


def remove_mention(text):
    mention_pat = r'@mention|@[a-z]+'
    return re.sub(mention_pat, ' ', text, flags=re.I)


def remove_short_link(text):
    short_link_pat = r"bit\.ly/[a-z0-9/\-:\.=%;,\+\*())&\$!@\[\]#\?~_\.']*"
    return re.sub(short_link_pat, ' ', text, flags=re.I)


def remove_http_link(text):
    link_permit = r"[a-z0-9/\-:\.=%;,\+\*())&\$!@\[\]#\?~_\.']"
    http_link_pat = r"http[s]?://" + link_permit + "+|//" + link_permit + "+|[\w\.]+\.[a-z]+/" + link_permit + "+"
    return re.sub(http_link_pat, ' ', text, flags=re.I)


def remove_sub_link(text):
    link_pat = r'{link}'
    return re.sub(link_pat, ' ', text, flags=re.I)


def remove_html_char(text):
    html_char_pat = r'&[a-z]+;'
    return re.sub(html_char_pat, ' ', text, flags=re.I)


def remove_date(text):
    pipe = r'|'
    date_pat_mon = str()
    months = [r'January',
              r'February',
              r'March',
              r'April',
              r'May',
              r'June',
              r'July',
              r'August',
              r'September',
              r'October',
              r'November',
              r'December']
    for month in months:
        date_pat_mon = date_pat_mon + month + r' \d\d, \d\d\d\d|'
        date_pat_mon = date_pat_mon + month[:3] + r' \d\d, \d\d\d\d|'

    date_pat_mon = date_pat_mon[:-1]
    #     date_pat_mon
    date_pat = r'\d\d/\d\d/\d\d\d\d|\d\d/\d\d/\d\d' + pipe + \
               r'\d\d\.\d\d\.\d\d\d\d|\d\d\.\d\d\.\d\d' + pipe + \
               r'\d\d-\d\d-\d\d\d\d|\d\d-\d\d-\d\d' + pipe + \
               r'{}'.format(date_pat_mon)
    date_pat = r'{}'.format(date_pat)
    return re.sub(date_pat, ' ', text, flags=re.I)


def remove_short_date(text):
    pipe = r'|'
    short_date_pat = r'[\d]?\d/\d\d[\d\d]?' + pipe + r'[\d]?\d\.\d\d[\d\d]?'
    short_date_pat = r'{}'.format(short_date_pat)
    return re.sub(short_date_pat, ' ', text)


def remove_time(text):
    pipe = r"|"
    time_pat = r"\d\d:\d\d:\d\d[ ]?pm" + pipe + \
               r"\d\d:\d\d:\d\d[ ]?am" + pipe + \
               r"\d\d:\d\d:\d\d" + pipe + \
               r"\d\d:\d\d:\d\d" + pipe + \
               r"[\d]?\d:\d\d[ ]?pm" + pipe + \
               r"[\d]?\d:\d\d[ ]?am" + pipe + \
               r"[\d]?\d:\d\d" + pipe + \
               r"[\d]?\d:\d\d" + pipe + \
               r"[\d]?\d.\d\d[ ]?pm" + pipe + \
               r"[\d]?\d.\d\d[ ]?am" + pipe + \
               r"[\d]?\d.\d\d" + pipe + \
               r"[\d]?\d.\d\d"
    time_pat = r"{}".format(time_pat)
    return re.sub(time_pat, ' ', text, flags=re.I)


def remove_punctuation(text):
    punctuation_pat_s = r'\'s'
    punctuation_pat_t = r'\'t'
    punctuation_pat_d = r'\'d'
    punctuation_pat_ve = r'\'ve'
    punctuation_pat_ll = r'\'ll'
    temp = text
    temp = re.sub(punctuation_pat_s, ' ', temp, flags=re.I)
    temp = re.sub(punctuation_pat_t, 't', temp, flags=re.I)
    temp = re.sub(punctuation_pat_d, ' would', temp, flags=re.I)
    temp = re.sub(punctuation_pat_ve, ' have', temp, flags=re.I)
    temp = re.sub(punctuation_pat_ll, ' will', temp, flags=re.I)
    return temp


def remove_rt(text):
    rt_pat = r'^RT'
    return re.sub(rt_pat, ' ', text)


def remove_emojis(text):
    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               "]+", flags=re.UNICODE)
    return re.sub(emoji_pattern, '', text)


def remove_not_alnum(text):
    """
    remove everything that is not alpha numeric
    """
    only_text = r'[^a-z0-9 &]'
    return re.sub(only_text, ' ', text, flags=re.I)


def replace_text_emoji(text):
    love_pat = r"&lt;3+"
    happy_pat = r":[-]?[\)]+"
    sad_pat = r":[-]?[\(]+"
    playful_pat = r":[-]?[p]+"
    wink_pat = r";[-]?[\)]+"
    straightface_pat = r":[-]?[\|]+"

    # replace text emojis
    temp = text
    temp = re.sub(love_pat, ' love ', temp)
    temp = re.sub(happy_pat, ' smiley ', temp)
    temp = re.sub(sad_pat, ' sad ', temp)
    temp = re.sub(playful_pat, ' playful ', temp, flags=re.I)
    temp = re.sub(wink_pat, ' wink ', temp)
    temp = re.sub(straightface_pat, ' straightface ', temp)
    return temp


def clean_text(text, lower=True):
    # replace links witn {link}
    text = remove_short_link(text)
    text = remove_http_link(text)
    # remove {link}
    text = remove_sub_link(text)
    # hash
    text = remove_hash(text)
    # mention
    text = remove_mention(text)
    # rt
    text = remove_rt(text)
    # html spl chars
    text = remove_html_char(text)
    # time
    text = remove_time(text)
    # date
    text = remove_date(text)
    text = remove_short_date(text)
    # punctuation
    text = remove_punctuation(text)
    # emotjis
    text = remove_emojis(text)
    # remove spl chars
    text = remove_not_alnum(text)
    # return lower
    if lower:
        return text.lower()
    else:
        return text


# Removal Functions
def remove_sxsw(text):
    sxsw_pat = r"sxsw[a-z]*"
    return re.sub(sxsw_pat, ' ', text, flags=re.I)


def remove_austin_texas(text):
    austin_pat = r"austin"
    texas_pat = r"texas|tx"
    temp = text
    temp = re.sub(austin_pat, ' ', temp, flags=re.I)
    temp = re.sub(texas_pat, ' ', temp, flags=re.I)
    return temp


# Mapping Functions
def map_iphone4(text):
    iphone4_pat = r"iphone[ ]+4"
    return re.sub(iphone4_pat, 'iPhone4', text, flags=re.I)


def map_ipads(text):
    # 1
    ipads_pat = r"ipads"
    return re.sub(ipads_pat, 'ipad', text, flags=re.I)


def map_ipad2(text):
    # 2
    ipad2_pat = r"ipad[ ]+2[s]?"
    return re.sub(ipad2_pat, 'ipad2', text, flags=re.I)


def map_brand(text):
    # EDA
    iphone_pat = r"iphone"
    ipad_pat = r"ipad"
    apple_pat = r"apple"
    temp = text
    temp = re.sub(iphone_pat, 'iPhone', temp, flags=re.I)
    temp = re.sub(ipad_pat, 'iPad', temp, flags=re.I)
    temp = re.sub(apple_pat, 'Apple', temp, flags=re.I)
    return temp


def replace_amp(text):
    pat_amp = r"&amp;"
    return re.sub(pat_amp, '&', text, flags=re.I)


def remove_amp(text):
    pat_amp_single = r" & "
    return re.sub(pat_amp_single, '', text, flags=re.I)


def lemma(nlp, text):
    doc = nlp(text)
    return " ".join([token.lemma_ for token in doc])


def lemmatize_text(text):
    lnlp = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    return lemma(lnlp, text)


def remove_html_syn(text):
    #     html_syn_pat = r"&lt;[a-z]+&gt;"
    html_syntax_pat = r"&lt;[/a-z]+&gt;"
    return re.sub(html_syntax_pat, ' ', text, flags=re.I)


def replace_text_emoji(text):
    love_pat = r"&lt;3+"
    happy_pat = r":[-]?[\)]+"
    sad_pat = r":[-]?[\(]+"
    playful_pat = r":[-]?[p]+"
    wink_pat = r";[-]?[\)]+"
    straightface_pat = r":[-]?[\|]+"

    # replace text emojis
    temp = text
    temp = re.sub(love_pat, ' love ', temp)
    temp = re.sub(happy_pat, ' smiley ', temp)
    temp = re.sub(sad_pat, ' sad ', temp)
    temp = re.sub(playful_pat, ' playful ', temp, flags=re.I)
    temp = re.sub(wink_pat, ' wink ', temp)
    temp = re.sub(straightface_pat, ' straightface ', temp)
    return temp


def refine_before_cleaning(text):
    temp = text
    temp = map_iphone4(temp)
    temp = map_ipads(temp)
    temp = map_ipad2(temp)
    temp = remove_html_syn(temp)
    temp = replace_amp(temp)
    temp = remove_amp(temp)
    temp = replace_text_emoji(temp)
    return temp

# Other
def tokenize_Treebank(text):
    tb = TreebankWordTokenizer()
    return tb.tokenize(text)


def porter_stem_text(text):
    ps = PorterStemmer()
    return [ps.stem(t) for t in text]

def tokenize_stem_join(text):
    tokens = tokenize_Treebank(text)
    stemmed_tokens = porter_stem_text(tokens)
    return ' '.join(stemmed_tokens)


def vectorizer_fit_transform(vectorizer, fit_transform_on, transform_on = None):
    """
    :param vectorizer: vectorizer instance (Count /TFIDF Vectorizer)
    :param fit_transform_on: DataFrame/Series Fit and Transform vectorizer on.
    :param transform_on: DataFrame/Series Transform vectorizer on.
    :return: vectorized 'fit_transform_on' and/or 'transform_on'
    """
    fit_transform_on_vec = vectorizer.fit_transform(fit_transform_on)
    if transform_on is None:
        return fit_transform_on_vec
    else:
        transform_on_vec = vectorizer.transform(transform_on)
        return fit_transform_on_vec, transform_on_vec


def fit_models(X_train, X_test, y_train, y_test):
    """
    Fit models and display classification report
    """
    rf = RandomForestClassifier()
    gb = GradientBoostingClassifier()
    lr = LogisticRegression()
    xg = XGBClassifier()

    for model, name in zip([rf, gb, lr, xg],
                           ['RandomForestClassifier', 'GradientBoostingClassifier', 'LogisticRegression','XGBClassifier']):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        print(f'CLF report for {name}')
        print(classification_report(y_test, y_pred))
        print('---\n')