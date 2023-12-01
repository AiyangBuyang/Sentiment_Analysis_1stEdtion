import jieba
import pickle
import re


# 加载模型
with open('models/sentiment_model_with_tsv_and_ngrams.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('models/tfidf_vectorizer_with_tsv_and_ngrams.pkl', 'rb') as vectorizer_file:
    tfidf_vectorizer = pickle.load(vectorizer_file)


def predict_sentiment(text_tfidf):
    # 进行情感预测
    prediction = model.predict(text_tfidf)

    # 返回预测结果
    return prediction[0]


def predict_sentence(text):
    # 分词和TF-IDF转换
    text_cut = ' '.join(jieba.cut(text))
    text_tfidf = tfidf_vectorizer.transform([text_cut])
    sentiment = predict_sentiment(text_tfidf)
    if sentiment == 1:
        return "文本情感应该是积极的。", 1
    else:
        return "文本情感应该是消极的。", 0


def analyze_txt(file_text):
    # 按回车分割文本
    sentences = re.split(r'\s', file_text)

    positive_count = 0
    negative_count = 0

    for sentence in sentences:
        # 分词和TF-IDF转换
        sentence_cut = ' '.join(jieba.cut(sentence))
        sentence_tfidf = tfidf_vectorizer.transform([sentence_cut])
        sentiment = predict_sentiment(sentence_tfidf)
        if sentiment == 1:
            positive_count += 1
        else:
            negative_count += 1

    total_sentences = len(sentences)
    if positive_count >= total_sentences / 2:
        return "文本情感应该是积极的。", 1
    else:
        return "文本情感应该是消极的。", 0


def analyze_csv(df):
    positive_count = 0
    negative_count = 0

    for index, row in df.iterrows():
        for column in df.columns:
            text = str(row[column])
            text_cut = ' '.join(jieba.cut(text))
            text_tfidf = tfidf_vectorizer.transform([text_cut])
            sentiment = predict_sentiment(text_tfidf)
            if sentiment == 1:
                positive_count += 1
            else:
                negative_count += 1

    total_items = df.size
    if positive_count >= total_items / 2:
        return "文本情感应该是积极的。", 1
    else:
        return "文本情感应该是消极的。", 0
