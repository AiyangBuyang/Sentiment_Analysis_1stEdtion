import os
from flask import Flask, render_template, request, jsonify
from sentiment_analysis import predict_sentence, analyze_txt, analyze_csv
import pandas as pd
import jieba
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import pickle

app = Flask(__name__)

# 检查是否支持文件反馈
feedback_supported = True


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        input_text = request.form['input_text']

        feedback_supported = True

        # 文件上传处理
        file = request.files['file']
        if file:
            feedback_supported = False
            # 保存上传的文件
            file_path = os.path.join('uploads', file.filename)
            file.save(file_path)

            # 判断文件类型
            if file_path.endswith('.txt'):
                with open(file_path, 'r', encoding='utf-8') as file_content:
                    file_text = file_content.read()
                result, label = analyze_txt(file_text)
            elif file_path.endswith('.csv'):
                df = pd.read_csv(file_path)
                result, label = analyze_csv(df)
            else:
                return render_template('index.html', error="不支持的文件类型")

            return render_template('result.html', result=result, label=label, feedback_supported=feedback_supported)

        if not input_text.strip():
            return render_template('index.html', error="请输入文本或上传文件")

        # 对输入文本进行情感分析
        result, label = predict_sentence(input_text)

        return render_template('result.html', result=result, label=label, feedback_supported=feedback_supported, text=input_text)


@app.route('/train', methods=['POST'])
def train():
    # 调用重新训练模型的函数
    train_sentiment_model()
    # 模拟训练过程
    # import time
    # time.sleep(3)
    # 返回重新训练成功的信息
    return jsonify()


@app.route('/feedback', methods=['POST'])
def feedback():
    if request.method == 'POST':
        text = request.form.get('text')
        label = request.form.get('label')

        # 将文本和标签追加到训练文件
        with open('weibo.csv', 'a', encoding='utf-8') as file:
            file.write(f'{1-int(label)},{text}\n')

        return jsonify()



def train_sentiment_model():
    # 读取原始数据集 weibo_senti_100k.csv
    data_csv = pd.read_csv('sources/weibo_senti_100k.csv')

    # 读取 weibo.tsv 作为额外的训练数据
    data_tsv = pd.read_csv('sources/train.tsv', sep='\t')

    # 合并两个数据集作为训练集
    train_data = pd.concat([data_csv['review'], data_tsv['text_a']], axis=0)
    train_labels = pd.concat([data_csv['label'], data_tsv['label']], axis=0)

    # 读取 test.tsv 作为测试集
    test_data = pd.read_csv('sources/test.tsv', sep='\t')
    test_labels = test_data['label']

    # 加载停用词
    stopwords = set()
    with open('sources/hit_stopwords.txt', 'r', encoding='utf-8') as f:
        stopwords = set([line.strip() for line in f])

    # 分词函数，考虑使用n-grams
    def cut_text(text, n=1):
        words = jieba.lcut(text)  # 使用精确模式分词
        words = [word for word in words if word not in stopwords]

        # 使用n-grams
        if n > 1:
            words += [''.join(words[i:i + n]) for i in range(len(words) - n + 1)]

        return ' '.join(words)

    # 对训练集和测试集进行分词处理，同时考虑bi-grams（n=2）
    train_data_cut = train_data.apply(lambda x: cut_text(x, n=2))
    test_data_cut = test_data['text_a'].apply(lambda x: cut_text(x, n=2))

    # 使用TF-IDF进行特征提取
    tfidf_vectorizer = TfidfVectorizer()
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_data_cut)
    X_test_tfidf = tfidf_vectorizer.transform(test_data_cut)

    # 使用朴素贝叶斯进行训练
    model = MultinomialNB()
    model.fit(X_train_tfidf, train_labels)

    # 保存模型
    with open('sentiment_model_with_tsv_and_ngrams.pkl', 'wb') as model_file:
        pickle.dump(model, model_file)

    with open('tfidf_vectorizer_with_tsv_and_ngrams.pkl', 'wb') as vectorizer_file:
        pickle.dump(tfidf_vectorizer, vectorizer_file)

    # 进行预测
    predictions = model.predict(X_test_tfidf)

    # 评估模型性能
    accuracy = accuracy_score(test_labels, predictions)
    print(f'模型准确率：{accuracy:.2f}')

    # 输出分类报告
    print('分类报告:\n', classification_report(test_labels, predictions))


if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5000)