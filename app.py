# coding:utf-8


from flask import Flask, request, jsonify
import logging
import summarize_4zh
from pegasus_generate import generate_title

app = Flask(__name__)


@app.route('/ai/generate/title', methods=['POST'])
def generate_summary():
    try:
        data = request.get_json()
        content = data.get('content')
        app.logger.info('成功接收content: ' + content)
        origin_content = content

        # content过短
        if len(content) < 30:
            app.logger.info('content过短')
            return jsonify(code=200, msg='content too short')

        # 抽取式抽取关键句
        if len(content) > 700:  # 文章本来就很短的就不用再抽取句子了
            res_textrank = summarize_4zh.textrank_top(content, 10, 2)
            content = ''
            for r in res_textrank:
                content += r['sentence']
            app.logger.info('textrank提取关键句成功: ' + content)

        # 根据关键句生成标题
        titles, scores = generate_title(content[:1025], 3)
        for t in titles:
            app.logger.info('生成标题成功' + t)

        # 数据封装
        generate_titles = []
        for title, score in zip(titles, scores):
            generate_titles.append({'title': title, 'score': score})
        result = {"content": origin_content, "generateTitles": generate_titles}
        app.logger.info('数据封装成功')

        return jsonify(code=200, result=result, msg="success")

    except Exception as e:
        print(e)
        return jsonify(code=200, msg='fail')


def template_resp(status=True, data=[], msg="success"):
    return {
        "status": status,
        "data": data,
        "msg": msg
    }


# 执行
if __name__ == "__main__":
    app.debug = True
    handler = logging.FileHandler('flask.log')
    app.logger.addHandler(handler)
    app.run(
        host='0.0.0.0',
        port=5057
    )
