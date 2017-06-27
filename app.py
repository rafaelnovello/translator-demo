# -*- coding: utf-8 -*-

import helper

from japronto import Application
from jinja2 import Environment, FileSystemLoader, select_autoescape


env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html'])
)


def index(request):
    if request.method == 'POST':
        source = request.form.get('source', '')
        translated = helper.translate(source)
        helper.save_translation(source, translated)
    else:
        translated = ''

    template = env.get_template('index.html')
    return request.Response(
        text=template.render(translated=translated),
        mime_type='text/html'
    )


def words(request):
    with open('words.txt', 'r') as f:
        words = f.readlines()

    template = env.get_template('vocab.html')
    return request.Response(
        text=template.render(words=words),
        mime_type='text/html'
    )


app = Application()
app.router.add_route('/', index, methods=['GET', 'POST'])
app.router.add_route('/vocab/', words, methods=['GET'])
app.run(debug=True)
