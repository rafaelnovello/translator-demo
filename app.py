# -*- coding: utf-8 -*-
import os
import helper

from japronto import Application


@helper.simple_localizer
def index(request, env):
    if request.method == 'POST':
        source = request.form.get('source', '')
        translated = helper.translate(source)
        helper.save_translation(source, translated)
    else:
        source = translated = ''

    template = env.get_template('index.html')
    return request.Response(
        text=template.render(
            source=source,
            translated=translated
        ),
        mime_type='text/html'
    )


@helper.simple_localizer
def words(request, env):
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
port = os.environ.get('PORT') or 8080
app.run(debug=True, port=int(port), reload=True)
