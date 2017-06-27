# -*- coding: utf-8 -*-

import helper

import tensorflow as tf
from japronto import Application
from jinja2 import Environment, FileSystemLoader, select_autoescape


_, (source_vocab_to_int, _), (_, target_int_to_vocab) = helper.load_preprocess()
env = Environment(
    loader=FileSystemLoader('templates'),
    autoescape=select_autoescape(['html'])
)


def index(request):
    if request.method == 'POST':
        source = request.form.get('source', '')
        translated = translate(source)
    else:
        translated = ''

    template = env.get_template('index.html')
    return request.Response(
        text=template.render(translated=translated),
        mime_type='text/html'
    )


def sentence_to_seq(sentence, vocab_to_int):
    sentence = sentence.lower()
    unk_id = vocab_to_int.get('<UNK>')
    ids = [vocab_to_int.get(word, unk_id) for word in sentence.split()]
    return ids


def translate(translate_sentence):
    batch_size = 256
    load_path = 'checkpoint/dev'
    translate_sentence = sentence_to_seq(translate_sentence, source_vocab_to_int)

    loaded_graph = tf.Graph()
    with tf.Session(graph=loaded_graph) as sess:
        # Load saved model
        loader = tf.train.import_meta_graph(load_path + '.meta')
        loader.restore(sess, load_path)

        input_data = loaded_graph.get_tensor_by_name('input:0')
        logits = loaded_graph.get_tensor_by_name('predictions:0')
        target_sequence_length = loaded_graph.get_tensor_by_name('target_sequence_length:0')
        source_sequence_length = loaded_graph.get_tensor_by_name('source_sequence_length:0')
        keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

        translate_logits = sess.run(
            logits, {
                input_data: [translate_sentence]*batch_size,
                target_sequence_length: [len(translate_sentence)*2]*batch_size,
                source_sequence_length: [len(translate_sentence)]*batch_size,
                keep_prob: 1.0
            }
        )[0]
        translated = " ".join([target_int_to_vocab[i] for i in translate_logits[:-1]])
        return translated

app = Application()
app.router.add_route('/', index, methods=['GET', 'POST'])
app.run(debug=True)
