# -*- coding: utf-8 -*-

import pickle

import tensorflow as tf
from tinydb import TinyDB
from tinyrecord import transaction


def load_preprocess():
    """
    Load the Preprocessed Training data and return them in batches of <batch_size> or less
    """
    with open('checkpoint/preprocess.p', mode='rb') as in_file:
        return pickle.load(in_file)


_, (source_vocab_to_int, _), (_, target_int_to_vocab) = load_preprocess()


def sentence_to_seq(sentence, vocab_to_int):
    sentence = sentence.lower()
    unk_id = vocab_to_int.get('<UNK>')
    ids = [vocab_to_int.get(word, unk_id) for word in sentence.split()]
    return ids


def save_translation(source, translated):
    table = TinyDB('db.json').table('transactions')
    with transaction(table) as tr:
        tr.insert({'source': source, 'translated': translated})


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
