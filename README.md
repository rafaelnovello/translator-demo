# English to French Translation App


[Demo project](japronto-translator.herokuapp.com) for language translation with seq2seq tensorflow model.

## Install

Clone the repository or download the zip package.

Install packages

```bash
$ pip install -r requirements.txt
```

## Model Training

The web application depends on the trained model. This process happens on .ipynb file here. Follow the instructions there 
to train the model.

## Web App Running

The web app uses the [japronto](https://github.com/squeaky-pl/japronto). To run them, execute the file:

```bash
$ python app.py
```
## About the Model

[Sequence to sequence](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) is a 
Recurent Neural Network architecture that uses two RNN, an encoder that processes the input and a decoder that generates 
the output.
