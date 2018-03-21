from model.seq2seq_attn import Config, SequencePredictor
from model.data_util import create_GloVe_embedding, preprocess_input
from model.util import tokens_to_sentences
import tensorflow as tf

def create_config(voc, E):
    config = Config()
    config.voc_size = len(voc)
    config.embedding_size = E.shape[1]
    config.max_length_x=200
    config.max_length_y=6
    config.voc = voc
    config.idx2word = dict([[v,k] for k,v in voc.items()])
    config.batch_size=1
    return config


def get_model_api():
    voc, E = create_GloVe_embedding()
    config = create_config(voc,E)

    # Need to load embedding matrix?
    model = SequencePredictor(config, E)
    saver = tf.train.Saver()
    #model.build() # don't need, does in the constructor
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, './best/weights/model.weights')

    def model_api(input_data):
        # Pre-process input
        indexed_input = preprocess_input(input_data, voc)
        # Make prediction
        fake_target = [0]
        preds, _, _, _, _ = model.predict_on_batch(sess, [indexed_input], [fake_target])
        pred = list(preds)[0]
        # Post-process output
        output_data = tokens_to_sentences(pred, model.config.idx2word)

        return output_data
    
    return model_api

