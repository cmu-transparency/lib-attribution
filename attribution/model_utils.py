import os
import tempfile

import keras

def replace_softmax_with_logits(model, softmax_layer=-1):
    model_p = keras.models.clone_model(model)
    model_p.layers[softmax_layer].activation = keras.activations.linear
    tmp_path = os.path.join(tempfile.gettempdir(), next(tempfile._get_candidate_names()) + '.h5')
    try:
        model_p.save(tmp_path)
        return keras.models.load_model(tmp_path)
    finally:
        os.remove(tmp_path)