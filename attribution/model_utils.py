import os
import tempfile

from keras.layers import Input
from keras.models import Model
from keras.models import clone_model as _clone_model

import keras


def clone_model(model):
    m2 = _clone_model(model)
    m2.set_weights(model.get_weights())
    return m2

def replace_softmax_with_logits(model, softmax_layer=-1, custom_objects=None):
    model_p = clone_model(model)
    model_p.compile(model.optimizer, model.loss_functions, model.metrics_names[1:])
    model_p.layers[softmax_layer].activation = keras.activations.linear
    tmp_path = os.path.join(
        tempfile.gettempdir(), 
        next(tempfile._get_candidate_names()) + '.h5')
    try:
        model_p.save(tmp_path)
        # model_p.save_weights(tmp_path)
        return keras.models.load_model(tmp_path, custom_objects=custom_objects)
        # model_p.load_weights(tmp_path)
        return model_p
    finally:
        os.remove(tmp_path)

def top_slice(model, start_layer, input_tensor=None, clone=True):
    '''
    Given a model, f = g o h, returns g, i.e., the top slice of the model,
    starting after the given layer (starting with the output of the given 
    layer).

    Parameters
    ----------
    model : keras.models.Model
        The model we would like to take the top of. The computation graph of the
        given model is not modified unless `clone` is set to False.
    start_layer: keras.layers.Layer or int or str
        The layer to begin the slice after, given as either an instance of 
        keras.layers.Layer, an integer index of a layer in model, or the string
        name of a layer in model. The returned model will have the same input 
        shape as the output shape of the given layer. It is assumed that 
        `start_layer` defines a valid slice, i.e., that `start_layer` covers the 
        output of the computation graph (all paths from the input to the output 
        of the computation graph contain `start_layer`).
    input_tensor : K.Tensor, optional
        A tensor to be passed as input to g. If `input_tensor` is None, new 
        placeholders (Input layers) will be created. If provided, `input_tensor` 
        must match the shape of the output of `start_layer`.
    clone : boolean
        If set to True, the computation graph of `model` will not be modified,
        and the returned model will not be connected to the original model.


    Returns
    -------
    keras.models.Model
        A keras model representing the top of the given model, beginning at (and
        including) `start_layer`.
    '''
    if isinstance(start_layer, int):
        start_layer = model.layers[start_layer]
    elif isinstance(start_layer, keras.layers.Layer):
        start_layer = start_layer
    elif isinstance(start_layer, str):
        start_layer = model.get_layer(start_layer)
    else:
        raise ValueError('Need to pass layer index, name, or instance.')

    # First, if specified, make a new copy of the model so it can be modified 
    # without changing the computation graph of the original model.
    if clone:
        f = clone_model(model)
        start_layer = f.get_layer(start_layer.name)
    else:
        f = model

    # If input_tensor is not specified, we will make new input placeholders for
    # the top slice model.
    if input_tensor is None:
        z = Input(start_layer.output_shape[1:])

    # Otherwise we use the given tensor (or tensors) as input.
    else:
        z = Input(tensor=input_tensor)

    # TODO(kleino): should make sure this is general.
    top_layer = f.output._keras_history[0]

    return Model(z, _model_on_prev_layer(top_layer, start_layer, z))


def _get_inbound_layers(l):
    '''
    Given a keras layer, l, returns a list of the inbound layers to l, i.e., the
    layers preceding l in the network.

    TODO(kleino): make sure this works on both backends.
    '''
    inbound_layers = []
    for i_node in l._inbound_nodes:
        for i_layer in i_node.inbound_layers:
            inbound_layers.append(i_layer)
    return inbound_layers


def _model_on_prev_layer(l, l0, new_in):
    if l == l0:
        # Hook this up to our new input.
        return new_in

    # Call recursively.
    inbound_layers = _get_inbound_layers(l)
    if len(inbound_layers) == 1:
        return l(_model_on_prev_layer(inbound_layers[0], l0, new_in))
    else:
        return l([
            _model_on_prev_layer(p, l0, new_in)
            for p in _get_inbound_layers(l)])
