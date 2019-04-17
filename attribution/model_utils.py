import os
import tempfile

from keras.layers import Input
from keras.models import Model, clone_model

import keras


def replace_softmax_with_logits(model, softmax_layer=-1):
    model_p = keras.models.clone_model(model)
    model_p.layers[softmax_layer].activation = keras.activations.linear
    tmp_path = os.path.join(
        tempfile.gettempdir(), 
        next(tempfile._get_candidate_names()) + '.h5')
    try:
        model_p.save(tmp_path)
        return keras.models.load_model(tmp_path)
    finally:
        os.remove(tmp_path)


def top_slice(model, start_layer, input_tensor=None):
    '''
    Given a model, f = g o h, returns g, i.e., the top slice of the model,
    starting at the given layer.

    Parameters
    ----------
    model : keras.models.Model
        The model we would like to take the top of. The computation graph of the
        given model is not modified.
    start_layer: keras.layers.Layer or int or str
        The layer to begin the slice at, given as either an instance of 
        keras.layers.Layer, an integer index of a layer in model, or the string
        name of a layer in model. The returned model will have the same input 
        shape as the given layer. It is assumed that start_layer defines a valid
        slice, i.e., that start_layer covers the output of the computation graph
        (all paths from the input to the output of the computation graph contain
        start_layer).
    input_tensor : K.Tensor or list of K.Tensor, optional
        A tensor or list of tensors to be passed as input to g. If input_tensor
        is None, new placeholders (Input layers) will be created. If provided,
        there must be one input tensor per input to start_layer.

    Returns
    -------
    keras.models.Model
        A keras model representing the top of the given model, beginning at (and
        including) start_layer.
    '''
    if isinstance(layer, int):
        layer = model.layers[layer]
    elif isinstance(layer, keras.layers.Layer):
        layer = layer
    elif isinstance(layer, str):
        layer = model.get_layer(layer)
    else:
        raise ValueError('Need to pass layer index, name, or instance.')

    # First, make a new copy of the model so it can be modified without changing
    # the computation graph of the original model.
    f = clone_model(model)

    # If input_tensor is not specified, we will make new input placeholders for
    # the top slice model.
    if input_tensor is None:
        if isinstance(start_layer.input_shape, list):
            # There were multiple input layers; make an input for each.
            z = [Input(shape[1:]) for shape in start_layer.input_shape]
        else:
            z = Input(start_layer.input_shape[1:])

    # OTherwise we use the given tensor (or tensors) as input.
    else:
        if isinstance(input_tensor, list):
            # There were multiple input tensors; make an input for each.
            z = [Input(tensor=tensor) for tensor in input_tensor]
        else:
            z = Input(tensor=input_tensor)

    # TODO: should make sure this is general.
    top_layer = f.output._keras_history[0]

    return Model(z, _model_on_prev_layer(top_layer, start_layer, z))


def _get_inbound_layers(l):
    '''
    Given a keras layer, l, returns a list of the inbound layers to l, i.e., the
    layers preceding l in the network.

    TODO: make sure this works on both backends.
    '''
    inbound_layers = []
    for i_node in l._inbound_nodes:
        for i_layer in i_node.inbound_layers:
            inbound_layers.append(i_layer)
    return inbound_layers


def _model_on_prev_layer(l, l0, new_in):
    if l == l0:
        # Hook this up to our new input.
        return new_in if isinstance(l0, Input) else l0(new_in)

    # Call recursively.
    inbound_layers = _get_inbound_layers(l)
    if len(inbound_layers) == 1:
        return l(model_on_prev_layer(inbound_layers[0], l0, new_in))
    else:
        return l([
            model_on_prev_layer(p, l0, new_in)
            for p in get_inbound_layers(l)])