import io
import os
import numpy as np
import urllib
import tensorflow as tf
import tensorflow_compression as tfc    # pylint:disable=unused-import

# Default URL to fetch metagraphs from.
URL_PREFIX = "https://storage.googleapis.com/tensorflow_compression/metagraphs"
# Default location to store cached metagraphs.
METAGRAPH_CACHE = "/tmp/tfc_metagraphs"


def read_png(filename):
    """Loads a PNG image file."""
    string = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(string, channels=3)
    return tf.expand_dims(image, 0)

def write_png(filename, image):
    """Writes a PNG image file."""
    image = tf.squeeze(image, 0)
    if image.dtype.is_floating:
        image = tf.round(image)
    if image.dtype != tf.uint8:
        image = tf.saturate_cast(image, tf.uint8)
    string = tf.image.encode_png(image)
    tf.io.write_file(filename, string)

def load_cached(filename):
    """Downloads and caches files from web storage."""
    pathname = os.path.join(METAGRAPH_CACHE, filename)
    try:
        with tf.io.gfile.GFile(pathname, "rb") as f:
            string = f.read()
    except tf.errors.NotFoundError:
        print(f"Model {filename} not found locally, downloading")
        url = f"{URL_PREFIX}/{filename}"
        request = urllib.request.urlopen(url)
        try:
            string = request.read()
        finally:
            request.close()
        tf.io.gfile.makedirs(os.path.dirname(pathname))
        with tf.io.gfile.GFile(pathname, "wb") as f:
            f.write(string)
    return string

def get_graph(model):
    string = load_cached(model + ".metagraph")
    metagraph = tf.compat.v1.MetaGraphDef()
    metagraph.ParseFromString(string)
    wrapped_import = tf.compat.v1.wrap_function(
            lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
    graph = wrapped_import.graph
    return graph

def instantiate_model_signature(model, signature=None, inputs=None, outputs=None):
    """Imports a trained model and returns one of its signatures as a function."""
    string = load_cached(model + ".metagraph")
    metagraph = tf.compat.v1.MetaGraphDef()
    metagraph.ParseFromString(string)
    wrapped_import = tf.compat.v1.wrap_function(
            lambda: tf.compat.v1.train.import_meta_graph(metagraph), [])
    graph = wrapped_import.graph

    if inputs is None:
        inputs = metagraph.signature_def[signature].inputs
        inputs = [graph.as_graph_element(inputs[k].name) for k in sorted(inputs)]
    else:
        inputs = [graph.as_graph_element(t) for t in inputs]
    if outputs is None:
        outputs = metagraph.signature_def[signature].outputs
        outputs = [graph.as_graph_element(outputs[k].name) for k in sorted(outputs)]
    else:
        outputs = [graph.as_graph_element(t) for t in outputs]
    print(f"Created GraphFunc for model {model}.")
    return wrapped_import.prune(inputs, outputs)

def get_model_for_level(level, latent=True):
    """Get GraphFunc of the compression model for a given level.
       If latent is true, the func outputs the latent representation, otherwise outputs the compressed image."""
    if latent:
        return instantiate_model_signature(f"bmshj2018-hyperprior-msssim-{level}", inputs=["input_image:0"], outputs=["entropy_model/entropy_model_2/Cast:0"])
    else:
        return instantiate_model_signature(f"bmshj2018-hyperprior-msssim-{level}", inputs=["input_image:0"], outputs=["GridAlign_1/strided_slice:0"])
