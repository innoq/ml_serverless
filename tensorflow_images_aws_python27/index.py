# https://github.com/Accenture/serverless-ephemeral/blob/master/docs/build-tensorflow-package.md

import tensorflow as tf
import numpy as np
import json
import urllib2
from urllib import urlretrieve

labels_dict = {0: 'daisy', 1: 'dandelion', 2: 'roses', 3: 'sunflowers', 4: 'tulips'}
graph_path = 'https://s3-eu-west-1.amazonaws.com/aws.example.event2015/retrained_graph.pb'

graph_def_map = {}

def run_inference_on_image():

    img = tf.image.decode_jpeg(tf.read_file('/tmp/image.jpg'), channels=3)
    img = tf.image.resize_images(img, [224, 224])
    img = tf.cast(img, np.float32)
    img = tf.expand_dims(img, 0)
    img = img / 255.

    graph_def = graph_def_map.get("graph_def")
    if(graph_def == None):
        urlretrieve(graph_path, '/tmp/retrained_graph.pb')
        f = tf.gfile.GFile('/tmp/retrained_graph.pb', "rb")
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        tf.import_graph_def(graph_def)
        graph_def_map["graph_def"] = graph_def

    with tf.Session() as sess:
        input_node = sess.graph.get_tensor_by_name('import/input:0')
        output_node = sess.graph.get_tensor_by_name('import/final_result:0')
        predictions = sess.run(output_node, feed_dict={
                               input_node: img.eval()})[0]
        top_5_predictions = predictions.argsort()[-5:][::-1]
        prediction_names = [labels_dict[i] for i in top_5_predictions]
        return prediction_names[0]


def which_flower(event, context):
    url = event.get('queryStringParameters').get('url')

    f = open('/tmp/image.jpg', 'wb')
    f.write(urllib2.urlopen(url).read())
    f.close()

    strResult = run_inference_on_image()

    objRet = {
        'statusCode': 200,
        'body': json.dumps({"return": strResult})
    }
    return objRet
