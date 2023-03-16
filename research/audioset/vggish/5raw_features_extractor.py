# Copyright 2017 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

r"""Extractor of the 5 first raw features from embedding vectors.

This file answers Biophonia question for the technical interview they
have in their recruitment process. It is widely inspired from the 
vggish inference demo, with all the command line capabilities that have
been removed to let he code be launched from VS Code

Usage:
  # Run this script in an code editor (VS Code for instance)
"""

import tensorflow.compat.v1 as tf

import vggish_input
import vggish_params
import vggish_slim

# Read the wav file provided
examples_batch = vggish_input.wavfile_to_examples("Data/great_tit.wav")
print(examples_batch)

# Paths to downloaded VGGish files.
checkpoint_path = 'vggish_model.ckpt'

with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint_path)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                feed_dict={features_tensor: examples_batch})
    
    # Print the 5 first raw features of the embedding layer
    print(f"The five first raw features of the 0.96 s are: {embedding_batch[0,:5]}")
