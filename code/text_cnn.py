import tensorflow as tf
import numpy as np


class TextCNN(object):
    """
    A CNN for text classification.
    Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    """
    def __init__(
      self, sequence_length, num_classes, vocab_size,
      embedding_size, filter_sizes, num_filters, l2_reg_lambda=0.0):

        # TODO: Add placeholders for the input, label and dropout
        # self.input_x should be the name of the placeholder for the input
        
        # self.pretrained_embeddings is a matrix which is a lookup table - for every word in the vocabulary
        # it contains a low dimensional word vector representation
        # TODO: Define tf.Variable of shape [vocab_size, embedding_size] and type tf.float32 with
        # uniform random initialization (in interval [-1,1])
        self.pretrained_embeddings = ...

        # Keeping track of l2 regularization loss (optional)
        l2_loss = 0.0

        # Embedding layer
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            # embedding_lookup is the operation which does the actual embedding i.e. it replaces each word (represented
            # as the index of the word in the vocabulary) with its corresponding embedding from the lookup matrix
            # self.embedded_tokens will be a 3-dim tensor of shape [None, sequence_length, embedding_size] where None
            # will depend on the batch size
            self.embedded_tokens = tf.nn.embedding_lookup(self.pretrained_embeddings, self.input_x)
            # conv2d expects a 4-dim tensor of size [batch size, width, height, channels]. self.embedded_tokens doesn't
            # contain the dimension for the channel so we expand the tensor to have one more dimension manually
            self.embedded_tokens_expanded = tf.expand_dims(self.embedded_tokens, -1)

        # TODO: Create a convolution + maxpool layer for each filter size
        # Since we have several filters, we need several convolutions, collect
        # them an combine them afterwards
           

        # Create a convolution + maxpool layer for each filter size
        # Store filter outputs in pooled_outputs
        pooled_outputs = []
        for _, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
              # For the convolution, use tf.nn.conv2
              # For the max-pooling, use tf.nn.max_pool
              # the filter should be a 4-D tensor of shape [filter_height, filter_width, in_channels, out_channels]
              # filter_height represents how many words the filter cover
              # filter_width is the same as the embedding size
              filter_shape = [...]
              W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
              b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
              
              # TODO: Apply convolutions
              # For NLP tasks the stride is typically [1, 1, 1, 1] and in_channels=1
              conv = ...
              
              # TODO: Use a relu non-linearity before max-pooling
              
              # TODO: Maxpooling over the outputs
              pooled = ...
              
              # Append 
              pooled_outputs.append(pooled)
                
                
        # TODO: Combine all the pooled features
        self.h_pool_flat = ...
        
        # TODO: Add dropout 
        with tf.name_scope("dropout"):
          self.h_drop = ...
                   
        # Final (unnormalized) scores and predictions (optional: add L2 regularization of soft-max weights)
        with tf.name_scope("output"):
            self.scores = ...
            self.predictions = ...
            
            
        # TODO: Define Mean cross-entropy loss using  tf.nn.sparse_softmax_cross_entropy_with_logits
        with tf.name_scope("loss"):
          self.loss = ...
            
        # TODO: Define Accuracy (hint: use  tf.equal and tf.argmax)
        with tf.name_scope("accuracy"):
            self.accuracy = ...