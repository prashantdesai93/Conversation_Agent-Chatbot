from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
from random import random
import sys
import time

import numpy as np
from six.moves import xrange  
import tensorflow as tf
import heapq
import data_utils
import seq2seq_model
tf.logging.set_verbosity(tf.logging.ERROR)

try:
    reload
except NameError:
 
    pass
else:
    reload(sys).setdefaultencoding('utf-8')
    
try:
    from ConfigParser import SafeConfigParser
except:
    from configparser import SafeConfigParser 
    
gCon = {}

def get_config(config_file='seq2seq.ini'):
    parser = SafeConfigParser()
    parser.read(config_file)
  
    _conf_ints = [ (key, int(value)) for key,value in parser.items('ints') ]
    _conf_floats = [ (key, float(value)) for key,value in parser.items('floats') ]
    _conf_strings = [ (key, str(value)) for key,value in parser.items('strings') ]
    return dict(_conf_ints + _conf_floats + _conf_strings)


_buckets = [(5, 10), (10, 15), (20, 25), (40, 50)]


def read_data(source_path, target_path, max_size=None):
  data_set = [[] for _ in _buckets]
  with tf.gfile.GFile(source_path, mode="r") as source_file:
    with tf.gfile.GFile(target_path, mode="r") as target_file:
      source, target = source_file.readline(), target_file.readline()
      count = 0
      while source and target and (not max_size or count < max_size):
        count += 1
        if count % 100000 == 0:
          print("  reading data line %d" % count)
          sys.stdout.flush()
        source_ids = [int(x) for x in source.split()]
        target_ids = [int(x) for x in target.split()]
        target_ids.append(data_utils.EOS_ID)
        for bucket_id, (source_size, target_size) in enumerate(_buckets):
          if len(source_ids) < source_size and len(target_ids) < target_size:
            data_set[bucket_id].append([source_ids, target_ids])
            break
        source, target = source_file.readline(), target_file.readline()
  return data_set


def create_model(session, forward_only):

  """Create model and initialize or load parameters"""
  model = seq2seq_model.Seq2SeqModel( gCon['enc_vocab_size'], gCon['dec_vocab_size'], _buckets, gCon['layer_size'], gCon['num_layers'], gCon['max_gradient_norm'], gCon['batch_size'], gCon['learning_rate'], gCon['learning_rate_decay_factor'], forward_only=forward_only)

  if 'pretrained_model' in gCon:
      model.saver.restore(session,gCon['pretrained_model'])
      return model

  ckpt = tf.train.get_checkpoint_state(gCon['working_directory'])
  
  checkpoint_suffix = ""
  if tf.__version__ > "0.12":
      checkpoint_suffix = ".index"
  if ckpt and tf.gfile.Exists(ckpt.model_checkpoint_path + checkpoint_suffix):
    print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
    model.saver.restore(session, ckpt.model_checkpoint_path)
  else:
    print("Created model with fresh parameters.")
    session.run(tf.initialize_all_variables())
  return model


def train():
  
  print("Preparing data in %s" % gCon['working_directory'])
  enc_train, dec_train, enc_dev, dec_dev, _, _ = data_utils.prepare_custom_data(gCon['working_directory'],gCon['train_enc'],gCon['train_dec'],gCon['test_enc'],gCon['test_dec'],gCon['enc_vocab_size'],gCon['dec_vocab_size'])


  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.666)
  config = tf.ConfigProto(gpu_options=gpu_options)
  config.gpu_options.allocator_type = 'BFC'

  with tf.Session(config=config) as sess:
   
    print("Creating %d layers of %d units." % (gCon['num_layers'], gCon['layer_size']))
    model = create_model(sess, False)

  
    print ("Reading development and training data (limit: %d)."
           % gCon['max_train_data_size'])
    dev_set = read_data(enc_dev, dec_dev)
    train_set = read_data(enc_train, dec_train, gCon['max_train_data_size'])
    train_bucket_sizes = [len(train_set[b]) for b in xrange(len(_buckets))]
    train_total_size = float(sum(train_bucket_sizes))

    
    train_buckets_scale = [sum(train_bucket_sizes[:i + 1]) / train_total_size
                           for i in xrange(len(train_bucket_sizes))]


    step_time, loss = 0.0, 0.0
    current_step = 0
    previous_losses = []
    while True:
      random_number_01 = np.random.random_sample()
      bucket_id = min([i for i in xrange(len(train_buckets_scale))
                       if train_buckets_scale[i] > random_number_01])

      
      start_time = time.time()
      encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          train_set, bucket_id)
      _, step_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                   target_weights, bucket_id, False)
      step_time += (time.time() - start_time) / gCon['steps_per_checkpoint']
      loss += step_loss / gCon['steps_per_checkpoint']
      current_step += 1

      
      if current_step % gCon['steps_per_checkpoint'] == 0:
        
        perplexity = math.exp(loss) if loss < 300 else float('inf')
        print ("global step %d learning rate %.4f step-time %.2f perplexity "
               "%.2f" % (model.global_step.eval(), model.learning_rate.eval(),
                         step_time, perplexity))
        
        if len(previous_losses) > 2 and loss > max(previous_losses[-3:]):
          sess.run(model.learning_rate_decay_op)
        previous_losses.append(loss)
        
        checkpoint_path = os.path.join(gCon['working_directory'], "seq2seq.ckpt")
        model.saver.save(sess, checkpoint_path, global_step=model.global_step)
        step_time, loss = 0.0, 0.0
        
        for bucket_id in xrange(len(_buckets)):
          if len(dev_set[bucket_id]) == 0:
            print("  eval: empty bucket %d" % (bucket_id))
            continue
          encoder_inputs, decoder_inputs, target_weights = model.get_batch(
              dev_set, bucket_id)
          _, eval_loss, _ = model.step(sess, encoder_inputs, decoder_inputs,
                                       target_weights, bucket_id, True)
          eval_ppx = math.exp(eval_loss) if eval_loss < 300 else float('inf')
          print("  eval: bucket %d perplexity %.2f" % (bucket_id, eval_ppx))
        sys.stdout.flush()
		
### Greedy decoder
# def decode():

  
  # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  # config = tf.ConfigProto(gpu_options=gpu_options)

  # with tf.Session(config=config) as sess:
    
    # model = create_model(sess, True)
    # model.batch_size = 1  

    # enc_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.enc" % gCon['enc_vocab_size'])
    # dec_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.dec" % gCon['dec_vocab_size'])

    # enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    # _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)


    # sys.stdout.write("> ")
    # sys.stdout.flush()
    # sentence = sys.stdin.readline()
    # while sentence:
     
      # token_ids = data_utils.sentence_to_token_ids(tf.compat.as_bytes(sentence), enc_vocab)
      
      # bucket_id = min([b for b in xrange(len(_buckets))
                       # if _buckets[b][0] > len(token_ids)])
      
      # encoder_inputs, decoder_inputs, target_weights = model.get_batch(
          # {bucket_id: [(token_ids, [])]}, bucket_id)
      
      # _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs,
                                       # target_weights, bucket_id, True)
     
      # outputs = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      
      # if data_utils.EOS_ID in outputs:
        # outputs = outputs[:outputs.index(data_utils.EOS_ID)]
      # # print(output_logits)
      # print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
      # print("> ", end="")
      # sys.stdout.flush()
      # sentence = sys.stdin.readline()
## Beam Search Decoder
def decode():

  
  gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.2)
  config = tf.ConfigProto(gpu_options=gpu_options)

  with tf.Session(config=config) as sess:
    
    model = create_model(sess, True)
    model.batch_size = 1  

    enc_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.enc" % gCon['enc_vocab_size'])
    dec_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.dec" % gCon['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)


    sys.stdout.write("> ")
    sys.stdout.flush()
    sentence = sys.stdin.readline()
    while sentence:
        sentence = tf.compat.as_bytes(sentence)
        predicted_sentence = get_predicted_sentence(5,_buckets,0.7, sentence, enc_vocab, rev_dec_vocab, model, sess)
        # print(predicted_sentence)
        if isinstance(predicted_sentence, list):
            #for sent in predicted_sentence:
                #print("  (%s) -> %s" % (sent['prob'], sent['dec_inp']))
            # print(predicted_sentence[1]['dec_inp'])
            # if data_utils.EOS_ID in predicted_sentence[1]:
            #   outputs = predicted_sentence[1]['dec_inp'][:predicted_sentence[1]['dec_inp'].index(data_utils.EOS_ID)]
            #print(" ".join([tf.compat.as_str(predicted_sentence[1]['dec_inp'])]))
            # print(str(data_utils._EOS) in predicted_sentence[1]['dec_inp'])
            print(str(predicted_sentence[0]['dec_inp']).replace('_GO',"").replace("_PAD","").replace("_EOS",""))
                  
                  


            # outputs = [int(np.argmax(logit, axis=1)) for logit in predicted_sentence]
      
            # if data_utils.EOS_ID in outputs:
            #   outputs = outputs[:outputs.index(data_utils.EOS_ID)]
              
            # print(" ".join([tf.compat.as_str(rev_dec_vocab[output]) for output in outputs]))
        else:
            print(sentence, ' -> ', predicted_sentence)
            
        sys.stdout.write("> ")
        sys.stdout.flush()
        sentence = sys.stdin.readline()
		
def get_predicted_sentence(beam_size,_buckets,antilm, input_sentence, vocab, rev_vocab, model, sess, debug=False, return_raw=False):
    
    def softmax(x):
      prob = np.exp(x) / np.sum(np.exp(x), axis=0)
      return prob
    
    def dict_lookup(rev_vocab, out):
      # print(rev_vocab)
      # print(out[0])
      word = rev_vocab[out[0]] if (out[0] < len(rev_vocab)) else data_utils._UNK
      if isinstance(word, bytes):
        word = word.decode()
      return word

    def model_step(enc_inp, dec_inp, dptr, target_weights, bucket_id):
      _, _, logits = model.step(sess, enc_inp, dec_inp, target_weights, bucket_id, forward_only=True)
      prob = softmax(logits[dptr][0])
      # print("model_step @ %s" % (datetime.now()))
      return prob

    def greedy_dec(output_logits, rev_vocab):
      selected_token_ids = [int(np.argmax(logit, axis=1)) for logit in output_logits]
      if data_utils.EOS_ID in selected_token_ids:
        eos = selected_token_ids.index(data_utils.EOS_ID)
        selected_token_ids = selected_token_ids[:eos]
      output_sentence = ' '.join([dict_lookup(rev_vocab, t) for t in selected_token_ids])
      return output_sentence
    
    # print(input_sentence)
    input_token_ids = data_utils.sentence_to_token_ids(input_sentence, vocab)

    # Which bucket does it belong to?
    #bucket_id = min([b for b in xrange(len(_buckets)) if _buckets[b][0] > len(token_ids)])
    bucket_id = min([b for b in range(len(_buckets)) if _buckets[b][0] > len(input_token_ids)])
    outputs = []
    feed_data = {bucket_id: [(input_token_ids, outputs)]}

    # Get a 1-element batch to feed the sentence to the model.
    encoder_inputs, decoder_inputs, target_weights = model.get_batch(feed_data, bucket_id)
    if debug: print("\n[get_batch]\n", encoder_inputs, decoder_inputs, target_weights)

    ### Original greedy decoding
    if beam_size == 1:
      _, _, output_logits = model.step(sess, encoder_inputs, decoder_inputs, target_weights, bucket_id, forward_only=True)
      return [{"dec_inp": greedy_dec(output_logits, rev_vocab), 'prob': 1}]

    # Get output logits for the sentence.
    beams, new_beams, results = [(1, 0, {'eos': 0, 'dec_inp': decoder_inputs, 'prob': 1, 'prob_ts': 1, 'prob_t': 1})], [], [] # initialize beams as (log_prob, empty_string, eos)
    dummy_encoder_inputs = [np.array([data_utils.PAD_ID]) for _ in range(len(encoder_inputs))]
    
    for dptr in range(len(decoder_inputs)-1):
      if dptr > 0: 
        target_weights[dptr] = [1.]
        beams, new_beams = new_beams[:beam_size], []
      if debug: print("=====[beams]=====", beams)
      heapq.heapify(beams)  # since we will remove something
      for prob, _, cand in beams:
        if cand['eos']: 
          results += [(prob, 0, cand)]
          continue

        # normal seq2seq
        if debug: print(cand['prob'], " ".join([dict_lookup(rev_vocab, w) for w in cand['dec_inp']]))

        all_prob_ts = model_step(encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
        if antilm:
          # anti-lm
          # print("<><><><M><><> ",antilm)
          all_prob_t  = model_step(dummy_encoder_inputs, cand['dec_inp'], dptr, target_weights, bucket_id)
          # adjusted probability
          all_prob    = all_prob_ts - antilm * all_prob_t + 0.05 * dptr + random() * 1e-50
        else:
          all_prob_t  = [0]*len(all_prob_ts)
          all_prob    = all_prob_ts

        # suppress copy-cat (respond the same as input)
        if dptr < len(input_token_ids):
          all_prob[input_token_ids[dptr]] = all_prob[input_token_ids[dptr]] * 0.01

        # for debug use
        if return_raw: return all_prob, all_prob_ts, all_prob_t
        
        # beam search  
        for c in np.argsort(all_prob)[::-1][:beam_size]:
          new_cand = {
            'eos'     : (c == data_utils.EOS_ID),
            'dec_inp' : [(np.array([c]) if i == (dptr+1) else k) for i, k in enumerate(cand['dec_inp'])],
            'prob_ts' : cand['prob_ts'] * all_prob_ts[c],
            'prob_t'  : cand['prob_t'] * all_prob_t[c],
            'prob'    : cand['prob'] * all_prob[c],
          }
          new_cand = (new_cand['prob'], random(), new_cand) # stuff a random to prevent comparing new_cand
          
          try:
            if (len(new_beams) < beam_size):
              heapq.heappush(new_beams, new_cand)
            elif (new_cand[0] > new_beams[0][0]):
              heapq.heapreplace(new_beams, new_cand)
          except Exception as e:
            print("[Error]", e)
            print("-----[new_beams]-----\n", new_beams)
            print("-----[new_cand]-----\n", new_cand)
    
    results += new_beams  # flush last cands

    # post-process results
    res_cands = []
    for prob, _, cand in sorted(results, reverse=True):
      cand['dec_inp'] = " ".join([dict_lookup(rev_vocab, w) for w in cand['dec_inp']])
      res_cands.append(cand)
    return res_cands[:beam_size]


def init_session(sess, conf='seq2seq.ini'):
    global gCon
    gCon = get_config(conf)
 
    
    model = create_model(sess, True)
    model.batch_size = 1  

    
    enc_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.enc" % gCon['enc_vocab_size'])
    dec_vocab_path = os.path.join(gCon['working_directory'],"vocab%d.dec" % gCon['dec_vocab_size'])

    enc_vocab, _ = data_utils.initialize_vocabulary(enc_vocab_path)
    _, rev_dec_vocab = data_utils.initialize_vocabulary(dec_vocab_path)

    return sess, model, enc_vocab, rev_dec_vocab

if __name__ == '__main__':
    if len(sys.argv) - 1:
        gCon = get_config(sys.argv[1])
    else:
       
        gCon = get_config()

    print('\n>> Mode : %s\n' %(gCon['mode']))

    if gCon['mode'] == 'train':
       
        train()
    elif gCon['mode'] == 'test':
        
        decode()
    else:
        
        print('Please set mode to train or test')

