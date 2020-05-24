from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util
import time
from bert import tokenization


# The fine-tuned model to use. Options are:
# bert_base
# spanbert_base
# bert_large
# spanbert_large
model_name = "spanbert_base"

def process_input():
    # The Ontonotes data for training the model contains text from several sources
    # of very different styles. You need to specify the most suitable one out of:
    # "bc": broadcast conversation
    # "bn": broadcast news
    # "mz": magazine
    # "nw": newswire
    # "pt": Bible text
    # "tc": telephone conversation
    # "wb": web data
    genre = "nw"

    text = [
    '"I voted for Obama because he was most aligned with my values", she said.']

    data = {
        'doc_key': genre,
        'sentences': [["[CLS]"]],
        'speakers': [["[SPL]"]],
        'clusters': [],
        'sentence_map': [0],
        'subtoken_map': [0],
    }

    # Determine Max Segment
    max_segment = None
    for line in open('experiments.conf'):
        if line.startswith(model_name):
            max_segment = True
        elif line.strip().startswith("max_segment_len"):
            if max_segment:
                max_segment = int(line.strip().split()[-1])
                break

    tokenizer = tokenization.FullTokenizer(vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False)
    subtoken_num = 0
    for sent_num, line in enumerate(text):
        raw_tokens = line.split()
        tokens = tokenizer.tokenize(line)
        if len(tokens) + len(data['sentences'][-1]) >= max_segment:
            data['sentences'][-1].append("[SEP]")
            data['sentences'].append(["[CLS]"])
            data['speakers'][-1].append("[SPL]")
            data['speakers'].append(["[SPL]"])
            data['sentence_map'].append(sent_num - 1)
            data['subtoken_map'].append(subtoken_num - 1)
            data['sentence_map'].append(sent_num)
            data['subtoken_map'].append(subtoken_num)

        ctoken = raw_tokens[0]
        cpos = 0
        for token in tokens:
            data['sentences'][-1].append(token)
            data['speakers'][-1].append("-")
            data['sentence_map'].append(sent_num)
            data['subtoken_map'].append(subtoken_num)

            if token.startswith("##"):
                token = token[2:]
            if len(ctoken) == len(token):
                subtoken_num += 1
                cpos += 1
                if cpos < len(raw_tokens):
                    ctoken = raw_tokens[cpos]
            else:
                ctoken = ctoken[len(token):]

    data['sentences'][-1].append("[SEP]")
    data['speakers'][-1].append("[SPL]")
    data['sentence_map'].append(sent_num - 1)
    data['subtoken_map'].append(subtoken_num - 1)

    return data

def write_data(file_name, data):      
    with open(file_name, "w") as output_file:
      output_file.write(json.dumps(data))
      output_file.write("\n")

def process_output(output):
  comb_text = [word for sentence in output['sentences'] for word in sentence]

  def convert_mention(mention):
      start = output['subtoken_map'][mention[0]]
      end = output['subtoken_map'][mention[1]] + 1
      nmention = (start, end)
      mtext = ''.join(' '.join(comb_text[mention[0]:mention[1]+1]).split(" ##"))
      return (nmention, mtext)

  clusters = []
  print('Clusters:')
  for cluster in output['predicted_clusters']:
      mapped = []
      for mention in cluster:
          mapped.append(convert_mention(mention))
      clusters.append(mapped)
  return clusters

  # print('\nMentions:')
  # for mention in output['top_spans']:
  #     if tuple(mention) in seen:
  #         continue
  #     print(convert_mention(mention), end=",\n")



def run():
  start = time.time()
  config = util.initialize_from_env(model_name)
  log_dir = config["log_dir"]

  model = util.get_model(config)
  print("\n***loaded model...", time.time()-start)
  start = time.time()

  saver = tf.train.Saver()

  with tf.Session() as session:
    model.restore(session)
    print("\n***restored session...", time.time()-start)
    start = time.time()

    example = process_input()
    print("\n***process_input...", time.time()-start)
    start = time.time()

    tensorized_example = model.tensorize_example(example, is_training=False)
    feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
    _, _, _, top_span_starts, top_span_ends, top_antecedents, top_antecedent_scores = session.run(model.predictions, feed_dict=feed_dict)
    print("\n***session.run...", time.time()-start)
    start = time.time()

    predicted_antecedents = model.get_predicted_antecedents(top_antecedents, top_antecedent_scores)

    example["predicted_clusters"], _ = model.get_predicted_clusters(top_span_starts, top_span_ends, predicted_antecedents)

    return process_output(example)

print(run())