from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import json

import tensorflow as tf
import util
import time
from bert import tokenization
from multiprocessing.managers import BaseManager
import queue


# The fine-tuned model to use. Options are:
# bert_base
# spanbert_base
# bert_large
# spanbert_large
model_name = "spanbert_base"


def process_input(tokenizer, max_segment, text):
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

    data = {
        "doc_key": genre,
        "sentences": [["[CLS]"]],
        "speakers": [["[SPL]"]],
        "clusters": [],
        "sentence_map": [0],
        "subtoken_map": [0],
    }

    subtoken_num = 0
    for sent_num, line in enumerate(text):
        raw_tokens = line.split()
        tokens = tokenizer.tokenize(line)
        if len(tokens) + len(data["sentences"][-1]) >= max_segment:
            data["sentences"][-1].append("[SEP]")
            data["sentences"].append(["[CLS]"])
            data["speakers"][-1].append("[SPL]")
            data["speakers"].append(["[SPL]"])
            data["sentence_map"].append(sent_num - 1)
            data["subtoken_map"].append(subtoken_num - 1)
            data["sentence_map"].append(sent_num)
            data["subtoken_map"].append(subtoken_num)

        ctoken = raw_tokens[0]
        cpos = 0
        for token in tokens:
            data["sentences"][-1].append(token)
            data["speakers"][-1].append("-")
            data["sentence_map"].append(sent_num)
            data["subtoken_map"].append(subtoken_num)

            if token.startswith("##"):
                token = token[2:]
            if len(ctoken) == len(token):
                subtoken_num += 1
                cpos += 1
                if cpos < len(raw_tokens):
                    ctoken = raw_tokens[cpos]
            else:
                ctoken = ctoken[len(token) :]

    data["sentences"][-1].append("[SEP]")
    data["speakers"][-1].append("[SPL]")
    data["sentence_map"].append(sent_num - 1)
    data["subtoken_map"].append(subtoken_num - 1)

    return data


def process_output(output):
    comb_text = [word for sentence in output["sentences"] for word in sentence]

    def convert_mention(mention):
        start = output["subtoken_map"][mention[0]]
        end = output["subtoken_map"][mention[1]] + 1
        nmention = (start, end)
        mtext = "".join(" ".join(comb_text[mention[0] : mention[1] + 1]).split(" ##"))
        return (nmention, mtext)

    clusters = []
    for cluster in output["predicted_clusters"]:
        mapped = []
        for mention in cluster:
            mapped.append(convert_mention(mention))
        clusters.append(mapped)
    
    return clusters


def run():
    start = time.time()
    config = util.initialize_from_env(model_name)
    log_dir = config["log_dir"]

    model = util.get_model(config)
    print("\n***loaded model...", time.time() - start)
    start = time.time()

    saver = tf.train.Saver()
    tokenizer = tokenization.FullTokenizer(
        vocab_file="cased_config_vocab/vocab.txt", do_lower_case=False
    )

    # Determine Max Segment
    max_segment = None
    for line in open("experiments.conf"):
        if line.startswith(model_name):
            max_segment = True
        elif line.strip().startswith("max_segment_len"):
            if max_segment:
                max_segment = int(line.strip().split()[-1])
                break

    with tf.Session() as session:
        model.restore(session)
        print("\n***restored session...", time.time() - start)
        start = time.time()

        sent_queue = queue.Queue()
        coref_queue = queue.Queue()
        BaseManager.register("sent_queue", callable=lambda: sent_queue)
        BaseManager.register("coref_queue", callable=lambda: coref_queue)
        m = BaseManager(address=("", 50000), authkey=b"random auth")
        m.start()

        shared_sent_queue = m.sent_queue()
        shared_coref_queue = m.coref_queue()

        print("Coref server is ready...")
        while True:
            text = [shared_sent_queue.get()]
            start = time.time()
            example = process_input(tokenizer, max_segment, text)

            tensorized_example = model.tensorize_example(example, is_training=False)
            feed_dict = {i: t for i, t in zip(model.input_tensors, tensorized_example)}
            (
                _,
                _,
                _,
                top_span_starts,
                top_span_ends,
                top_antecedents,
                top_antecedent_scores,
            ) = session.run(model.predictions, feed_dict=feed_dict)

            predicted_antecedents = model.get_predicted_antecedents(
                top_antecedents, top_antecedent_scores
            )

            example["predicted_clusters"], _ = model.get_predicted_clusters(
                top_span_starts, top_span_ends, predicted_antecedents
            )

            shared_coref_queue.put(process_output(example))
            print("Took...", time.time() - start)


run()
