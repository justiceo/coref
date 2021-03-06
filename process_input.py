# Lint as: python3
from bert import tokenization
import json

genre = "nw"
# The Ontonotes data for training the model contains text from several sources
# of very different styles. You need to specify the most suitable one out of:
# "bc": broadcast conversation
# "bn": broadcast news
# "mz": magazine
# "nw": newswire
# "pt": Bible text
# "tc": telephone conversation
# "wb": web data

model_name = "spanbert_base"
# The fine-tuned model to use. Options are:
# bert_base
# spanbert_base
# bert_large
# spanbert_large

text = [
"John has a red car, he drive it around the block every morning to Sarah who owns a store around the block."]

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

with open("sample.in.json", 'w') as out:
    json.dump(data, out, sort_keys=True)

