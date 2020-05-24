# Usage: python process_output.py
import json

output = json.load(open("sample.out.json"))

comb_text = [word for sentence in output['sentences'] for word in sentence]

def convert_mention(mention):
    start = output['subtoken_map'][mention[0]]
    end = output['subtoken_map'][mention[1]] + 1
    nmention = (start, end)
    mtext = ''.join(' '.join(comb_text[mention[0]:mention[1]+1]).split(" ##"))
    return (nmention, mtext)

seen = set()
print('Clusters:')
for cluster in output['predicted_clusters']:
    mapped = []
    for mention in cluster:
        seen.add(tuple(mention))
        mapped.append(convert_mention(mention))
    print(mapped, end=",\n")

print('\nMentions:')
for mention in output['top_spans']:
    if tuple(mention) in seen:
        continue
    print(convert_mention(mention), end=",\n")

