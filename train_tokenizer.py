import ruamel_yaml as yaml
from dataset import create_dataset
from tokenizers import BertWordPieceTokenizer
from transformers import BertTokenizer
import json

my_vocab_size = 30522
my_limit_alphabet = 6000
my_special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]
user_defined_symbols = ['[BOS]','[EOS]']
unused_token_num = 200
unused_list = ['[unused{}]'.format(n) for n in range(unused_token_num)]
user_defined_symbols = user_defined_symbols + unused_list
my_special_tokens = my_special_tokens + user_defined_symbols

config = yaml.load(open('./configs/Retrieval.yaml', 'r'), Loader=yaml.Loader)

train_dataset, val_dataset, test_dataset = create_dataset('re', config)

texts = train_dataset.text
with open('./corpus.txt', 'w') as f:
    for data in texts:
        f.write(data+'\n')

tokenizer = BertWordPieceTokenizer(
    clean_text=True,
    handle_chinese_chars=True,
    strip_accents=True,
    lowercase=True,
    wordpieces_prefix="##"
)

tokenizer.train(
    files='./corpus.txt',
    limit_alphabet=my_limit_alphabet,
    vocab_size=my_vocab_size,
    min_frequency=1,
    show_progress=True,
    special_tokens=my_special_tokens
)

tokenizer.save("./ch-{}-wpm-{}-pretty".format(my_limit_alphabet, my_vocab_size),True)

vocab_path = "./ch-6000-wpm-30522-pretty"

vocab_file = './data/wpm-vocab-all.txt'
f = open(vocab_file,'w',encoding='utf-8')
with open(vocab_path) as json_file:
    json_data = json.load(json_file)
    for item in json_data["model"]["vocab"].keys():
        f.write(item+'\n')

    f.close()

vocab_path = "./data/wpm-vocab-all.txt"
tokenizer = BertTokenizer(vocab_file=vocab_path, do_lower_case=True)

tokenizer.save_pretrained('./my_tokenizer/')

