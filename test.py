from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
print(tokenizer.tokenize("YaoMing is a basketball player."))
print(tokenizer.tokenize("Bob is a basketball player."))
print(tokenizer.tokenize("[CLS]YaoMing[SEP]job[SEP]basketball player[SEP]"))
print(tokenizer.tokenize("[CLS] the weather is colder than yesterday"))
print(tokenizer.encode("[CLS] the weather is colder than yesterday"))
print(tokenizer.decode(tokenizer.encode("[CLS] the weather is colder than yesterday")))
print(tokenizer.tokenize("tall taller tallest"))
print(tokenizer.tokenize("cold colder coldest coldness"))
print(tokenizer.tokenize("token tokenize tokenization"))
print(tokenizer.tokenize("token tokenize tokenifdsgfdf"))
print(tokenizer.tokenize("writer driver singer teacher"))