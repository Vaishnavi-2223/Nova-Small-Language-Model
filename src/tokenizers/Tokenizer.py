from datasets import load_dataset

#Load dataset
dataset=load_dataset("Salesforce/wikitext",
                     "wikitext-2-raw-v1")
text="\n".join(dataset["train"]["text"])


#build vocab
chars=sorted(list(set(text)))
vocab_size=len(chars)

print("Vocabulary size: ", vocab_size)
print("Vocabulary: ","".join(chars))

#create mapping
# Create character-to-integer mapping, i.e converts text into numbers
stoi = {ch: i for i, ch in enumerate(chars)}

# Create integer-to-character mapping, i.e converts generated numbers back to text
itos = {i: ch for i, ch in enumerate(chars)}

# Unknown character token
unk_id = len(chars)

stoi["<UNK>"] = unk_id
itos[unk_id] = "<UNK>"

#encodeing
def encode(text):
    return[stoi.get(ch,unk_id) for ch in text]

#decoding
def decode(tokens):
    return "".join(
        itos[i] if itos[i] != "<UNK>" else "�"
        for i in tokens
    )

#test
sample_text="Hello World"
encoded=encode(sample_text)
decoded=decode(encoded)

print("\nOriginal text: ", sample_text)
print("Encoded text: ", encoded)
print("Decoded text: ", decoded)

print("\nTokenizer test successful!")


print("\nCharacter to Integer:")
print(stoi)

print("\nInteger to Character:")
print(itos)

my_sentence = "Hello Korea I'm Nabi"

result = encode(my_sentence)

print(result)
my_sentence = "Hello Korea I'm Nabi"

result = encode(my_sentence)
reconstructed = decode(result)

print("Original:", my_sentence)
print("Reconstructed:", reconstructed)