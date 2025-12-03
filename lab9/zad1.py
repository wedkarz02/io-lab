import nltk

with open("bbc_waffles.txt", "r") as f:
    content = f.read()

tokens = nltk.word_tokenize(content, language="english")
print(f"token count: {len(tokens)}")

stop_words = nltk.corpus.stopwords.words("english")
stop_words.extend([".", ",", "’", ":"])

filtered_tokens = [token for token in tokens if token not in stop_words]

print(f"filtered token count: {len(filtered_tokens)}")
print(filtered_tokens)
