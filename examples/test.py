from coml.context import reproduce_function_parameters


def foo():
    x = reproduce_function_parameters(
        1,
        2,
        3
    )

    d = {"a": 1, "b": 2}
    a = {"b": 3, "c": d}
    y = reproduce_function_parameters(1, 2, a=3) or 3

    R = reproduce_function_parameters
    z = R(a, **d)



foo()


# import nltk
# from nltk.corpus import stopwords 

# def ProperNounExtractor(text):
    
#     print('PROPER NOUNS EXTRACTED :')
    
#     sentences = nltk.sent_tokenize(text)
#     for sentence in sentences:
#         words = nltk.word_tokenize(sentence)
#         # words = [word for word in words if word not in set(stopwords.words('english'))]
#         tagged = nltk.pos_tag(words)
#         for (word, tag) in tagged:
#             if tag == 'NNP': # If the word is a proper noun
#                 print(word)

# text =  "Rohan is a wonderful player. He was born in India. He is a fan of the movie Wolverine. He has a dog named Bruno."
# # Calling the ProperNounExtractor function to extract all the proper nouns from the given text. 
# ProperNounExtractor(text)