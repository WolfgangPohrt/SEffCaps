from collections import Counter

def remove_sents(caps_timestamps):
    cleaned = []
    for cap in caps_timestamps:
        if 'speaking' not in cap[0] and check_for_repeating_bigram(cap[0].split(' ')) == False:
            cleaned.append(cap)
    return cleaned


def check_for_repeating_bigram(tokens):
    l = []
    for i in range(len(tokens) - 1):
        l.append((tokens[i], tokens[i+1]))
    for i in Counter(l).values():
        if i > 1:
            return True
    return False
    

l = [['a music is played while a music is played', 0, 5], ['a music is played followed by a low hum', 10, 15], ['a music is played followed by a man speaking', 40, 45], ['water is splashing and gurgling and a motor is running', 115, 120], ['a man speaking followed by another man speaking then a loud pop', 120, 125], ['a man speaking', 145, 150], ['a man speaking followed by a loud buzzing', 150, 155], ['silence followed by silence', 165, 170], ['a small click followed by a loud buzzing', 230, 235], ['water is trickling and gurgling and an electronic device is playing', 235, 240], ['a man speaking followed by a brief silence', 240, 245], ['a man speaks followed by a loud buzzing', 255, 260], ['a man speaking', 300, 305], ['a door is closed', 310, 315], ['a man speaking', 325, 330]]
print(remove_sents(l))
