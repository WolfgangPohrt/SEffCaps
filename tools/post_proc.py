


def remove_speaking(caps_timestamps):
    stop_words = ['speaking', 'speaks']
    clean_ct = []
    for ct in caps_timestamps:
        caption = ct[0].split(' ')
        if len(list(set(caption) - set(stop_words))) == len(caption):
            clean_ct.append(ct) 
    return clean_ct