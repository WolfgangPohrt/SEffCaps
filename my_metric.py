from sqlite3 import Timestamp
import pysrt 
import string
from metrics.der import build_cost_matrix
from metrics.eval_metrics import evaluate_metrics_from_lists
from metrics.sed import sed_metrics

def to_seconds(timestamp):
    return float(timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds)

def ref_srt_to_timestamps(srt_path):
    subs = pysrt.open(srt_path)
    seff_caps = []
    for sub in subs:
        if '[' in sub.text or '(' in sub.text:
            sub.text = sub.text[1:-1]
            seff_caps.append(sub)
    return seff_caps

def hyp_srt_to_timestamps(srt_path):
    subs = pysrt.open(srt_path)
    return subs


def subs2dcase(subs, t):
    path = f'evaluation_data/{t}.txt'
    with open(path, 'w') as f:
        for sub in subs:
            onset = to_seconds(sub.start)
            offset = to_seconds(sub.end)
            text = 'caption'
            f.write(f'{onset},{offset},{text}\n')

def get_duration(ref_timestamps):
    """
    Get duration of Movie/Video.
    
    !Currently assuming duration
    is equal to the end time of the
    last subtitle.!
    
    !!!todo get accual duration todo!!!
    """
    return to_seconds(ref_timestamps[-1].end)

def _sentence_process(sentence, add_specials=True):

    # transform to lower case
    sentence = sentence.lower()

    # remove any punctuation and double space
    sentence = sentence.translate(str.maketrans('', '', string.punctuation))
    
    sentence = sentence.replace('\n', ' ')
    return sentence

def prep_timestamps(subs, duration):
    """
    Reforamt subs from pysrt format.
    """
    start_time_sp = .0
    timestamps_final = []
    for i, sub in enumerate(subs):
        start_time_nosp = to_seconds(sub.start)
        end_time_nosp = to_seconds(sub.end)
        end_time_sp = start_time_nosp
        text = _sentence_process(sub.text)
        timestamps_final.append(('sp', start_time_sp, end_time_sp))
        timestamps_final.append(('nosp', start_time_nosp, end_time_nosp, text))
        start_time_sp = end_time_nosp   
    timestamps_final.append(('sp', start_time_sp, duration))
    return timestamps_final

def matched_captions(ref, hyp):
    """
    Find the overlaping ref and hyp captions
    and return a list of tuples (ref,hyp)
    """
    matched_caps = []
    for h in hyp:
        s_h, e_h, cap_h = h[1:]
        for r in ref:
            s_r, e_r, cap_r = r[1:]
            if (s_h >= s_r and s_h < e_r) or (e_h <= e_r and e_h > s_r) or (s_h <= s_r and e_h >= e_r):
                matched_caps.append((cap_r, cap_h))
    return matched_caps

def calculate_spider(matched_caps):
    hyp = [t[0] for t in matched_caps] 
    ref = [t[1] for t in matched_caps]
    ref = [[r] for r in ref]
    ids = [i for i in range(1,len(matched_caps)-1)]
    print(matched_caps)
    m, _ = evaluate_metrics_from_lists(hyp, ref, ids)
    print(m)
    return m['SPICE']

def my_metric(ref_srt, hyp_srt):
    """
    cost_matrix: a 2-dim numpy array:
        elem (0,0) --> correct nosp
        elem (0,1) --> miss
        elem (1,0) --> false alarm
        elem (1,1) --> correct sp
     """
    ref = ref_srt_to_timestamps(ref_srt)
    duration = (get_duration(ref))
    ref = prep_timestamps(ref, duration)
    hyp = hyp_srt_to_timestamps(hyp_srt)
    print([h.start.minutes * 60 + h.start.seconds for h in hyp])
    hyp = prep_timestamps(hyp, duration)
    print(ref)
    print(hyp)
    cost_matrix = build_cost_matrix(ref, hyp)

    ref_nosp = [r for r in ref if r[0] == 'nosp']
    hyp_nosp = [h for h in hyp if h[0] == 'nosp']
    matched_caps = matched_captions(ref_nosp, hyp_nosp)
    spider = calculate_spider(matched_caps)

    total_ref_time = cost_matrix[1,0] + cost_matrix[1,1]
    confusion = cost_matrix[0,0] * (1 - spider)
    print(f'false alarm: {cost_matrix[1,0]}, miss: {cost_matrix[0,1]}, confusion_time: {cost_matrix[0,0]} spice: {spider}')

    metric =( cost_matrix[1,0] + cost_matrix[0,1] + confusion) / duration
    return f'false alarm: {cost_matrix[1,0]}, miss: {cost_matrix[0,1]}, confusion_time: {cost_matrix[0,0]} ,spice: {spider}, metric: {metric}'


if __name__ == '__main__':


    # ref_srt = '/home/theokouz/src/tmp/SEffCaps/Hostage.2021.720p.WEBRip.800MB.x264-GalaxyRG-HI.srt'
    ref_srt = '/home/theokouz/src/tmp/SEffCaps/test_metric.srt'
    hyp_srt = '/home/theokouz/src/tmp/SEffCaps/test_metric_hyp.srt'

    # hyp = hyp_srt_to_timestamps(hyp_srt)
    # ref = ref_srt_to_timestamps(ref_srt)
    # subs2dcase(ref, 'ref')
    # subs2dcase(hyp, 'hyp')
    sed_metrics('ref.txt', 'hyp.txt')
