import pysrt 
from tools.utils import _sentence_process
from metrics.eval_metrics import evaluate_metrics_from_lists
from metrics.sed import sed_metrics

def to_seconds(timestamp):
    return float(timestamp.hours * 3600 + timestamp.minutes * 60 + timestamp.seconds)


def hyp_srt_to_dcase(srt_path,dcase_path):
    subs = pysrt.open(srt_path)
    with open(dcase_path, 'w') as f:
        hyp = []
        for sub in subs:
            onset = to_seconds(sub.start)
            offset = to_seconds(sub.end)
            text = _sentence_process(sub.text)
            hyp.append((onset,offset,text))
            f.write(f'{onset},{onset},caption\n')
        return hyp

def ref_srt_to_dcase(srt_path,dcase_path):
    subs = pysrt.open(srt_path)
    with open(dcase_path, 'w') as f:
        ref = []
        for sub in subs:
            if '[' in sub.text or '(' in sub.text:
                sent = sub.text[1:-1]
                sent = _sentence_process(sent)
                onset = to_seconds(sub.start)
                ofset = to_seconds(sub.end)
                ref.append((onset, ofset, sent))
                f.write(f'{onset},{onset},caption\n')
    return ref

def matched_captions(ref, hyp):
    """
    Find the overlaping ref and hyp captions
    and return a list of tuples (ref,hyp)
    """
    matched_caps = []
    for h in hyp:
        s_h, e_h, cap_h = h
        for r in ref:
            s_r, e_r, cap_r = r
            if (s_h >= s_r and s_h < e_r) or (e_h <= e_r and e_h > s_r) or (s_h <= s_r and e_h >= e_r):
                matched_caps.append((cap_r, cap_h))
    return matched_caps


def calculate_spice(matched_caps):
    hyp = [t[0] for t in matched_caps] 
    ref = [t[1] for t in matched_caps]
    ref = [[r] for r in ref]
    ids = [i for i in range(1,len(matched_caps)-1)]
    print(matched_caps)
    m, _ = evaluate_metrics_from_lists(hyp, ref, ids)
    print(m)
    return m['SPICE']



def my_sed(ref_srt, hyp_srt):
    dcase_hyp = '/home/theokouz/src/tmp/SEffCaps/evaluation_data/hyp.txt'
    dcase_ref = '/home/theokouz/src/tmp/SEffCaps/evaluation_data/ref.txt'
    hyp = hyp_srt_to_dcase(hyp_srt, dcase_hyp)
    ref = ref_srt_to_dcase(ref_srt, dcase_ref)
    matched = matched_captions(ref, hyp)
    spice = calculate_spice(matched)
    event_based, seg_based = sed_metrics(dcase_ref, dcase_hyp, spice)
    event_based['error_rate']['error_rate'] += (1-spice)
    event_based['error_rate']['substitution_rate'] = (1-spice)
    return event_based
if __name__ == '__main__':
    ref_srt = '/home/theokouz/src/tmp/SEffCaps/test_metric.srt'
    hyp_srt = '/home/theokouz/src/tmp/SEffCaps/test_metric_hyp.srt'
    event_based = my_sed(ref_srt, hyp_srt)
    print(event_based)



    