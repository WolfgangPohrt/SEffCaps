
from sqlite3 import Timestamp
import pysrt 
from der import DER, build_cost_matrix
from eval_metrics import evaluate_metrics_from_lists

def to_seconds(timestamp):
    return float(timestamp.minutes * 60 + timestamp.seconds)

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

def get_duration(ref_timestamps):
    return to_seconds(ref_timestamps[-1].end)



def prep_timestamps(subs, duration):
    
    start_time_sp = .0
    timestamps_final = []
    for i, sub in enumerate(subs):
        start_time_nosp = to_seconds(sub.start)
        end_time_nosp = to_seconds(sub.end)
        end_time_sp = start_time_nosp
        timestamps_final.append(('sp', start_time_sp, end_time_sp))
        timestamps_final.append(('nosp', start_time_nosp, end_time_nosp, sub.text))
        start_time_sp = end_time_nosp   
    timestamps_final.append(('sp', start_time_sp, duration))
    return timestamps_final

def matched_captions(ref, hyp):
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
    ref = [[h]*5 for h in hyp]
    ids = [i for i in range(1,len(matched_caps)-1)]
    m, _ = evaluate_metrics_from_lists(hyp, ref, ids)
    return m['SPIDEr']

def my_metric(cost_matrix):
    """
    cost_matrix: a 2-dim numpy array:
        elem (0,0) --> correct nosp
        elem (0,1) --> miss
        elem (1,0) --> false alarm
        elem (1,1) --> correct sp
     """
    spider = 0.5
    total_ref_time = cost_matrix[1,0] + cost_matrix[1,1]
    confusion = cost_matrix[0,0] * (1 - spider)
    metric = (cost_matrix[1,0] + cost_matrix[0,1] + confusion) / total_ref_time
    return metric

ref = ref_srt_to_timestamps('/home/theokouz/src/audio_desc_pipeline/subs_scraping/srt_files/Blade.Runner.Black.Lotus.S01E07.WEBRip.x264-ION10.srt')
duration = (get_duration(ref))
ref = prep_timestamps(ref, duration)
hyp = hyp_srt_to_timestamps('/home/theokouz/src/audio_desc_pipeline/test_subs.srt')
hyp = prep_timestamps(hyp, duration)
# hyp = [('nosp', 0., 0.5), ('sp', 0.5, 1248.)]

ref_nosp = [r for r in ref if r[0] == 'nosp']
hyp_nosp = [h for h in hyp if h[0] == 'nosp']




# total_time = sum([e-s for _, s, e, txt in nosp_intervals])
cost_matrix = build_cost_matrix(ref, hyp)

# r = ['hitler as kalki']
# h = [['hitler as kalki', 'hitler as kalki', 'hitler as kalki', 'hitler as kalki', 'hitler as kalki']]
# m = evaluate_metrics_from_lists(r,h,[1])
# print(m)

matched_caps = matched_captions(ref_nosp, hyp_nosp)
spider = calculate_spider(matched_caps)
print(spider)
# print(my_metric(cost_matrix))
# print(ref)
# print(hyp)
