import ffmpeg
import pysrt
import os

def create_subs_srt(subs_path, subs_list):    
    #create empty file to white
    with open(subs_path, 'w') as emptyfile: pass
    subs = pysrt.open(subs_path)
    for index, sub in enumerate(subs_list):
        start_time = pysrt.SubRipTime.from_ordinal(sub[1] * 1000)
        end_time = pysrt.SubRipTime.from_ordinal(sub[2] * 1000)
        next_sub = pysrt.SubRipItem(index=1, text=sub[0], start=start_time, end=end_time)
        subs.append(next_sub)
    subs.save(subs_path, encoding='utf-8')

def add_subs(input_path, subs_path, output_path):
    video = ffmpeg.input(input_path)
    audio = video.audio
    ffmpeg.concat(video.filter("subtitles", subs_path), audio, v=1, a=1).output(output_path).run()

if __name__ == '__main__':
    subs_path  = 'test_subs.srt'
    input_path = 'test_video.mp4'
    output_path = 'test_video_subs.mp4'
    subs_list = [("These are some test subs", 1, 2), ('These are some test subs too!', 6, 10)]
    create_subs_srt(subs_path, subs_list)
    add_subs(input_path, subs_path, output_path)
