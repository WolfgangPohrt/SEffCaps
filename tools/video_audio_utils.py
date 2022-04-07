from glob import glob
from pytube import YouTube
import os
def download_video(url, workingdir, basename):
    """
    Download video from Youtube and save it
    to workingdir.

    url: link to Youtube video
    workingdir: directory to save it.
    basename: rename downloaded video to basename.mp4 
    """
    yt = YouTube(url) 
    video = yt.streams.filter(file_extension='mp4').first()
    downloaded = video.download(output_path=workingdir)
    base, ext = os.path.splitext(downloaded)
    renamed_video = os.path.join(workingdir, f'{basename}.mp4')
    os.rename(downloaded, renamed_video)
    return renamed_video

def extract_audio(video_path):
    """
    Extract audio in wav format from
    video.
    """
    base, ext = os.path.splitext(video_path)
    audio_path = f'{base}.wav'
    command = f"ffmpeg -i {video_path} -ab 160k -ac 2 -ar 44100 -vn {audio_path}"
    os.system(command)
    return audio_path

def segment_audio(audio_path, seg_dur):
    """
    Segment wav file to 10s segments
    and save them to the same folder.
    """
    base, ext = os.path.splitext(audio_path)
    command = f"ffmpeg -y -i {audio_path} -f segment -segment_time {seg_dur} -c copy {base}%03d.wav"
    os.system(command)
    audio_paths = glob(f'{base}0*.wav')
    return audio_paths
