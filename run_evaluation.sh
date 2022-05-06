for movie in evaluation_data/test_movies/*.wav; do
	b_name=`basename $movie .wav`;
	srt=evaluation_data/test_srts/${b_name}.srt
	echo $srt
	echo $movie
	cp $movie downloaded_audio/test.wav
	python3 main.py --audiodir_path downloaded_audio/ --ref_srt $srt
       	rm downloaded_audio/*	
done

