import os
from app import app
import urllib.request
from flask import Flask, flash, request, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import shutil
import moviepy.editor as mp
import subprocess
import speech_recognition as sr 
from pydub import AudioSegment
from pydub.silence import split_on_silence
import time

r = sr.Recognizer()

async def get_large_audio_transcription(path):
   
   
    sound = AudioSegment.from_wav(path)  
    chunks = split_on_silence(sound,
        
        min_silence_len = 500,
        
        silence_thresh = sound.dBFS-14,
       
        keep_silence=500,
    )
    folder_name = "audio-chunks"
    # create a directory to store the audio chunks
    if not os.path.isdir(folder_name):
        os.mkdir(folder_name)
    whole_text = ""
    # process each chunk 
    for i, audio_chunk in enumerate(chunks, start=1):
       
        chunk_filename = os.path.join(folder_name, f"chunk{i}.wav")
        audio_chunk.export(chunk_filename, format="wav")
        # recognize the chunk
        with sr.AudioFile(chunk_filename) as source:
            audio_listened = r.record(source)
            # try converting it to text
            try:
                text = r.recognize_google(audio_listened)
            except sr.UnknownValueError as e:
                print("Error:", str(e))
            else:
                text = f"{text.capitalize()}. "
                #print(chunk_filename, ":", text)
                whole_text += text
    # return the text for all chunks detected
    return whole_text

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/', methods=['POST'])
async def upload_video():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        flash('No image selected for uploading')
        return redirect(request.url)
    else:
        filename = secure_filename(file.filename)
        file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        #print('upload_video filename: ' + filename)
        transc=await modelCall(file)
        flash('Video successfully uploaded and displayed below')
        
        return render_template('upload.html', filename=filename,transc=transc)

@app.route('/display/<filename>')
async def display_video(filename):
    #print('display_video filename: ' + filename)
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

async def modelCall(file):
    ftxt="static/uploads/"+file.filename
    time.sleep(5)
    with open(f'{"static/uploads/"+file.filename}','rb') as buffer:
        shutil.copyfileobj(file.file,buffer)
        
    #time.sleep(10)
    clip = mp.VideoFileClip(ftxt)
    clip.audio.write_audiofile("static/uploads/"+"AIaudio.mp3")
    '''with open(r"static/files/AIaudio.mp3",'rb') as buffer:
        shutil.copyfileobj(r'static/uploads/AIaudio.mp3',buffer)'''
    sound = AudioSegment.from_mp3('static/uploads/AIaudio.mp3')
    sound.export("static/uploads/result.wav", format="wav")
    full_text=await get_large_audio_transcription("static/uploads/result.wav")
    return full_text
    


if __name__ == "__main__":
    app.run()