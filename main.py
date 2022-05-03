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

import numpy as np
import pandas as pd
import nltk
nltk.download('punkt') # one time execution
nltk.download('stopwords')
import re 
from nltk.tokenize import sent_tokenize
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
    f1=open("Transcript.txt","w+")
    f1.write(full_text)
    full_text
    text_lst=full_text.split(".")
    sentences = []
    for s in text_lst:
        sentences.append(sent_tokenize(s))
    sentences = [y for x in sentences for y in x]
    word_embeddings = {}
    f = open('glove.6B.100d.txt', encoding='utf-8')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        word_embeddings[word] = coefs
    f.close()


    # In[68]:


    clean_sentences = pd.Series(sentences).str.replace("[^a-zA-Z]", " ",regex=True)

    # make alphabets lowercase
    clean_sentences = [s.lower() for s in clean_sentences]


    # In[69]:


    from nltk.corpus import stopwords


    # In[70]:


    stop_words = stopwords.words('english')
    def remove_stopwords(sen):
        sen_new = " ".join([i for i in sen if i not in stop_words])
        return sen_new
    clean_sentences = [remove_stopwords(r.split()) for r in clean_sentences]
    clean_sentences


    # In[71]:


    sentence_vectors = []
    for i in clean_sentences:
        if (len(i) != 0):
            v = sum([word_embeddings.get(w, np.zeros((100,))) for w in i.split()])/(len(i.split())+0.001)
        else:
            v = np.zeros((100,))
        sentence_vectors.append(v)
    sentence_vectors


    # In[72]:


    # similarity matrix
    sim_mat = np.zeros([len(sentences), len(sentences)])
    sim_mat


    # In[73]:


    from sklearn.metrics.pairwise import cosine_similarity


    # In[74]:


    for i in range(len(sentences)):
        for j in range(len(sentences)):
            if i != j:
                sim_mat[i][j] = cosine_similarity(sentence_vectors[i].reshape(1,100), sentence_vectors[j].reshape(1,100))[0,0]


    # In[75]:


    sim_mat


    # In[76]:


    import networkx as nx

    nx_graph = nx.from_numpy_array(sim_mat)
    scores = nx.pagerank(nx_graph)


    # In[77]:


    ranked_sentences = sorted(((scores[i],s) for i,s in enumerate(sentences)), reverse=True)


    # In[78]:

    sumy=""
    for i in range(10):
        sumy+=ranked_sentences[i][1]
        print(ranked_sentences[i][1])


    # In[ ]:








    
    return sumy


    return full_text
    


if __name__ == "__main__":
    app.run()