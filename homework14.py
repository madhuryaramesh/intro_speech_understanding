import gtts
import speech_recognition as sr
import librosa
import soundfile as sf
import os


def synthesize(text, lang, filename):
    '''
    Use gtts.gTTS(text=text, lang=lang) to synthesize speech,
    then save it to filename.
    '''
    tts = gtts.gTTS(text=text, lang=lang)
    tts.save(filename)


def make_a_corpus(texts, languages, filenames):
    '''
    Create speech files, convert MP3 to WAV,
    then recognize each file.
    '''
    recognizer = sr.Recognizer()
    recognized_texts = []

    for text, lang, root in zip(texts, languages, filenames):

        mp3file = root + ".mp3"
        wavfile = root + ".wav"

        # Create MP3
        synthesize(text, lang, mp3file)

        # Convert MP3 -> WAV
        waveform, Fs = librosa.load(mp3file, sr=None)
        sf.write(wavfile, waveform, Fs)

        # Recognize speech
        with sr.AudioFile(wavfile) as source:
            audio = recognizer.record(source)

        try:
            recognized = recognizer.recognize_google(audio, language=lang)
        except Exception:
            recognized = ""

        recognized_texts.append(recognized)

    return recognized_texts