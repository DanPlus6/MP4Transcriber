from os import path, environ, remove
from faster_whisper import WhisperModel
import torch

import ffmpegio
base_dir = path.dirname(path.abspath(__file__))

from time import time

def get_tempdir():
    for var in ("TMPDIR", "TEMP", "TMP"):
        attemptPath = environ.get(var)
        if attemptPath and path.isdir(attemptPath):
            return attemptPath

    return r"/tmp"


def extract_audio(input_path,output_path):
    print("[+] Beginning audio extraction...")


    try:
        ffmpegio.transcode(input_path, output_path, overwrite=True, acodec='pcm_s16le', format='wav')
    except Exception as e:
        print(f"[-] Error during audio extraction: {e}")
        print("[+] Re-attempting audio extraction.")
        try:
            ffmpegio.transcode(input_path, output_path, overwrite=True, acodec='pcm_s16le', format='wav')
        except:
            print(f"[-] Error occured during audio extraction, terminating.")
            remove(output_path)
            exit()

    print("[+] Finished audio extraction.")

class STTProcessor:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.compute_type = "float32" if self.device == "cuda" else "int8"
        try:
            self.model = WhisperModel("large", device=self.device, compute_type=self.compute_type)
        except Exception as e:
            print(f"[-] Error loading model: {e}")
            self.model = None

    def transcribe(self, audio_file_path, language="zh", task="translate", beam_size=5):
        if not self.model:
            print("[-] Model is not loaded.")
            return None

        if not path.exists(audio_file_path):
            print(f"[-] Error: Audio file not found at {audio_file_path}")
            return None

        print("[+] Transcribing audio...")

        try:
            segments, info = self.model.transcribe(
                audio_file_path,
                language=language,
                task=task,
                beam_size=beam_size
            )
            transcribed_text = "".join(segment.text for segment in segments)
            print("[+] Transcription complete!")
            return transcribed_text
        except Exception as e:
            print(f"[-] Error during transcription: {e}")
            print("[+] Re-attempting transcription.")
            try:
                segments, info = self.model.transcribe(
                    audio_file_path,
                    language=language,
                    task=task,
                    beam_size=beam_size
                )
                transcribed_text = "".join(segment.text for segment in segments)
                print("[+] Transcription complete!")
            except:
                return None


    def transcribe_and_save(self, audio_path, output_txt_path, **kwargs):
        transcription = self.transcribe(audio_path, **kwargs)

        if transcription:
            with open(output_txt_path, "w", encoding="utf-8") as script:
                script.write(transcription)
            return
        else:
            print(f"[-] Error occured during transcription, terminating.")
            exit()



if __name__ == '__main__':
    openPath = rf'working/working.mp4'
    tmp = path.join(get_tempdir(),(path.basename(openPath)[:-4]+'.wav'))
    extract_audio(openPath,tmp)

    savePath = rf'working/product.txt'
    startTime = time()
    STTProcessor().transcribe_and_save(tmp,savePath)
    remove(tmp)
    endTime = time()
    timeTaken = round((endTime-startTime),2)
    print("=======================================Total taken time:",timeTaken,"=======================================")

