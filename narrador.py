import sounddevice as sd
from scipy.io.wavfile import write
from IPython.display import Audio, display
import whisper
from openai import OpenAI
from gtts import gTTS
import os
from openai import OpenAI



# Função para gravar áudio
def record(seconds=10, filename="request_audio.wav"):
    fs = 44100  # taxa de amostragem (Hz)
    print("Ouvindo...")
    audio = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
    sd.wait()  # espera terminar a gravação
    write(filename, fs, audio)  # salva em WAV
    print(f"Gravação salva em {filename}")
    return filename

# Grava 5 segundos de áudio
record_file = record(10)

# Reproduz o áudio dentro do notebook
display(Audio(record_file, autoplay=True))

# Carrega o modelo Whisper local
model = whisper.load_model("base")  # pode trocar por "small", "medium" ou "large"

# Transcreve o áudio gravado
result = model.transcribe(record_file)

texto_usuario = result["text"]

print("Transcrição:", texto_usuario)

# Conecta ao GPT usando a variável de ambiente
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

response = client.chat.completions.create(
    model="gpt-5.4-mini",  # pode usar outro modelo disponível
    messages=[
        {"role": "system", "content": "Você é um narrador de histórias criativo e envolvente."},
        {"role": "user", "content": texto_usuario}
    ]
)


historia = response.choices[0].message.content
print("História:", historia)

# Converte a continuação em áudio
tts = gTTS(historia, lang="pt")
tts.save("historia.mp3")
display(Audio("historia.mp3", autoplay=True))
