import torch
print("torch", torch.__version__, "cuda", torch.version.cuda, "available", torch.cuda.is_available())
print("gpu", torch.cuda.get_device_name(0), "cudnn", torch.backends.cudnn.version())
import ctranslate2
print("ctranslate2", ctranslate2.__version__, "cuda devices", ctranslate2.get_cuda_device_count())
import faster_whisper; print("faster_whisper", faster_whisper.__version__)
try:
    from chatterbox.mtl_tts import ChatterboxMultilingualTTS, SUPPORTED_LANGUAGES
    print("chatterbox ML import OK; ms supported:", "ms" in SUPPORTED_LANGUAGES, "| id supported:", "id" in SUPPORTED_LANGUAGES)
except Exception as e:
    print("chatterbox import FAILED:", type(e).__name__, e)
