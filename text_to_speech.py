from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from transformers import AutoProcessor, BarkModel
from scipy.io.wavfile import write as write_wav
import torch
import numpy as np
from io import BytesIO
from fastapi.responses import StreamingResponse

# Define router
router = APIRouter()

# Load Bark processor and model
processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

class AudioRequest(BaseModel):
	text_prompt: str
	language: str
	voice_name: str

@router.post("/generate-audio")
async def generate_audio(request: AudioRequest):
	# Extract values from the request
	text_prompt = request.text_prompt
	language = request.language
	voice_name = request.voice_name

	# Validate input parameters
	if not text_prompt or not language or not voice_name:
		raise HTTPException(status_code=400, detail="Missing or invalid parameters: 'text_prompt', 'language', and 'voice_name' are required.")

	# Validate language and voice name
	if language not in AVAILABLE_VOICES:
		raise HTTPException(status_code=400, detail="Language not supported")
	if voice_name not in AVAILABLE_VOICES[language]:
		raise HTTPException(status_code=400, detail="Voice preset not available for this language")

	# Convert voice name to preset ID using VOICE_NAME_MAP
	voice_preset = VOICE_NAME_MAP[voice_name]

	# Process the input text
	inputs = processor(text_prompt, voice_preset=voice_preset, return_tensors="pt")

	# Generate audio with attention mask and pad token id
	attention_mask = inputs.get("attention_mask")
	pad_token_id = processor.tokenizer.pad_token_id
	audio_array = model.generate(
		input_ids=inputs["input_ids"],
		attention_mask=attention_mask,
		pad_token_id=pad_token_id
	)

	# Convert to numpy and ensure it's in the right format
	audio_array = audio_array.cpu().numpy().squeeze()
	sample_rate = model.generation_config.sample_rate

	# Save to buffer
	wav_buffer = BytesIO()
	write_wav(wav_buffer, sample_rate, np.int16(audio_array * 32767))  # Scales to 16-bit PCM
	wav_buffer.seek(0)

	# Set filename in Content-Disposition header based on voice_name
	headers = {
		"Content-Disposition": f"attachment; filename={voice_name}.wav"
	}

	return StreamingResponse(wav_buffer, media_type="audio/wav", headers=headers)

@router.get("/available-voices")
async def get_available_voices():
	return AVAILABLE_VOICES

# Define available voices with friendly names
AVAILABLE_VOICES = {
	"English": ["Bob", "Alice", "Charlie", "Diana", "Edward", "Fiona", "George", "Hannah", "Ian", "Jane"],
	"Spanish": ["Carlos", "Maria"],
	"French": ["Pierre", "Chloe"],
	"German": ["Hans", "Greta"],
	"Italian": ["Luca", "Sofia"],
	"Japanese": ["Hiro", "Yuki"],
	"Korean": ["Minho", "Jiwoo"],
	"Polish": ["Jan", "Anna"],
	"Portuguese": ["Miguel", "Ines"],
	"Russian": ["Ivan", "Olga"],
	"Turkish": ["Emre", "Ayse"],
	"Chinese": ["Li Wei", "Xiao Hong"]
}

# Mapping friendly names to actual voice preset IDs
VOICE_NAME_MAP = {
	"Bob": "v2/en_speaker_0",
	"Alice": "v2/en_speaker_1",
	"Charlie": "v2/en_speaker_2",
	"Diana": "v2/en_speaker_3",
	"Edward": "v2/en_speaker_4",
	"Fiona": "v2/en_speaker_5",
	"George": "v2/en_speaker_6",
	"Hannah": "v2/en_speaker_7",
	"Ian": "v2/en_speaker_8",
	"Jane": "v2/en_speaker_9",
	"Carlos": "v2/es_speaker_0",
	"Maria": "v2/es_speaker_1",
	"Pierre": "v2/fr_speaker_0",
	"Chloe": "v2/fr_speaker_1",
	"Hans": "v2/de_speaker_0",
	"Greta": "v2/de_speaker_1",
	"Luca": "v2/it_speaker_0",
	"Sofia": "v2/it_speaker_1",
	"Hiro": "v2/ja_speaker_0",
	"Yuki": "v2/ja_speaker_1",
	"Minho": "v2/ko_speaker_0",
	"Jiwoo": "v2/ko_speaker_1",
	"Jan": "v2/pl_speaker_0",
	"Anna": "v2/pl_speaker_1",
	"Miguel": "v2/pt_speaker_0",
	"Ines": "v2/pt_speaker_1",
	"Ivan": "v2/ru_speaker_0",
	"Olga": "v2/ru_speaker_1",
	"Emre": "v2/tr_speaker_0",
	"Ayse": "v2/tr_speaker_1",
	"Li Wei": "v2/zh_speaker_0",
	"Xiao Hong": "v2/zh_speaker_1"
}
