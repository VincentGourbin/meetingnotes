import os
import tempfile
import torch
import torchaudio
from pydub import AudioSegment
from moviepy import VideoFileClip
import gradio as gr
from pyannote.audio import Pipeline
import mlx_whisper
from mlx_lm import load, generate

class WavConverter:
    def convert_to_wav(self, input_file: str) -> str:
        extension = os.path.splitext(input_file)[1].lower()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_wav:
            temp_wav_path = tmp_wav.name

        if extension == ".mp3" or extension == ".wav":
            audio = AudioSegment.from_file(input_file)
        else:
            # Traitement vidéo
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_audio_extracted:
                tmp_audio_path = tmp_audio_extracted.name
            clip = VideoFileClip(input_file)
            clip.audio.write_audiofile(tmp_audio_path, codec='pcm_s16le', fps=48000)
            audio = AudioSegment.from_file(tmp_audio_path)
            os.remove(tmp_audio_path)

        audio = audio.set_channels(1).set_frame_rate(16000)
        audio.export(temp_wav_path, format="wav")
        return temp_wav_path

class Normalizer:
    def normalize(self, wav_path: str, target_dBFS: float = -20.0) -> str:
        audio = AudioSegment.from_file(wav_path, format="wav")
        change_in_dBFS = target_dBFS - audio.dBFS
        normalized_audio = audio.apply_gain(change_in_dBFS)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_normalized:
            normalized_path = tmp_normalized.name
        normalized_audio.export(normalized_path, format="wav")
        return normalized_path

class Diarizer:
    def __init__(self, hf_token: str):
        self.pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hf_token
        )
        device = torch.device("mps") if torch.backends.mps.is_built() else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pipeline.to(device)

    def diarize(self, wav_path: str, num_speakers: int = None):
        waveform, sample_rate = torchaudio.load(wav_path)
        if num_speakers is not None:
            diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate}, num_speakers=num_speakers)
        else:
            diarization = self.pipeline({"waveform": waveform, "sample_rate": sample_rate})
        return diarization

def extract_speaker_snippet(segments, wav_path, speaker):
    speaker_segments = [seg for seg in segments if seg["speaker"] == speaker]
    if not speaker_segments:
        return None

    audio = AudioSegment.from_file(wav_path, format="wav")
    five_seconds = 5000  # ms
    chosen_seg = None
    max_length = 0
    for seg in speaker_segments:
        length = (seg["end"] - seg["start"]) * 1000
        if length >= five_seconds:
            chosen_seg = seg
            break
        else:
            if length > max_length:
                max_length = length
                chosen_seg = seg

    start_ms = chosen_seg["start"] * 1000
    end_ms = chosen_seg["end"] * 1000
    excerpt_length = end_ms - start_ms
    if excerpt_length > five_seconds:
        end_ms = start_ms + five_seconds

    excerpt = audio[start_ms:end_ms]

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_excerpt:
        excerpt_path = tmp_excerpt.name
    excerpt.export(excerpt_path, format="wav")
    return excerpt_path

def process_file(file, hf_token, num_speakers=None, start_trim=0, end_trim=0):
    if file is None:
        return [], None, None, None

    converter = WavConverter()
    normalizer = Normalizer()
    diarizer = Diarizer(hf_token=hf_token)

    wav_path = converter.convert_to_wav(file.name)
    wav_path = normalizer.normalize(wav_path)

    # Trim audio
    if start_trim > 0 or end_trim > 0:
        audio = AudioSegment.from_file(wav_path, format="wav")
        duration_ms = len(audio)
        start_ms = start_trim * 1000
        end_ms = duration_ms - (end_trim * 1000)
        if start_ms < end_ms:
            trimmed_audio = audio[start_ms:end_ms]
        else:
            trimmed_audio = audio

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_trimmed:
            trimmed_path = tmp_trimmed.name
        trimmed_audio.export(trimmed_path, format="wav")
        wav_path = trimmed_path

    diarization = diarizer.diarize(wav_path, num_speakers=num_speakers)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    unique_speakers = sorted(set(s["speaker"] for s in segments))

    mapping_rows = []
    excerpt_dict = {}
    for spk in unique_speakers:
        excerpt_path = extract_speaker_snippet(segments, wav_path, spk)
        mapping_rows.append([spk, ""])  
        excerpt_dict[spk] = excerpt_path

    return mapping_rows, segments, wav_path, excerpt_dict

def on_process(file, hf_token, num_speakers, start_trim, end_trim):
    mapping_rows, segments, wav_path, excerpt_dict = process_file(file, hf_token, num_speakers, start_trim, end_trim)
    return mapping_rows, segments, wav_path, excerpt_dict

def on_select_table(evt: gr.SelectData, mapping_table, excerpt_dict):
    row_index, col_index = evt.index
    if hasattr(mapping_table, 'iloc'):
        row = mapping_table.iloc[row_index].tolist()
    else:
        row = mapping_table[row_index]
    old_speaker = row[0]
    excerpt_path = excerpt_dict.get(old_speaker, None)
    return excerpt_path

def apply_renaming(mapping_table, segments):
    #print("Mapping Table:")
    #print(mapping_table)

    # 1. S'assurer qu'on a bien une liste de listes
    #    et si c'est un DataFrame, on le convertit en liste de listes
    if hasattr(mapping_table, 'iloc'):  # <- On teste si c'est un DataFrame
        # Convertir en liste de listes
        # ex : [[old_speaker_1, new_speaker_1], [old_speaker_2, new_speaker_2], ...]
        mapping_table = mapping_table.values.tolist()
    
    # 2. Créer un dictionnaire old_speaker -> new_speaker
    mapping_dict = {
        row[0].strip(): row[1].strip() 
        for row in mapping_table 
        if row[1].strip()
    }
    #print("Mapping Dictionary:")
    #print(mapping_dict)
    
    # 3. Appliquer le renommage aux segments
    for seg in segments:
        if seg["speaker"] in mapping_dict and mapping_dict[seg["speaker"]]:
            seg["speaker"] = mapping_dict[seg["speaker"]]
    
    #print("Updated Segments:")
    #print(segments)
    return segments

def on_transcribe(mapping_table, segments, wav_path, whisper_model, language):
    # Renommer les speakers avant transcription 
    segments = apply_renaming(mapping_table, segments)

    if not wav_path:
        return "", segments

    lang_code = "fr" if language == "french" else "en"

    result = mlx_whisper.transcribe(
        wav_path,
        path_or_hf_repo=whisper_model, 
        language=lang_code
    )
    full_transcription = result.get("text", "").strip()
    return full_transcription, segments

def on_summarize(mapping_table, full_transcription, segments, summary_model, language):
    # Ré-appliquer le renommage si jamais le user a modifié les noms après la transcription
    segments = apply_renaming(mapping_table, segments)

    # Construire la chaîne de diarisation avec les speakers renommés
    diarisation_lines = []
    for seg in segments:
        spk = seg['speaker']
        start = seg['start']
        end = seg['end']
        diarisation_lines.append(f"{spk}: {start:.2f} - {end:.2f}")
    diarisation_str = "\n".join(diarisation_lines)

    prompt = f"""Here is the transcription of a meeting:
{full_transcription}

Here are the timestamps of each participant’s interventions:
{diarisation_str}

I would like you to write a structured meeting report with the following elements:
	1.	Summary in 5 lines: A concise summary of the meeting, highlighting its main objective and the results achieved.
	2.	Key points discussed: A list of the topics or themes discussed, presented as bullet points, briefly summarizing each point.
	3.	Actions to be taken: A list of actions decided during the meeting. For each action, please include:
• A description of the action to be performed.
• The person or group responsible for this action (if mentioned).

Whenever possible, try to associate the speaker’s name with each action or key point.

Expected final structure:

Summary
[5-line summary]

Key points discussed
• [Point 1]
• [Point 2]
• [Point 3]

Actions to be taken
	1.	[Action 1]: [Description of the action] – Responsible: [Name or group].
	2.	[Action 2]: [Description of the action] – Responsible: [Name or group].
	3.	[Action 3]: [Description of the action] – Responsible: [Name or group].

Please avoid adding any information that does not appear in the transcription.
Your reply should in {language}
"""

    # Charger dynamiquement le modèle choisi
    model, tokenizer = load(summary_model)

    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=4000)
    return response

def main():
    hf_token = os.environ.get("HUGGINGFACE_TOKEN")
    if hf_token is None:
        raise ValueError("La variable d'environnement HUGGINGFACE_TOKEN n'est pas définie.")

    with gr.Blocks() as demo:
        gr.Markdown("# Diarisation, Transcription Complète et Compte Rendu")

        with gr.Row():
            file_input = gr.File(label="Fichier audio/vidéo")
            num_speakers_input = gr.Number(label="Forcer le nombre de locuteurs (optionnel)", value=None)

        # Ajout des entrées pour le trimming
        with gr.Row():
            start_trim_input = gr.Number(label="Enlever X secondes au début", value=0)
            end_trim_input = gr.Number(label="Enlever Y secondes à la fin", value=0)

        # Liste déroulante pour la langue
        language_input = gr.Dropdown(choices=["french", "english"], value="french", label="Langue du meeting")
        
        btn_process = gr.Button("Lancer la diarisation")
        gr.Markdown("### Renommer les Speakers")
        gr.Markdown("Modifiez la colonne `new_speaker` pour changer le nom du speaker. Sélectionnez une ligne pour écouter l'extrait audio associé.")

        with gr.Row():
            old_speakers = gr.Dataframe(
                headers=["old_speaker", "new_speaker"], 
                datatype=["str", "str"], 
                label="Mapping Speaker",
                interactive=True
            )
            audio_excerpt = gr.Audio(label="Extrait du speaker sélectionné", type="filepath")

        state_segments = gr.State()
        state_wav_path = gr.State()
        state_excerpt_dict = gr.State()

        whisper_model_input = gr.Textbox(label="Modèle Whisper (HF Hub ou chemin local)", value="mlx-community/whisper-large-v3-turbo")
        btn_transcribe = gr.Button("Transcrire")

        # Affichage de la transcription complète dans un Textbox
        full_transcription_output = gr.Textbox(label="Transcription complète")

        # Liste déroulante pour le modèle de compte-rendu
        summary_model_input = gr.Dropdown(
            label="Modèle pour le compte rendu",
            choices=["mlx-community/AIDC-AI_Marco-o1_MLX-8bit", 
                     "mlx-community/Llama-3.3-70B-Instruct-6bit", 
                     "mlx-community/phi-4-bf16"],
            value="mlx-community/Llama-3.3-70B-Instruct-6bit"
        )

        btn_summarize = gr.Button("Générer le compte rendu")
        summary_output = gr.Textbox(label="Compte Rendu")

        btn_process.click(
            on_process, 
            inputs=[file_input, gr.State(value=hf_token), num_speakers_input, start_trim_input, end_trim_input], 
            outputs=[old_speakers, state_segments, state_wav_path, state_excerpt_dict]
        )

        old_speakers.select(on_select_table, inputs=[old_speakers, state_excerpt_dict], outputs=audio_excerpt)

        # on_transcribe retourne full_transcription et segments mis à jour
        btn_transcribe.click(
            on_transcribe,
            inputs=[old_speakers, state_segments, state_wav_path, whisper_model_input, language_input],
            outputs=[full_transcription_output, state_segments]
        )

        btn_summarize.click(
            on_summarize,
            inputs=[old_speakers, full_transcription_output, state_segments, summary_model_input, language_input],
            outputs=[summary_output]
        )

    demo.launch()

if __name__ == "__main__":
    main()