# logic.py
import os
import tempfile
import gradio as gr

from pydub import AudioSegment
import mlx_whisper
from mlx_lm import load, generate

from wav_converter import WavConverter
from normalizer import Normalizer
from diarizer import Diarizer, extract_speaker_snippet

def process_file(file, hf_token, num_speakers=None, start_trim=0, end_trim=0):
    if file is None:
        return [], None, None, None

    converter = WavConverter()
    normalizer = Normalizer()
    diarizer = Diarizer(hf_token=hf_token)

    # Convertir en WAV puis normaliser
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

    # Diarisation
    diarization = diarizer.diarize(wav_path, num_speakers=num_speakers)
    segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })

    # Préparation du tableau (pour renommer si besoin)
    unique_speakers = sorted(set(s["speaker"] for s in segments))
    mapping_rows = []
    excerpt_dict = {}
    for spk in unique_speakers:
        excerpt_path = extract_speaker_snippet(segments, wav_path, spk)
        mapping_rows.append([spk, ""])  
        excerpt_dict[spk] = excerpt_path

    return mapping_rows, segments, wav_path, excerpt_dict


def on_process(file, hf_token, num_speakers, start_trim, end_trim):
    """
    Fonction appelée par le bouton Gradio pour lancer la diarisation.
    """
    mapping_rows, segments, wav_path, excerpt_dict = process_file(
        file, hf_token, num_speakers, start_trim, end_trim
    )
    return mapping_rows, segments, wav_path, excerpt_dict


def on_select_table(evt: gr.SelectData, mapping_table, excerpt_dict):
    """
    Appelé lorsqu'une ligne est sélectionnée dans le tableau des anciens locuteurs.
    """    
    row_index = evt.index
    row_index, col_index = evt.index
    if hasattr(mapping_table, 'iloc'):
        row = mapping_table.iloc[row_index].tolist()
    else:
        row = mapping_table[row_index]
    old_speaker = row[0]
    excerpt_path = excerpt_dict.get(old_speaker, None)
    return excerpt_path


def apply_renaming(mapping_table, segments):
    """
    Met à jour les noms de locuteurs (old_speaker -> new_speaker)
    """
    # Si c'est un DataFrame, on le convertit en liste de listes
    if hasattr(mapping_table, 'iloc'):
        mapping_table = mapping_table.values.tolist()

    # Création d’un dictionnaire pour le renommage
    mapping_dict = {
        row[0].strip(): row[1].strip() 
        for row in mapping_table 
        if row[1].strip()
    }

    # Application du renommage aux segments
    for seg in segments:
        if seg["speaker"] in mapping_dict and mapping_dict[seg["speaker"]]:
            seg["speaker"] = mapping_dict[seg["speaker"]]

    return segments


def on_transcribe(mapping_table, segments, wav_path, whisper_model, language):
    """
    Renomme les speakers, puis transcrit l'audio avec mlx_whisper.
    Retourne la transcription complète et la liste de segments mise à jour.
    """
    # Appliquer le renommage avant de lancer la transcription
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
    """
    Renomme une dernière fois les speakers si nécessaire, puis génère un compte rendu
    structuré grâce à un modèle MLM (mlx_lm).
    """
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

    # Charger le modèle de résumé 
    model, tokenizer = load(summary_model)

    # Cas où le tokenizer propose un template de style "chat"
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template is not None:
        messages = [{"role": "user", "content": prompt}]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    response = generate(model, tokenizer, prompt=prompt, verbose=False, max_tokens=4000)
    return response