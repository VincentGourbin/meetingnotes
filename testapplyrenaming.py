def apply_renaming(mapping_table, segments):
    print("Mapping Table:")
    print(mapping_table)

    # Créer un dictionnaire de mappage avec les noms complets des speakers
    mapping_dict = {row[0].strip(): row[1].strip() for row in mapping_table if row[1].strip()}
    print("Mapping Dictionary:")
    print(mapping_dict)
    
    # Appliquer le renommage aux segments
    for seg in segments:
        if seg["speaker"] in mapping_dict and mapping_dict[seg["speaker"]]:
            seg["speaker"] = mapping_dict[seg["speaker"]]
    
    print("Updated Segments:")
    print(segments)
    return segments

# Exemple d'utilisation
mapping_table = [
    ["SPEAKER_00", "quentin"],
    ["SPEAKER_01", ""],  # Cas où le nouveau nom est vide
    ["SPEAKER_02", "Vincent"]
]

segments = [
    {'start': 4.148468750000001, 'end': 4.199093749999999, 'speaker': 'SPEAKER_00'},
    {'start': 4.199093749999999, 'end': 7.03409375, 'speaker': 'SPEAKER_02'},
    {'start': 7.91159375, 'end': 10.645343750000002, 'speaker': 'SPEAKER_02'},
    {'start': 11.47221875, 'end': 14.10471875, 'speaker': 'SPEAKER_02'},
    {'start': 15.285968750000002, 'end': 17.024093750000002, 'speaker': 'SPEAKER_01'},
    {'start': 17.39534375, 'end': 18.0, 'speaker': 'SPEAKER_00'}
]

apply_renaming(mapping_table, segments)