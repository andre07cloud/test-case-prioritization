import pandas as pd
import json
import os
import shutil
import sys
from .load_config_file import load_config_file

def manual_analysis(config_file):

    config = load_config_file(config_file)
    if not config:
        print("ERREUR : Impossible de charger le fichier de configuration.")
        return


    difficulty_levels = config.get("difficulty_levels")

    # --- 1. Définition des Fichiers ---

    # Les fichiers CSV contenant les listes d'images diversifiées
    
    csv_files = [
        difficulty_levels[0]['level_1'],
        difficulty_levels[1]['level_2'],
        difficulty_levels[2]['level_3'],
        difficulty_levels[3]['level_4']
    ]

    # Le fichier JSON "maître" contenant tous les chemins et caractéristiques
    master_json_path = config.get("master_json_path")

    # Le dossier de sortie principal pour votre analyse
    main_output_folder = config.get("main_output_folder")
    collected_file_folder = config.get("collected_file_folder") 

    print(f"--- Démarrage de l'organisation des fichiers d'analyse ---")

    # --- 2. Chargement du JSON Maître (une seule fois) ---
    try:
        with open(master_json_path, 'r') as f:
            master_data = json.load(f)
        print(f"Fichier maître '{master_json_path}' chargé.")
    except FileNotFoundError:
        print(f"ERREUR : Le fichier JSON maître '{master_json_path}' est introuvable.")
        sys.exit()
    except json.JSONDecodeError:
        print(f"ERREUR : Le fichier JSON maître '{master_json_path}' est corrompu.")
        sys.exit()

    # --- 3. Boucle sur chaque fichier CSV de niveau de difficulté ---
    for csv_file in csv_files:
        if not os.path.exists(csv_file):
            print(f"\n⚠️ Fichier CSV '{csv_file}' introuvable. Passage au suivant.")
            continue

        # 1. Créer le nom du dossier de sortie (ex: "diversifier_ANN_niveau_1")
        folder_name = os.path.splitext(os.path.basename(csv_file))[0]
        output_dir_images = os.path.join(main_output_folder, folder_name, "images")
        output_dir_labels = os.path.join(main_output_folder, folder_name, "labels")
        output_dir = os.path.join(main_output_folder, folder_name)
        # Créer le dossier
        os.makedirs(output_dir_images, exist_ok=True)
        os.makedirs(output_dir_labels, exist_ok=True)
        os.makedirs(output_dir, exist_ok=True)
        print(f"\n--- Traitement de '{csv_file}' ---")
        print(f"Création du dossier : '{output_dir}/'")

        # 2. Lire le fichier CSV
        try:
            df = pd.read_csv(csv_file)
            if 'image_id' not in df.columns:
                print(f"  ERREUR : Pas de colonne 'image_id' dans '{csv_file}'.")
                continue
        except Exception as e:
            print(f"  ERREUR : Impossible de lire '{csv_file}': {e}")
            continue
            
        # 3. Préparer le nouveau JSON (filtré)
        new_json_data = {}
        files_copied_count = 0

        # 4. Boucle sur chaque image sélectionnée dans le CSV
        for image_id in df['image_id']:
            json_key = str(image_id) # Les clés JSON sont des strings

            # Vérifier si l'ID existe dans le JSON maître
            if json_key not in master_data:
                print(f"  Alerte : image_id '{json_key}' du CSV non trouvé dans le JSON maître.")
                continue
            
            entry = master_data[json_key]
            
            # 5. Copier les fichiers image et label
            img_path = os.path.join(collected_file_folder, entry.get('image'))
            label_path = os.path.join(collected_file_folder, entry.get('label'))

            # Copie de l'image
            if img_path and os.path.exists(img_path):
                try:
                    shutil.copy(img_path, output_dir_images)
                    files_copied_count += 1
                except Exception as e:

                    print(f"  Erreur copie (image) '{img_path}': {e}")
            elif img_path:
                print(f"  Alerte : Fichier image '{img_path}' (ID: {json_key}) introuvable.")
                
            # Copie du label
            if label_path and os.path.exists(label_path):
                try:
                    shutil.copy(label_path, output_dir_labels)
                except Exception as e:
                    print(f"  Erreur copie (label) '{label_path}': {e}")
            elif label_path:
                print(f"  Alerte : Fichier label '{label_path}' (ID: {json_key}) introuvable.")

            # 6. Ajouter les caractéristiques au nouveau JSON
            new_json_data[json_key] = entry

        # 7. Sauvegarder le nouveau fichier JSON filtré
        new_json_path = os.path.join(output_dir, f"{folder_name}_features.json")
        try:
            with open(new_json_path, 'w') as f:
                json.dump(new_json_data, f, indent=4)
            print(f"  -> Nouveau JSON sauvegardé : '{new_json_path}' ({len(new_json_data)} entrées)")
            print(f"  -> {files_copied_count} fichiers images copiés dans le dossier.")
        except Exception as e:
            print(f"  ERREUR : Impossible de sauvegarder le nouveau JSON : {e}")

        print("\n--- ✅ Opération terminée ---")
        print(f"Tous les fichiers sont organisés dans le dossier : '{main_output_folder}'")
  