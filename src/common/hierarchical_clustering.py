import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import pairwise_distances
from scipy.cluster import hierarchy as shc
import sys
import faiss
import os





def hierarchical_clustering(data, method='ward', metric='euclidean', plot_dendrogram=True):
    df = pd.read_csv(data)
    difficulties = df['difficulty'].values
    # If difficulties is a 1-D vector (one difficulty value per image),
    # reshape to an (n_samples, 1) observation matrix so `linkage` treats
    # it as observations rather than a condensed distance matrix.
    if difficulties.ndim == 1:
        if difficulties.size < 2:
            raise ValueError("Need at least two observations for hierarchical clustering")
        # Check for NaNs
        if np.isnan(difficulties).any():
            raise ValueError("'difficulty' column contains NaN values; please clean the data before clustering")
        observations = difficulties.reshape(-1, 1)
    else:
        observations = difficulties
    #features = df.drop(columns=['difficulty']).values

    plt.figure(figsize=(15, 8))
    plt.title("Dendrograms Hierarchical Clustering")
    plt.xlabel("size of clusters(number of images)")
    plt.ylabel("Distance(Variance of Wards' method)")
    # Compute the linkage matrix from the observation matrix
    Z = shc.linkage(observations, method=method, metric=metric)

    if plot_dendrogram:
        dend = shc.dendrogram(
            Z, 
            truncate_mode='lastp',
            p=30,
            leaf_rotation=90.,
            leaf_font_size=12.,
            show_contracted=True,
            show_leaf_counts=True
        )
    example_cut_distance = 10 
    plt.axhline(y=example_cut_distance, color='r', linestyle='--', label=f'Ligne de coupe (exemple à y={example_cut_distance})')
    plt.legend()
    plt.savefig("dendrogramme_difficulte.png")
    plt.show()

    
    print("\n✅ Dendrogramme sauvegardé sous 'dendrogramme_difficulte.png'")    

    K_choisi = 4

    # 'fcluster' prend la matrice Z et le nombre de clusters K (critère 'maxclust')
    clusters = shc.fcluster(Z, K_choisi, criterion='maxclust')

    # 6. Ajout au DataFrame
    df['Difficulty_level_HC'] = clusters

    # Analyser la répartition
    print(f"\n--- Analyse pour K = {K_choisi} clusters ---")
    # cluster_analysis = df.groupby('Niveau_Difficulte_HC')['difficulty'].agg(
    #     Nombre_Images='count',
    #     Diff_Min='min',
    #     Diff_Moyenne='mean',
    #     Diff_Max='max'
    # ).sort_values(by='Diff_Moyenne') # Trier par difficulté moyenne
    # Grouper par N° de cluster, calculer la moyenne de 'difficulty', et trier
    cluster_analysis = df.groupby('Difficulty_level_HC')['difficulty'].agg(
        Mean_difficulty='mean'
    ).sort_values(by='Mean_difficulty') # Tri par ordre croissant de difficulté

    print("\nClassement des clusters (du + facile au + difficile) :")
    print(cluster_analysis)
    labels_difficulty = ['easy', 'normal', 'difficult', 'very difficult']

    mapping_dict = dict(zip(cluster_analysis.index, labels_difficulty))
    print(f"\nMapping qui sera appliqué : {mapping_dict}")

    df['Difficulty_label_level'] = df['Difficulty_level_HC'].map(mapping_dict)

    # Sauvegarder le résultat final
    output_csv = "features_avec_clusters_hc.csv"
    print(df.head())
    df.to_csv(output_csv, index=False)
    print(f"\n✅ Données avec clusters sauvegardées sous '{output_csv}'")

    return Z


def diversity_clustering(data):

    df = pd.read_csv(data)
    cluster_col = 'Difficulty_level_HC' # La colonne de clustering à utiliser
    output_folder = "diverse_files_ann"
    global_output_file = "features_diversifies_ANN_GLOBAL.csv"

    # Créez le dossier de sortie s'il n'existe pas
    os.makedirs(output_folder, exist_ok=True)

    # THRESHOLD : C'est le paramètre le plus important à ajuster.
    # Puisque nous utilisons StandardScaler, les distances seront plus grandes.
    # Un seuil plus élevé = moins d'images (plus diversifiées)
    # Un seuil plus bas = plus d'images (moins diversifiées)
    DISTANCE_THRESHOLD = 2.5 

    # k (voisins à rechercher) : Doit être un nombre raisonnable.
    K_NEIGHBORS = 50 

    print(f"--- Starting diversification ANN (Faiss) ---")
    print(f"Distance Threshold L2 (euclidienne) : {DISTANCE_THRESHOLD}")
    print(f"Output Folder : {output_folder}/\n")

    # --- 2. Chargement des données ---
    try:
        df = pd.read_csv(data)
    except FileNotFoundError:
        print(f"Erreur : Fichier '{data}' non trouvé.")
        sys.exit()
    except Exception as e:
        print(f"Erreur lors de la lecture du CSV : {e}")
        sys.exit()

    # --- 3. Identification des colonnes de features ---
    # Utilise uniquement les positions et orientations
    feature_cols = [col for col in df.columns if (col.startswith('pos_') or col.startswith('rot_') or col.startswith('lighting'))]

    if not feature_cols:
        print("Erreur : Aucune colonne 'pos_' or 'rot_' or 'lighting' trouvée.")
        sys.exit()

    print(f"Using {len(feature_cols)} feature columns (ex: {feature_cols[0]}, {feature_cols[1]}...).")

    # --- 4. Traitement par Cluster ---
    scaler = StandardScaler()
    all_selected_dfs = [] # Pour le fichier global

    unique_clusters = sorted(df[cluster_col].unique())

    for cluster_label in unique_clusters:
        
        print(f"\n--- Difficulty cluster processing : {cluster_label} ---")
        
        # 1. Créer le sous-dataframe
        df_cluster = df[df[cluster_col] == cluster_label].copy()
        
        if df_cluster.empty:
            print("  Aucune image dans ce cluster. Passage au suivant.")
            continue

        print(f"  Cluster length : {len(df_cluster)} images.")

        # Extraire les features et les convertir en float32 pour Faiss
        features = df_cluster[feature_cols].values.astype('float32')
        
        # Normalisation (Standardisation)
        features_scaled = scaler.fit_transform(features)
        
        # --- 5. Indexation Faiss ---
        d = features_scaled.shape[1]      # Dimension des vecteurs
        index = faiss.IndexFlatL2(d)   # Utilisation de la distance L2 (Euclidienne)
        index.add(features_scaled)     # Ajout des features à l'index
        
        # --- 6. Boucle de diversification (votre logique) ---
        N = features_scaled.shape[0]
        selected_indices_in_subset = [] # Index LOCAUX (de 0 à N-1)
        used = np.zeros(N, dtype=bool)
        
        # Assurer que k n'est pas plus grand que le nombre d'items
        k_search = min(N, K_NEIGHBORS) 
        
        for i in range(N):
            if not used[i]:
                # C'est un nouvel échantillon diversifié
                selected_indices_in_subset.append(i)
                
                # Trouver ses k voisins les plus proches
                D, I = index.search(features_scaled[i:i+1], k=k_search)
                
                # Identifier les voisins qui sont *dans* le seuil de distance
                neighbors_within_threshold = I[0][D[0] < DISTANCE_THRESHOLD]
                
                # Marquer tous ces voisins (y compris lui-même, i) comme "utilisés"
                used[neighbors_within_threshold] = True
                
        # --- 7. Sauvegarde des résultats pour ce cluster ---
        
        # Convertir les index locaux (ex: 0, 5, 12...) en index globaux du df (ex: 3, 22, 57...)
        original_indices = df_cluster.iloc[selected_indices_in_subset].index
        
        # Sélectionner les lignes complètes du DataFrame
        df_diverse = df.loc[original_indices].copy()
        all_selected_dfs.append(df_diverse)

        print(f"  Selecting {len(df_diverse)} diverse images (out of {N}).")

        # Sauvegarder le fichier individuel
        output_filename = os.path.join(output_folder, f"diversify_ANN_level_{cluster_label}.csv")
        df_diverse.to_csv(output_filename, index=False)
        print(f"  -> File saved : {output_filename}")

    # --- 8. Sauvegarde du fichier global ---
    if all_selected_dfs:
        # Définir l'ordre personnalisé des niveaux de difficulté
        difficulty_order = {'easy': 0, 'normal': 1, 'difficult': 2, 'very difficult': 3}
        
        # Concaténer tous les DataFrames
        df_global = pd.concat(all_selected_dfs)
        
        # Créer une colonne temporaire pour le tri
        df_global['difficulty_order'] = df_global['Difficulty_label_level'].map(difficulty_order)
        
        # Trier selon l'ordre personnalisé et l'image_id
        df_global = df_global.sort_values(by=['difficulty_order', 'image_id'], ascending=[False, True])
        
        # Supprimer la colonne temporaire de tri
        df_global = df_global.drop('difficulty_order', axis=1)
        
        # Sauvegarder le fichier trié
        df_global.to_csv(global_output_file, index=False)
        print(f"\n✅ GLOBAL file saved ({len(df_global)} images) : '{global_output_file}'")
    else:
        print("\nNo data was processed.")