import json
import numpy as np
import pickle
import sys

from gensim.models.doc2vec import Doc2Vec
from gensim.models import KeyedVectors

import utils

def euclidean_dist(narray1, narray2=None):
    if narray2 is None : return np.sqrt(np.sum(np.square(narray1)))
    else : return np.sqrt(np.sum(np.square(narray1 - narray2)))

def ablation_comparison(dataset="alex_", K=10, epochs=10, MUL_suffix=""):
    # gensim settings : 10 epochs, negative sampling 5, learning rate 0.025 to 0.0001
    model = Doc2Vec.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_model_epochs{epochs}{MUL_suffix}.d2v')
    model.dv = KeyedVectors.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_documents_epochs{epochs}{MUL_suffix}.dv') # learned documents
    model.wv = KeyedVectors.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_words_epochs{epochs}{MUL_suffix}.wv') # learned words
    null_word = '\0'
    model.sample = False

    doc_tag_to_text = {}
    with open(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_doc_id_to_text_epochs{epochs}{MUL_suffix}.json', encoding='utf-8-sig') as doc_tag_to_text_FH : doc_tag_to_text = json.load(doc_tag_to_text_FH)

    cluster_to_doc_tag = {}
    with open(f'../cluster_desc/data/{dataset}doc_tag_per_cluster_K{K}_epochs{epochs}{MUL_suffix}.json', encoding='utf-8-sig') as cluster_to_tag_FH : cluster_to_doc_tag = json.load(cluster_to_tag_FH)

    doc_tag_to_cluster = {}
    for cluster_label in cluster_to_doc_tag:
        for doc_tag in cluster_to_doc_tag[cluster_label] : doc_tag_to_cluster[doc_tag] = cluster_label

    doc_to_word_impact = {}
    all_updates = []
    all_normalised_updates = []
    overall_lowest_update, overall_highest_update = (float("inf"), 0.0)

    cluster_details = {}
    with open(f"../cluster_desc/data/{dataset}clusters_raw_recap_spellcheck_window2_timestamp_K{K}_epochs{epochs}{MUL_suffix}.pkl", "rb") as cluster_details_FH:
        cluster_details = pickle.load(cluster_details_FH)

    print("Computing distances with ablation")
    for doc_tag in doc_tag_to_cluster:
        doc_words = doc_tag_to_text[doc_tag]
        clean_doc = list(filter(lambda x : x in model.wv.key_to_index, doc_words))
        if not len(clean_doc): continue

        doc_tag_to_text[doc_tag] = clean_doc
        doc_vector = model.dv[doc_tag]
        doc_to_word_impact[doc_tag] = {'doc_words' : clean_doc, 'impacts' : {}, 'shift_from_centroid' : {}}

        cluster_label = doc_tag_to_cluster[doc_tag]
        doc_to_word_impact[doc_tag]['cluster_label'] = cluster_label
        cluster_centroid = cluster_details[cluster_label]['centroid']
        doc_dist_to_centroid = euclidean_dist(cluster_centroid, doc_vector)
        doc_to_word_impact[doc_tag]['dist_to_centroid'] = doc_dist_to_centroid

        for doc_word, word_idx in zip(clean_doc, range(len(clean_doc))):
            # if doc_word not in impactful_words : impactful_words[doc_word] = {'is_most_impactful_count' : 0, 'count' : 0}

            ablated_doc = clean_doc[0:word_idx] + [null_word] + clean_doc[word_idx+1:]
            ablated_doc_vector = model.infer_vector(ablated_doc)

            distance_to_init_doc = euclidean_dist(doc_vector, ablated_doc_vector)

            if distance_to_init_doc < overall_lowest_update : overall_lowest_update = distance_to_init_doc
            if distance_to_init_doc > overall_highest_update : overall_highest_update = distance_to_init_doc

            doc_to_word_impact[doc_tag]['impacts'][doc_word] = distance_to_init_doc ** 2

            ablated_doc_dist_to_centroid = euclidean_dist(cluster_centroid, ablated_doc_vector)
            doc_to_word_impact[doc_tag]['shift_from_centroid'][doc_word] = ablated_doc_dist_to_centroid - doc_dist_to_centroid

    print("Sorting results")
    for doc_tag in doc_to_word_impact:
        doc_to_word_impact[doc_tag]['impacts'] = {k : v for k, v in sorted(list(doc_to_word_impact[doc_tag]['impacts'].items()), key=lambda item: item[1], reverse=True)}
        biggest_update = list(doc_to_word_impact[doc_tag]['impacts'].items())[0][1]
        doc_to_word_impact[doc_tag]['context_normalised_impacts'] = {k : v/biggest_update for k, v in sorted(doc_to_word_impact[doc_tag]['impacts'].items(), key=lambda item: item[1], reverse=True)}
        all_updates += list(doc_to_word_impact[doc_tag]['impacts'].values())
        all_normalised_updates += list(doc_to_word_impact[doc_tag]['context_normalised_impacts'].values())

    global_quartiles = np.quantile(all_updates, q = np.arange(0.25, 1, 0.25)).tolist()
    global_normalised_quartiles = np.quantile(all_normalised_updates, q = np.arange(0.25, 1, 0.25)).tolist()
    print((global_quartiles, global_normalised_quartiles))

    update_ordered_words = {'0' : [], '1' : [], '2' : [], '3' : []}
    normalised_update_ordered_words = {'0' : [], '1' : [], '2' : [], '3' : []}
    for doc_tag in doc_to_word_impact: 
        cluster_label = doc_to_word_impact[doc_tag]['cluster_label']

        for doc_word in doc_to_word_impact[doc_tag]['impacts'] :
            update_value = doc_to_word_impact[doc_tag]['impacts'][doc_word]
            if update_value >= global_quartiles[2] : update_ordered_words['3'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            elif update_value >= global_quartiles[1] : update_ordered_words['2'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            elif update_value >= global_quartiles[0] : update_ordered_words['1'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            else : update_ordered_words['0'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})

            update_value = doc_to_word_impact[doc_tag]['context_normalised_impacts'][doc_word]
            if update_value >= global_normalised_quartiles[2] : normalised_update_ordered_words['3'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            elif update_value >= global_normalised_quartiles[1] : normalised_update_ordered_words['2'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            elif update_value >= global_normalised_quartiles[0] : normalised_update_ordered_words['1'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            else : normalised_update_ordered_words['0'].append({'cluster' : cluster_label, 'word' : doc_word, 'shift': update_value})
            
    for quartile_key in update_ordered_words :
        unsorted_quartile = update_ordered_words[quartile_key]
        unsorted_normalised_quartile = normalised_update_ordered_words[quartile_key]

        update_ordered_words[quartile_key] = sorted(unsorted_quartile, key=lambda item: item['shift'])
        normalised_update_ordered_words[quartile_key] = sorted(unsorted_normalised_quartile, key=lambda item: item['shift'])

    doc_to_word_impact["values_domain"] = (overall_lowest_update, overall_highest_update)

    with open(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}.json", "w", encoding="utf-8-sig") as ablation_study_raw_FH, open(f"outputs/ablation_study/{dataset}quartiles_K{K}_epochs{epochs}{MUL_suffix}.json", "w", encoding="utf-8-sig") as quartiles_FH, open(f"outputs/ablation_study/{dataset}normalised_quartiles_K{K}_epochs{epochs}{MUL_suffix}.json", "w", encoding="utf-8-sig") as normalised_quartiles_FH: 
        ablation_study_raw_FH.write(utils.dict_to_json(doc_to_word_impact))
        quartiles_FH.write(utils.dict_to_json(update_ordered_words))
        normalised_quartiles_FH.write(utils.dict_to_json(normalised_update_ordered_words))

if __name__ == "__main__" :
    dataset, K, epochs = sys.argv[1:4]
    MUL, MUL_suffix = 0, ""
    if len(sys.argv) > 4:
        MUL = sys.argv[4]
        MUL_suffix = f"_MUL{MUL}"
    ablation_comparison(dataset=f"{dataset}_", K=int(K), epochs=int(epochs), MUL_suffix=MUL_suffix)