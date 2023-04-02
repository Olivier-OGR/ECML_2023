import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import sys

from gensim.models import KeyedVectors

import utils

def to_csv(dataset="alex_", K=10, epochs=10, MUL_suffix=""):
    ablation_shifts, ablation_shifts_from_cluster, average_word_impact, doc_dist_to_centroids = {}, {}, {}, {}
    learned_words = KeyedVectors.load(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_words_epochs{epochs}{MUL_suffix}.wv") # learned words

    with open(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}.json", encoding="utf-8-sig") as ablation_shifts_FH: 
        ablation_shifts = json.load(ablation_shifts_FH)
    
    cluster_keys = []
    for doc_tag in ablation_shifts:
        if doc_tag == "values_domain": continue
        cluster_label = ablation_shifts[doc_tag]['cluster_label'] 
        if cluster_label not in ablation_shifts_from_cluster: ablation_shifts_from_cluster[cluster_label] = {}
        if cluster_label not in cluster_keys: cluster_keys.append(cluster_label)
        if cluster_label not in doc_dist_to_centroids: doc_dist_to_centroids[cluster_label] = {}

        doc_dist_to_centroids[cluster_label][doc_tag] = float(ablation_shifts[doc_tag]['dist_to_centroid']) ** 2

        for word in ablation_shifts[doc_tag]['impacts']:
            if word not in average_word_impact: average_word_impact[word] = {'shift':0, 'count':0}
            average_word_impact[word]['shift'] += float(ablation_shifts[doc_tag]['impacts'][word])
            average_word_impact[word]['count'] += 1

        for word in ablation_shifts[doc_tag]['shift_from_centroid']:
            if word not in ablation_shifts_from_cluster[cluster_label]: ablation_shifts_from_cluster[cluster_label][word] = {'shift' : 0, 'count' : 0}
            ablation_shifts_from_cluster[cluster_label][word]['shift'] += float(ablation_shifts[doc_tag]['shift_from_centroid'][word]) ** 2
            ablation_shifts_from_cluster[cluster_label][word]['count'] += 1

    for word in average_word_impact: average_word_impact[word]['shift'] = average_word_impact[word]['shift'] / average_word_impact[word]['count']

    sorted_average_word_impact = sorted(average_word_impact.items(), key=lambda item: item[1]['shift'])
    average_word_impact_csv = "word;shift\n"
    for average_impact_tuple in sorted_average_word_impact:#[:100] + sorted_average_word_impact[-100:]:
        average_word_impact_csv += f"{average_impact_tuple[0]};{str(average_impact_tuple[1]['shift']).replace('.', ',')}" + "\n"

    with open(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}_average_word_impact.csv", "w", encoding="utf-8-sig") as average_impact_FH: average_impact_FH.write(average_word_impact_csv)

    plt.plot([average_impact_tuple[1]['shift'] for average_impact_tuple in sorted_average_word_impact])
    plt.title(f"Average shifts by ablations")
    plt.savefig(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}_average_word_shift.png")
    plt.clf()

    fig, ax1 = plt.subplots()
    ax1.set_xlabel('vocab idx')
    ax1.set_ylabel('word count', fontdict={'color':'red'})
    ax1.plot([average_impact_tuple[1]['count'] for average_impact_tuple in sorted_average_word_impact], color='red')

    ax2 = ax1.twinx()
    ax2.set_ylabel('word average shift', fontdict={'color':'blue'})
    ax2.plot([average_impact_tuple[1]['shift'] for average_impact_tuple in sorted_average_word_impact], color='blue')

    plt.title(f"Average word shift with counts")
    plt.savefig(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}_average_word_shift_with_count.png")
    plt.clf()



    cluster_word_repartition_csv = "word"
    for cluster_label in cluster_keys : cluster_word_repartition_csv += f";{cluster_label}"
    cluster_word_repartition_csv += "\n"
    for word in learned_words.index_to_key:
        cluster_word_repartition_csv += f"{word}"
        for cluster_label in cluster_keys: 
           if word in ablation_shifts_from_cluster[cluster_label]: cluster_word_repartition_csv += f";{ablation_shifts_from_cluster[cluster_label][word]['count']}"
           else: cluster_word_repartition_csv += ";0"
        cluster_word_repartition_csv += "\n"

    with open(f"outputs/ablation_study/{dataset}cluster_word_repartition_K{K}_epochs{epochs}{MUL_suffix}_average_word_shift.csv", "w", encoding="utf-8-sig") as word_repart_FH: word_repart_FH.write(cluster_word_repartition_csv)



    for cluster_label in ablation_shifts_from_cluster:
        for word in ablation_shifts_from_cluster[cluster_label]: 
                shift = ablation_shifts_from_cluster[cluster_label][word]['shift'] 
                count = ablation_shifts_from_cluster[cluster_label][word]['count'] 
                ablation_shifts_from_cluster[cluster_label][word]['shift'] = shift / count


    ablation_nearing_shift_csv = "cluster_label;word;shift;count\n"
    ablation_receding_shift_csv = "cluster_label;word;shift;count\n"
    ablation_candidates = {} 
    outliers = []
    for cluster_label in ablation_shifts_from_cluster: 
        ablation_candidates[cluster_label] = []
        ablation_shifts_from_cluster[cluster_label] = sorted(ablation_shifts_from_cluster[cluster_label].items(), key=lambda item : item[1]['shift'])

        # high values : removing the word got the document far from its cluster centroid 
        high_shift_words = ablation_shifts_from_cluster[cluster_label][-100:]
        # low values : removing the word did not get the document far fom its cluster centroid
        lowest_shift_words = ablation_shifts_from_cluster[cluster_label][:100]

        for high_shift_tuple in high_shift_words: 
            ablation_nearing_shift_csv += f"{cluster_label};{high_shift_tuple[0]};{str(high_shift_tuple[1]['shift']).replace('.', ',')};{high_shift_tuple[1]['count']}" + "\n"
        for low_shift_tuple in lowest_shift_words: 
            ablation_receding_shift_csv += f"{cluster_label};{low_shift_tuple[0]};{str(low_shift_tuple[1]['shift']).replace('.', ',')};{low_shift_tuple[1]['count']}" + "\n"

        cluster_top_count, cluster_top_word  = 0, ""
        for elem in ablation_shifts_from_cluster[cluster_label]:
            # if elem
            if elem[1]['count'] > cluster_top_count: 
                cluster_top_count = elem[1]['count']
                cluster_top_word = elem[0]

        # plotting for cluster's trends
        plt.plot([nearing_tuple[1]['shift'] for nearing_tuple in ablation_shifts_from_cluster[cluster_label]])
        plt.title(f"Average shifts induced by ablations in Cluster {cluster_label}")
        plt.savefig(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}_ablation_shifts_cluster_{cluster_label}.png")
        plt.clf()

        cluster_shift_deciles = np.quantile([nearing_tuple[1]['shift'] for nearing_tuple in ablation_shifts_from_cluster[cluster_label]], q = np.arange(0.1, 1, 0.1)).tolist()
        # print(f"{cluster_label} {str(cluster_shift_deciles)}")

        for ablation_tuple in ablation_shifts_from_cluster[cluster_label]:
            word = ablation_tuple[0]
            value = ablation_tuple[1]['shift']

            if value >= cluster_shift_deciles[2] and value <= cluster_shift_deciles[-2]: 
                ablation_candidates[cluster_label].append(word)
            else: 
                if word not in outliers: outliers.append(word)

        ###
        fig, ax1 = plt.subplots()
        ax1.set_xlabel('vocab idx')
        ax1.set_ylabel('word cluster count', fontdict={'color':'red'})
        # if nearing_tuple[1]['count'] < cluster_top_count
        ax1.plot([nearing_tuple[1]['count'] for nearing_tuple in ablation_shifts_from_cluster[cluster_label]], color='red')

        ax2 = ax1.twinx()
        ax2.set_ylabel('mean shift from cluster', fontdict={'color':'blue'})
        ax2.plot([nearing_tuple[1]['shift'] for nearing_tuple in ablation_shifts_from_cluster[cluster_label]], color='blue')

        plt.title(f"Average shifts induced by ablations in Cluster {cluster_label} with counts")
        plt.savefig(f"outputs/ablation_study/{dataset}K{K}_epochs{epochs}{MUL_suffix}_ablation_shifts_cluster_{cluster_label}_and_count.png")
        plt.clf()

        ablation_shifts_from_cluster[cluster_label] = {k: v for k, v in ablation_shifts_from_cluster[cluster_label]}

    with open(f"outputs/ablation_study/{dataset}K{K}_ablation_outliers.txt", "w", encoding="utf-8-sig") as outliers_FH: outliers_FH.write(json.dumps(outliers))

    # with open("outputs/words_highest_mean_shift_per_cluster.csv", "w", encoding="utf-8-sig") as nearing_shifts_FH, open("outputs/words_lowest_mean_shift_per_cluster.csv", "w", encoding="utf-8-sig") as receding_shifts_FH: 
    #     nearing_shifts_FH.write(ablation_nearing_shift_csv)
    #     receding_shifts_FH.write(ablation_receding_shift_csv)
    
    doc_tag_to_text = {}
    with open(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_doc_id_to_text_epochs{epochs}{MUL_suffix}.json", encoding="utf-8-sig") as doc_tag_to_text_FH : doc_tag_to_text = json.load(doc_tag_to_text_FH)


    ablation_candidates['cpList'] = []
    ablation_candidates['duplicates'] = []
    for cluster_label in doc_dist_to_centroids:
        doc_dist_to_centroids[cluster_label] = {k: v for k, v in sorted(doc_dist_to_centroids[cluster_label].items(), key=lambda item: item[1])}

        close_words = []
        for closest_doc_tag in list(doc_dist_to_centroids[cluster_label].keys())[:12]: close_words += doc_tag_to_text[closest_doc_tag]
        print(len(close_words))

        for word in ablation_candidates[cluster_label]: 
            if word in close_words: 
                if word not in ablation_candidates['cpList'] : ablation_candidates['cpList'].append(word)
                else: ablation_candidates['duplicates'].append(word)

    ablation_candidates['uniques'] = list(filter(lambda item: item not in ablation_candidates['duplicates'], ablation_candidates['cpList']))
    print(f"{len(ablation_candidates['cpList'])} candidates")
    with open(f"outputs/ablation_study/{dataset}version_ablation_candidates_K{K}_epochs{epochs}{MUL_suffix}.json", "w", encoding="utf-8-sig") as ablat_cands_FH: 
        ablat_cands_FH.write(json.dumps(ablation_candidates))

    
if __name__ == "__main__" : 
    dataset, K, epochs = sys.argv[1:4]
    MUL, MUL_suffix = 0, ""
    if len(sys.argv) > 4:
        MUL = sys.argv[4]
        MUL_suffix = f"_MUL{MUL}"
    to_csv(dataset=f"{dataset}_", K=int(K), epochs=int(epochs), MUL_suffix=MUL_suffix)