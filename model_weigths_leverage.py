from copy import deepcopy
import json
import matplotlib.pyplot as plt
import numpy as np
import pickle
import re
import sys
import unicodedata

from kneed import KneeLocator
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import KeyedVectors
# import spacy
# from spacy.tokens import Doc

import utils

def strip_accents(text):
    try:
        text = unicode(text, 'utf-8')
    except NameError: # unicode is a default on python 3 
        pass

    text = unicodedata.normalize('NFD', text)\
           .encode('ascii', 'ignore')\
           .decode("utf-8")

    return str(text)

def heavy_entities(epochs=10, dataset="alex_", retrain_loops=1) :
    ###############
    ### Loading and setting resources

    # gensim settings : 10 epochs (default), negative sampling 5, learning rate 0.025 to 0.0001
    embedding_model = Doc2Vec.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_model_epochs{epochs}.d2v')
    embedding_model.dv = KeyedVectors.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_documents_epochs{epochs}.dv')
    embedding_model.wv = KeyedVectors.load(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_words_epochs{epochs}.wv')
    window = embedding_model.window
    null_word = '\0'

    doc_id_to_text = {}
    with open(f'../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_doc_id_to_text_epochs{epochs}.json', encoding='utf-8-sig') as doc_id_to_text_FH : doc_id_to_text = json.load(doc_id_to_text_FH)

    # keep_fusing = True
    # loop_count = 0
    data = []
    with open(f'../embeddings/expe_results/data/{dataset}ready_for_clustering_tweets_spellcheck.json', encoding='utf-8-sig') as data_FH : 
        data = [deepcopy(tweet_dict) for tweet_dict in json.load(data_FH)]

    # while keep_fusing:
    all_bidir_ents, MUL = [], 0
    for loop_count in range(retrain_loops):
        learned_documents = embedding_model.dv
        learned_words = embedding_model.wv
        learned_weights = embedding_model.syn1neg
        
        doc_tag_to_doc_idx = {}
        for doc_tag in doc_id_to_text : doc_tag_to_doc_idx[doc_tag] = learned_documents.key_to_index[doc_tag]
        print(list(doc_id_to_text.items())[0])
                
        doc_activation_range, word_activations_range  = [0, 0], [0, 0]
        doc_activations, word_activations = {}, {}

        for word in learned_words.index_to_key: word_activations[word] = {}
        
        ######
        ### Computing activations

        all_activation_values = []

        print(f"{len(doc_id_to_text)} documents to review")
        for doc_tag in doc_id_to_text :
            doc = doc_id_to_text[doc_tag]
            clean_doc = list(filter(lambda x : x in learned_words.key_to_index, doc))
            padded_doc = [null_word] * window + clean_doc + [null_word] * window

            # doc_activations[doc_tag] = {'text': clean_doc, 'activations': {}}

            for i in range(len(clean_doc)) :
                padded_idx = i + window
                target_word = padded_doc[padded_idx]
                target_idx = learned_words.key_to_index[target_word]
                context_words = padded_doc[i : padded_idx] + padded_doc[padded_idx + 1 : padded_idx + 1 + window ]

                context_vectors = [learned_words[context_word] for context_word in context_words]
                document_vector = learned_documents[doc_tag_to_doc_idx[doc_tag]]

                agreg_vector_tuple = tuple([document_vector] + context_vectors)
                agreg_vector = np.concatenate(agreg_vector_tuple)

                output_layer_param_vector = learned_weights[target_idx]

                context_activations = np.split(agreg_vector * output_layer_param_vector, (window * 2 + 1))

                ###
                ### Doc activation
                doc_activation = context_activations[0].sum() ** 2
                if doc_activation < doc_activation_range[0]: doc_activation_range[0] = doc_activation 
                elif doc_activation > doc_activation_range[1]: doc_activation_range[1] = doc_activation 

                # doc_activations[doc_tag]['activations'][target_word] = doc_activation ** 2

                ###
                ### Word activation
                context_words_activation = [context_activations[i].sum() ** 2 for i in range(1, len(context_activations))]
                for context_word, context_word_activation in zip(context_words, context_words_activation):
                    if context_word == target_word: continue
                    all_activation_values.append(context_word_activation)
                    if context_word_activation < word_activations_range[0]: word_activations_range[0] = context_word_activation 
                    elif context_word_activation > word_activations_range[1]: word_activations_range[1] = context_word_activation 

                    if context_word not in word_activations[target_word] : word_activations[target_word][context_word] = {'activation':0,  'count':0}
                    word_activations[target_word][context_word]['activation'] += context_word_activation 
                    word_activations[target_word][context_word]['count'] += 1

        print(doc_activation_range)
        print(word_activations_range)

        for target_word in word_activations:
            for context_word in word_activations[target_word]:
                count = word_activations[target_word][context_word]['count']
                act = word_activations[target_word][context_word]['activation']

                word_activations[target_word][context_word]['activation'] = act / count

            sorted_values = sorted(word_activations[target_word].items(), key=lambda item: item[1]['activation'], reverse=True)
            word_activations[target_word] = {k: v['activation'] for k, v in sorted_values}#[:20] + sorted_values[-20:]}

        with open(f"outputs/bidir/{dataset}knee_word_activations_it{loop_count}.json", "w", encoding="utf-8-sig") as word_act_FH: word_act_FH.write(utils.dict_to_json(word_activations))

        all_activation_values = sorted(list(set(all_activation_values)))
        activation_deciles = np.quantile(all_activation_values, q = np.arange(0.1, 1, 0.1)).tolist()
        print(f"values count : {len(all_activation_values)}")
        print(f"deciles : {activation_deciles}")

        # if activation_deciles[-1] >= (100 * activation_deciles[0]):
        #     keep_fusing = True
        # loop_count += 1

        # activation_value_idx = [elem[0] for elem in enumerate(all_activation_values)]
        # kneedle = KneeLocator(activation_value_idx, all_activation_values, curve="convex", direction="increasing")

        # plt.plot(all_activation_values)
        # plt.savefig(f"outputs/knee_activation_values_it{loop_count}.png")
        # plt.clf()

        threshold = activation_deciles[-1]
        # threshold = kneedle.knee_y
        print(threshold)

        bidir_entities = {}
        for target_word in word_activations:
            for context_word in word_activations[target_word]:
                # if context_word == target_word: continue
                context_over_target = word_activations[target_word][context_word]

                if context_over_target >= threshold:
                    if target_word in word_activations[context_word] and word_activations[context_word][target_word] >= threshold:
                        # context_over_target and target_over_context > threshold
                        entity_tuple = (target_word, context_word) 

                        entity_key, reversed_key = f"{entity_tuple[0]}_{entity_tuple[1]}", f"{entity_tuple[1]}_{entity_tuple[0]}"
                        if reversed_key in bidir_entities: 
                            activation = (bidir_entities[reversed_key] + word_activations[context_word][target_word]) / 2
                            bidir_entities[reversed_key] = activation
                        else : bidir_entities[entity_key] = context_over_target 

        bidir_entities = {k: v for k, v in sorted(bidir_entities.items(), key=lambda item:item[1])}
        bidir_entities_out = [bidir_entity.split('_') for bidir_entity in bidir_entities]
        all_bidir_ents += bidir_entities_out
            
        print(len(bidir_entities))
        with open(f"outputs/bidir/{dataset}bidir_entities_it{loop_count}.txt", "w", encoding="utf-8-sig") as bidir_FH: 
            bidir_FH.write(f"{len(bidir_entities)} entités dibirectionnelles" + "\n")
            new_line_count = 0
            for bidir_entity in bidir_entities_out:
                bidir_FH.write(str(bidir_entity) + "\t")
                new_line_count += 1
                if new_line_count == 4:
                    bidir_FH.write("\n")
                    new_line_count = 0

        token_count, fused_count = 0, 0
        new_data = []
        for tweet_dict in data:
            # old_texts.append(tweet_dict['text'])
            new_tweet_dict = deepcopy(tweet_dict)
            existing_units = tweet_dict['clean_text']
            token_count += len(existing_units)
            replacing_units = []

            idx = 0
            while idx < len(existing_units):
                ex_unit = existing_units[idx]
                not_fused = True
                for entity_couple in bidir_entities:
                    if ex_unit in entity_couple and (idx+1) < len(existing_units):
                        next_unit = existing_units[idx + 1]

                        if next_unit != ex_unit and next_unit in entity_couple:
                            not_fused = False
                            replacing_units.append(f"{ex_unit} {next_unit}")
                            idx += 1
                            break

                if not_fused:
                    replacing_units.append(ex_unit)
                idx += 1
            
            fused_count += len(replacing_units)
            new_tweet_dict['clean_text'] = replacing_units
            new_data.append(new_tweet_dict)

        print(f"new count : {fused_count} for old count : {token_count}")

        # #########
        # Retrain
        print(f"Retraining #{loop_count}...")

        tagged_documents = [TaggedDocument(tweet_dict['clean_text'], [f"doc{tweet_dict['id']}"]) for tweet_dict in new_data]

        doc_id_to_text = {}
        for elem in tagged_documents: doc_id_to_text[elem.tags[0]] = elem.words

        embedding_model = Doc2Vec(tagged_documents, min_count=3, workers=4, vector_size=500, window=2, dm_concat=1, epochs=epochs)

        data = new_data
        print(f"Retraining #{loop_count}... Done. To disk")

        # #########
        # Retrain completed, saving model
        MUL = 2 ** (loop_count + 1)
        embedding_model.save(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_model_epochs{epochs}_MUL{MUL}.d2v")
        embedding_model.dv.save(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_documents_epochs{epochs}_MUL{MUL}.dv")
        embedding_model.wv.save(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_words_epochs{epochs}_MUL{MUL}.wv")

        with open(f"../embeddings/expe_results/data/{dataset}ready_for_clustering_tweets_spellcheck_epochs{epochs}_MUL{MUL}.json", "w", encoding="utf-8-sig") as resultFh :
            resultFh.write(json.dumps(data))

        with open(f"../embeddings/expe_results/data/{dataset}ready_for_clustering_tweets_spellcheck_epochs{epochs}_MUL{MUL}_readable.txt", "w", encoding="utf-8-sig") as result_FH :
            for tweet_dict in data:
                result_FH.write(json.dumps(tweet_dict))
                result_FH.write(",\n")

        with open(f"../embeddings/expe_results/trained_models/{dataset}window2_spellcheck_concat_doc_id_to_text_epochs{epochs}_MUL{MUL}.json", "w", encoding="utf-8-sig") as doc_id_tag_to_text_FH :
            doc_id_tag_to_text_FH.write(json.dumps(doc_id_to_text))
    
    with open(f"outputs/ablation_study/{dataset}bidir_ents_epochs{epochs}_MUL{MUL}.txt", "w", encoding="utf-8-sig") as bidir_FH:
        bidir_FH.write(str(len(all_bidir_ents)) + "\n")
        idx, end = 0, 4
        while end < len(all_bidir_ents):
            bidir_FH.write(json.dumps(all_bidir_ents[idx:end]) + '\n')
            idx += 4
            end += 4


        ########## spacy NER on tweets score 0.278140062 acc, 0.380822981 rec, 0.321481239 F1
        ########## 1% of what we find in bidir entities shows up in recognized entities
        # nlp = spacy.load('fr_core_news_lg')
        # K = 10
        # cluster_details = {}
        # with open(f"../cluster_desc/data/clusters_raw_recap_spellcheck_window2_timestamp_K{K}.pkl", "rb") as cluster_details_FH:
        #     cluster_details = pickle.load(cluster_details_FH)

        # re_token_match = r"\W??([-#\w]+)\W??"
        # nlp.tokenizer.token_match = re.compile(re_token_match).match
        # # from nltk.corpus import stopwords
        # # stopwords = set(stopwords.words('french'))
        # recognized_ents_old = {}
        # for original_text in [tweet_obj['text'] for tweet_obj in cluster_details['8']['tweets']]:
        #     idx = 0
        #     analyzed_old_text = nlp(original_text)
        #     for ent in analyzed_old_text.ents:
        #     # for ent in analyazed_old_text.ents:
        #         if ' ' in ent.text and 'http' not in ent.text: 
        #             token_number = len(ent.text.split(' '))
        #             if token_number == 2:
        #                 clean_ent_key = ent.text.replace('"','').replace('\n', '')

        #                 if ent.label_ not in recognized_ents_old:
        #                     recognized_ents_old[ent.label_] = {clean_ent_key:1}
        #                 else:
        #                     if clean_ent_key not in recognized_ents_old[ent.label_]:
        #                         recognized_ents_old[ent.label_][clean_ent_key] = 1
        #                     else:
        #                         recognized_ents_old[ent.label_][clean_ent_key] += 1

        # with open("outputs/ner_old.json", "w", encoding="utf-8-sig") as old_FH:
        #     old_FH.write(utils.dict_to_json(recognized_ents_old))

        # found_amount = 0
        # for bidir_entity in bidir_entities_out:
        #     for entity_class in recognized_ents_old:
        #         if entity_class == 'MISC': continue
        #         found = False
        #         for entity_ner in recognized_ents_old[entity_class]:
        #             clean_ner = re.findall(r"\W??([-#\w]+)\W??", strip_accents(entity_ner).lower())
        #             if bidir_entity[0] in clean_ner and bidir_entity[1] in clean_ner:
        #                 found_amount += 1
        #                 found = True
        #                 break
        #         if found: break
        # found_percentage = (found_amount / len(bidir_entities)) * 100
        # print(found_amount)
        # print(found_percentage)
        


if __name__ == '__main__':
    epochs, dataset = sys.argv[1:3]
    heavy_entities(epochs=int(epochs), dataset=f"{dataset}_", retrain_loops=2)