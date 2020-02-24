import tensorflow as tf
from tensorflow.keras import layers as l
from transformers import *


def convert_labels(train_y_raw, val_y_raw, index=0, num_shards=1):
    labels = list(set(train_y_raw + val_y_raw))
    labels.sort()
    num_labels = len(labels)
    
    total_examples = len(train_y_raw)
    shard_size = total_examples//num_shards
    train_y_raw = train_y_raw[index*shard_size:(index+1)*shard_size]
    
    total_examples = len(val_y_raw)
    shard_size = total_examples//num_shards
    val_y_raw = val_y_raw[index*shard_size:(index+1)*shard_size]
    
    train_y = [labels.index(y) for y in train_y_raw]
    train_y = tf.keras.utils.to_categorical(train_y, num_classes=num_labels)
    val_y = [labels.index(y) for y in val_y_raw]
    val_y = tf.keras.utils.to_categorical(val_y, num_classes=num_labels)
    
    return {"classes": labels,
            "train_y": train_y,
            "val_y": val_y}


def docred_labels():
    return ['applies_to_jurisdiction', 'author', 'award_received', 'basin_country', 'capital', 'capital_of', 'cast_member', 'chairperson', 'characters', 'child', 'composer', 'conflict', 'contains_administrative_territorial_entity', 'continent', 'country', 'country_of_citizenship', 'country_of_origin', 'creator', 'date_of_birth', 'date_of_death', 'developer', 'director', 'dissolved,_abolished_or_demolished', 'educated_at', 'employer', 'end_time', 'ethnic_group', 'father', 'followed_by', 'follows', 'founded_by', 'genre', 'has_part', 'head_of_government', 'head_of_state', 'headquarters_location', 'inception', 'influenced_by', 'instance_of', 'languages_spoken,_written_or_signed', 'league', 'legislative_body', 'located_in_or_next_to_body_of_water', 'located_in_the_administrative_territorial_entity', 'located_on_terrain_feature', 'location', 'location_of_formation', 'lyrics_by', 'manufacturer', 'member_of', 'member_of_political_party', 'member_of_sports_team', 'military_branch', 'mother', 'mouth_of_the_watercourse', 'narrative_location', 'notable_work', 'official_language', 'operator', 'original_language_of_work', 'original_network', 'owned_by', 'parent_organization', 'parent_taxon', 'part_of', 'participant', 'participant_of', 'performer', 'place_of_birth', 'place_of_death', 'platform', 'point_in_time', 'position_held', 'present_in_work', 'producer', 'product_or_material_produced', 'production_company', 'publication_date', 'publisher', 'record_label', 'religion', 'replaced_by', 'replaces', 'residence', 'screenwriter', 'separated_from', 'series', 'sibling', 'sister_city', 'spouse', 'start_time', 'subclass_of', 'subsidiary', 'territory_claimed_by', 'unemployment_rate', 'work_location']


def convert_texts(tokenizer, text_raw, max_seq_len=512, index=0, num_shards=1):
    """
    Takes in a list of sentences, converts into the correct tokenized form.
    """
    text_x, text_x_attn_mask = [], []
    
    total_examples = len(text_raw)
    shard_size = total_examples//num_shards
    text_raw = text_raw[index*shard_size:(index+1)*shard_size]
    
    for x in text_raw:
        tokens = tokenizer.encode(x)
        if len(tokens) > max_seq_len:
            tokens_1 = tokens[:max_seq_len//2]
            tokens_2 = tokens[-max_seq_len//2:]
            tokens = tokens_1 + tokens_2
        text_x.append(tokens)
        attention_mask = [1 for _ in range(len(tokens))]
        text_x_attn_mask.append(attention_mask)
        
    text_x = tf.keras.preprocessing.sequence.pad_sequences(text_x,
                                                           maxlen=max_seq_len,
                                                           dtype="int32",
                                                           padding="post",
                                                           truncating="post")
    
    text_x_attn_mask = tf.keras.preprocessing.sequence.pad_sequences(text_x_attn_mask,
                                                           maxlen=max_seq_len,
                                                           dtype="int32",
                                                           padding="post",
                                                           truncating="post")
    
    return {"texts": text_x,
            "attention_masks": text_x_attn_mask}
    
    


def roberta_model(model_name, config, num_labels, max_seq_len=512, horovod=False):
    # use model = "roberta-large"
    xfmer = TFRobertaModel.from_pretrained(model_name, config=config)

    l_input_sentence = l.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_input_attn_mask = l.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_x = xfmer({"input_ids": l_input_sentence,
                 "attention_mask": l_input_attn_mask})
    l_x, l_pool = l_x[0], l.RepeatVector(1)(l_x[1])
    l_x = l.concatenate([l_x, l_pool], axis=1)
    l_x = l.Conv1D(filters=64, kernel_size=3, activation="relu")(l_x)
    l_x = l.Flatten()(l_x)
    l_x = l.Dropout(rate=0.5)(l_x)
    preds = l.Dense(num_labels, activation="softmax")(l_x)

    model = tf.keras.models.Model(inputs=[l_input_sentence, l_input_attn_mask],
                                  outputs=preds)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
    if horovod:
        import horovod.tensorflow.keras as hvd
        opt = hvd.DistributedOptimizer(opt,
                                       compression=hvd.Compression.fp16,
                                       sparse_as_dense=True)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"],
                  experimental_run_tf_function=not horovod)
    
    return model



def distilbert_model(model_name, config, num_labels, max_seq_len=512, horovod=False):
    # use model = "distilbert-base-cased"
    xfmer = TFDistilBertModel.from_pretrained(model_name, config=config)

    l_input_sentence = l.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_input_attn_mask = l.Input(shape=[max_seq_len,], dtype=tf.int32)
    l_x = xfmer({"input_ids": l_input_sentence,
                 "attention_mask": l_input_attn_mask})[0]
    l_x = l.Conv1D(filters=64, kernel_size=3, activation="relu")(l_x)
    l_x = l.Flatten()(l_x)
    l_x = l.Dropout(rate=0.5)(l_x)
    preds = l.Dense(num_labels, activation="softmax")(l_x)

    model = tf.keras.models.Model(inputs=[l_input_sentence, l_input_attn_mask],
                                  outputs=preds)

    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, epsilon=1e-08)
    if horovod:
        import horovod.tensorflow.keras as hvd
        opt = hvd.DistributedOptimizer(opt,
                                       compression=hvd.Compression.fp16,
                                       sparse_as_dense=True)
    opt = tf.keras.mixed_precision.experimental.LossScaleOptimizer(opt, "dynamic")

    model.compile(loss="categorical_crossentropy",
                  optimizer=opt,
                  metrics=["acc"],
                  experimental_run_tf_function=not horovod)
    
    return model
