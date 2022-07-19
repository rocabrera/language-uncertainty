


def squad_tokenizer_prompt(tokenizer, sample: dict):
    """
    Importante notar que não podemos retornar tensores se quisermos fazer em batch!
    """
    
    if isinstance(sample["question"], list):
        encoding = tokenizer(
            ['question answering: ' + s for s in sample["question"]],
            sample["context"],
            max_length=396,
            padding="max_length",
            truncation="only_second", # Se nao me engano trunca somente o contexto .... Problematico dependendo de onde a resposta esta
            return_attention_mask=True,
            add_special_tokens=True,
        )
                
    else:
        encoding = tokenizer(
            'question answering: ' + sample["question"],
            sample["context"],
            max_length=396,
            padding="max_length",
            truncation="only_second", # Se nao me engano trunca somente o contexto .... Problematico dependendo de onde a resposta esta
            return_attention_mask=True,
            add_special_tokens=True,
        )
    
    return encoding


def squad_tokenizer_answer(tokenizer, sample: dict):

    if isinstance(sample["answers"], list):
        
        texts = [s["text"][0] for s in sample["answers"]]
        answer_encoding = tokenizer(
            texts,
            max_length=32,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            add_special_tokens=True,
            return_tensors="pt"
        )
                
    else:
        answer_encoding = tokenizer(
        example["answers"]["text"][0],
        max_length=32,
        padding="max_length",
        truncation=True,
        return_attention_mask=True,
        add_special_tokens=True,
        return_tensors="pt"
    )

    labels_ids = answer_encoding["input_ids"]
    # https://huggingface.co/docs/transformers/model_doc/t5 
    labels_ids[labels_ids == tokenizer.pad_token_id] = -100
    encoding = {"labels": labels_ids.tolist()}

    return encoding