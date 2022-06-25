import textwrap
from termcolor import colored
from datasets import Dataset


def sample_example_dataset(data: Dataset, idx: int):
    
    """
    Mostra:
    - Pergunta em BRANCO
    - Contexto em VERDE
    - Respota dentro do contexto em AZUL
    - True Label em VERMELHO
    """
    
    elem = data[idx]
    question, context, answers = elem["question"], elem["context"], elem["answers"]
    answer_text, answer_start = answers["text"][0], answers["answer_start"][0]
    prompt = colored(question, "white") + \
             colored(context[:answer_start], "green") + \
             colored(context[answer_start:answer_start+len(answer_text)], "blue") + \
             colored(context[answer_start+len(answer_text):], "green")
    
    for wrap in textwrap.wrap(prompt, width = 120):
        print(wrap)
        
    print(colored("TRUE LABEL: " + answer_text, "red"))
    return prompt
