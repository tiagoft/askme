import fitz 
import re 
from tqdm import tqdm 

patterns = {
    "˜a" : "ã",
    "˜A" : "Ã",
    "˜e" : "ẽ",
    "˜E" : "Ẽ",
    "˜i" : "ĩ",
    "˜I" : "Ĩ",
    "˜o" : "õ",
    "˜O" : "Õ",
    "˜u" : "ũ",
    "˜U" : "Ũ",
    "c¸" : "ç",
    "C¸" : "Ç",
    "´a" : "á",
    "´A" : "Á",
    "´e" : "é",
    "´E" : "É",
    "´i" : "í",
    "´I" : "Í",
    "´o" : "ó",
    "´O" : "Ó",
    "´u" : "ú",
    "´U" : "Ú",
    "ˆa" : "â",
    "ˆA" : "Â",
    "ˆe" : "ê",
    "ˆE" : "Ê",
    "ˆo" : "ô",
    "ˆO" : "Ô",
    "`a" : "à",
    "`A" : "À",
    "´ı" : "í",
    "´I" : "Í",
}

def replace_accent_patterns(text, maps):
    for pattern, replacement in maps.items():
        # Use regex to replace the pattern with the replacement
        text = re.sub(re.escape(pattern), replacement, text)
    return text

def get_document(path):
    with fitz.open(path) as doc:
        
        document = ""
        document_paginated = []
        started = False
        for page in doc:
            #if not started and "Resumo" in page.get_text():
            #    started = True
                
            #if started:
            #print(page.get_text())
            fixed_text = replace_accent_patterns(page.get_text(), patterns)
            document += fixed_text
            document_paginated.append(fixed_text)
    return document, document_paginated

