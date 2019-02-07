import re
REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^0-9a-z #+_]')
STOPWORDS = set(stopwords.words('english'))

def text_prepare(text):
    """
        text: a string
        
        return: modified initial string
    """
    text = text.lower()
    text = re.sub(REPLACE_BY_SPACE_RE," ", text) # replace REPLACE_BY_SPACE_RE symbols by space in text
    text = re.sub(BAD_SYMBOLS_RE,"",text) # delete symbols which are in BAD_SYMBOLS_RE from text
    text = ' '.join([word for word in list(text.split()) if not word in STOPWORDS])# delete stopwords from text
    return text


a = "Hello{}World"
a = re.sub(REPLACE_BY_SPACE_RE," ", a)
print(a)

import numpy as np

a = np.ones([3,3],dtype=np.int32)*4

print(a)

