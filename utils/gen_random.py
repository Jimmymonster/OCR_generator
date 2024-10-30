import random
import string

# Define Thai Unicode range
THAI_CHARACTERS =[chr(i) for i in range(0x0E01, 0x0E3A + 1)] + [chr(i) for i in range(0x0E40, 0x0E4D + 1)]
# Define English character range (uppercase and lowercase letters)
ENGLISH_CHARACTERS = string.ascii_letters

def generate_random_strings(language, length, count):
    if language == "thai":
        character_set = THAI_CHARACTERS
    elif language == "english":
        character_set = ENGLISH_CHARACTERS
    else:
        raise ValueError("Language must be either 'thai' or 'english'")
    
    strings = []
    for _ in range(count):
        random_string = ''.join(random.choice(character_set) for _ in range(length))
        strings.append(random_string)
        
    return strings