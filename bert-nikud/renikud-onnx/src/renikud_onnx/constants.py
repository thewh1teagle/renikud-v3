"""
Hebrew Unicode constants for nikud marks.
"""

# Vowel marks
A_PATAH = '\u05b7'  # a
E_TSERE = '\u05b5'  # e
I_HIRIK = '\u05b4'  # i
O_HOLAM = '\u05b9'  # o
U_QUBUT = '\u05bb'  # u
SHVA = '\u05b0'  # silent shva
E_VOCAL_SHVA = '\u05bd'  # vocal shva (meteg) - sounds like e

# Other marks
DAGESH = '\u05bc'
S_SIN = '\u05c2'
STRESS_HATAMA = '\u05ab'
PREFIX_SEP = '|'

# Character sets
FINAL_LETTERS = 'ךםןףץ'
CAN_HAVE_DAGESH = 'בכפו'
CAN_HAVE_SIN = 'ש'
CAN_NOT_HAVE_NIKUD = "םןףץ"
LETTERS = 'אבגדהוזחטיכלמנסעפצקרשת' + FINAL_LETTERS

# Vowel mappings
ID_TO_VOWEL = {
    0: None,  # No vowel
    1: A_PATAH,
    2: E_TSERE,
    3: I_HIRIK,
    4: O_HOLAM,
    5: U_QUBUT,
    6: SHVA,
    7: E_VOCAL_SHVA,
}

