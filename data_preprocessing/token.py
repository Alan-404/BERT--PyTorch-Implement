from enum import Enum

class Token(Enum):
    CLS_TOKEN = "__CLS__"
    SEP_TOKEN = "__SEP__"
    MASK_TOKEN = "__MASK__"
    PAD_TOKEN = "__PAD__"
    START_TOKEN = "__START__"
    END_TOKEN = "__END__"