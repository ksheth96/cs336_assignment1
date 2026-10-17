## Problem 1
a. '\x00'
b. __repr__() returns "'\\x00'"
c. its escape sequence for null byte

## Problem 2
a. UTF-16 and UTF-32 inject frequent null bytes (00) for standard ASCII text. If you train a tokenizer on these encodings, the algorithm will waste dozens of vocabulary slots trying to merge common characters with their accompanying padding bytes.
b. function wrong as utf-8 is not a per unicode character mapping, its a variable length encoding.
c. bytes([200,200]) fails as first is start byte and second is not a continuation byte and a start byte as well for a multibyte char.

