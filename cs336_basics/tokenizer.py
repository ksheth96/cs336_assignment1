from collections.abc import Iterable, Iterator
import regex as re

class BPETokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ) -> None:
        self.vocab = vocab
        self.merges = merges
        # self.merges_dict = {merges[i]:i for i in range(len(merges))}
        self.special_tokens = special_tokens
        self.reverse_vocab = {v:k for k,v in vocab.items()}
        self.already_encoded_dict = {}
        self.special_set = set(self.special_tokens) if self.special_tokens else set()
        
        if self.special_set:
            # CRITICAL: Sort tokens by length descending so longer substrings match first
            sorted_special_tokens = sorted(list(self.special_set), key=len, reverse=True)
            # Construct pattern using the sorted list
            self.special_tokens_pat = f"({f'|'.join([re.escape(tok) for tok in sorted_special_tokens])})"            
        else:
            self.special_tokens_pat = None
                

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None = None,
    ) -> "BPETokenizer":
        raise NotImplementedError
    
    def __pretokenize(self, text: str) -> list[str]:

        PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
        if self.special_tokens_pat is not None:
            documents = re.split(self.special_tokens_pat, text)
            documents = [doc for doc in documents if doc]
        else:
            documents = [text]

        pre_tokens = []
        for doc in documents:
            if doc in self.special_set:
                pre_tokens.append(doc)
            else:
                pre_tokens.extend(re.findall(PAT, doc))
        return pre_tokens


    def encode(self, text: str) -> list[int]:
        
        pre_tokens = self.__pretokenize(text=text)
        encoded_text = []
        
        for pre_token in pre_tokens:
            if pre_token in self.already_encoded_dict:
                encoded_text.extend(self.already_encoded_dict[pre_token])
                continue
            if self.special_tokens is not None and pre_token in self.special_tokens:
                encoded_text.append(self.reverse_vocab[pre_token.encode(encoding="utf-8")])
                continue
            
            pre_token_bytes = list(map(bytes,([x] for x in pre_token.encode(encoding="utf-8"))))
            for merge in self.merges:
                i=0
                while i < len(pre_token_bytes)-1:
                    if (pre_token_bytes[i],pre_token_bytes[i+1])==merge:
                        left_half = []
                        right_half = []
                        if i==0:
                            left_half = []
                        else:
                            left_half = pre_token_bytes[:i]
                        if i+2>len(pre_token_bytes)-1:
                            right_half = []
                        else:
                            right_half = pre_token_bytes[i+2:]
                        pre_token_bytes = left_half+[merge[0]+merge[1]]+right_half
                    i+=1
            self.already_encoded_dict[pre_token] = list(self.reverse_vocab[bytes_obj] for bytes_obj in pre_token_bytes)
            encoded_text.extend(self.already_encoded_dict[pre_token])
        return encoded_text

    
    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for chunk in iterable:
            yield from self.encode(chunk)


    def decode(self, ids: list[int]) -> str:

        byte_sequence = b"".join([self.vocab[id] for id in ids])
        return byte_sequence.decode(encoding="utf-8",errors='replace')
