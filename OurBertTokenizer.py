import numpy as np
import re
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer
import unicodedata
import random


class MyTokenizer():
    def __init__(self,
                 data,
                 max_len,
                 padding,
                 truncating):
        
        self.data = data
        self.max_len = max_len
        self.padding = padding
        self.truncating = truncating
        
        self.tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        

        random.seed(0)
        np.random.seed(0)
    
    def sentence_split(self,reverse,to_lower):
        sentences = sent_tokenize(self.data)
        for i in range(len(sentences)):
            sentences[i] = re.sub(r'[^\w\s]', '', sentences[i])
        
        if to_lower:
            sentences = [sent.lower() for sent in sentences]
                    
        first_pair = sentences[0::2]
        second_pair = sentences[1::2]
        
        # This Part is for Next Sentence Prediction Classifier
        # First half of sentences list is correct order
        # Second half of sentences list is incorrect order
        batch_first_correct_half  = first_pair[:len(first_pair)//2]
        batch_second_correct_half = second_pair[:len(second_pair)//2]

        batch_first_incorrect_half  = first_pair[len(first_pair)//2:]
        batch_second_incorrect_half = second_pair[len(second_pair)//2:]

        random.shuffle(batch_first_incorrect_half)
        random.shuffle(batch_second_incorrect_half)
        
        first_pair = batch_first_correct_half+batch_first_incorrect_half
        second_pair = batch_second_correct_half+batch_second_incorrect_half
        
        if len(first_pair)!=len(second_pair):
            new_size = min(len(first_pair),len(second_pair))
            first_pair = first_pair[:new_size]
            second_pair = second_pair[:new_size]

        size = len(first_pair)
        #NSP_Label = [[1,0] if i<size//2 else [0,1] for i in range(size)]
        NSP_Label = [1 if i<size//2 else 0 for i in range(size)]
        
        if reverse:
            reverse_fist_pair,reverse_second_pair = [],[]
            for s1,s2 in zip(first_pair,second_pair):
                reverse_fist_pair.append(self.return_reverse(s1))
                reverse_second_pair.append(self.return_reverse(s2))
            return (NSP_Label,reverse_fist_pair,reverse_second_pair)
            
        return (NSP_Label,first_pair,second_pair)
    
    # It returns sentences indication. 
    # First pairs sentences is 0, second is 1
    def return_token_type_ids(self,input_id):
        segment = []
        s = 0
        for sentence_ids in input_id:
            sentence_segment = []
            for ids in sentence_ids:
                if ids == 0:
                    s=0
                    
                sentence_segment.append(s)
                
                if ids == 3 and s == 0:
                    s=1
                elif ids == 3 and s == 1:
                    s=0
                    
            segment.append(sentence_segment)
            
        #for attention mask
        attention = []
        a = 0
        for sentence_ids in input_id:
            sentence_attention = []
            for ids in sentence_ids:
                if ids == 0:
                    a = 0
                else:
                    a = 1
                sentence_attention.append(a)
            
            attention.append(sentence_attention)   
        
        output = np.array(segment)+np.array(attention)
        return output.tolist()
            
    
    def return_input_ids(self):
        (_,first_batch,second_batch) = self.sentence_split(self.reverse,self.to_lower)
        encoded_inputs = self.tokenizer(first_batch, 
                                   second_batch,
                                   padding=self.padding,
                                   truncation=self.truncating,
                                   max_length=self.max_len)
        return encoded_inputs['input_ids']
    
    # convert ids array to sentence
    def decode_sentences(self,ids):
        return self.tokenizer.decode(ids)
    
    def tokenize_text(self,text):
        return self.tokenizer.tokenize(text)
    
    def id_to_token(self,Id):
        return self.tokenizer.ids_to_tokens[Id]
    
    def _is_punctuation(self,char):
        cp = ord(char)
        if ((cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or
              (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126)):
            return True
        cat = unicodedata.category(char)
        if cat.startswith("P"):
            return True
        return False
    
    # return reverse of sentence
    def return_reverse(self,text):
        chars = list(text)
        chars = chars[::-1]
        i = 0
        start_new_word = True
        output = []
        while i < len(chars):
            char = chars[i]
            if self._is_punctuation(char):
                output.append([char])
                start_new_word = True
            else:
                if start_new_word or chars[i]==' ':
                    if len(output)>=2:
                        output[-1] = output[-1][::-1]
                    output.append([])
                start_new_word = False
                output[-1].append(char)
            i += 1
        output[-1] = output[-1][::-1]
        reverse_list = ["".join(x) for x in output]
        reverse_sentence = " ".join(reverse_list)
        reverse_sentence = re.sub(" +"," ",reverse_sentence)
        return reverse_sentence
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def masking_sentences_id(self,input_ids):
        masked_index_id = []
        masked_label = []
        for index,ids in enumerate(input_ids):
            seq_len = len([i for i in ids if i>max(self.tokenizer.all_special_ids)])
            
            words2mask = (15*seq_len)//100
            words2mask_Mask = (80*words2mask)//100
            words2mask_realWord = (10*words2mask)//100
            words2mask_randomWord = (10*words2mask)//100
            
            
            seq_index_mask = []
            seq_mask_label = []
            
            # if sentences is too low. it mask only first word
            if words2mask_Mask==0:
                select = 1
                seq_mask_label.append(ids[select])
                input_ids[index][select] = self.tokenizer.mask_token_id
                seq_index_mask.append(select)
                masked_index_id.append(seq_index_mask)
                masked_label.append(seq_mask_label)
                continue
            
            
            if(words2mask_randomWord == 0):
                random = np.random.randint(2)
                if random==1:
                    words2mask_realWord = 1
                else:
                    words2mask_randomWord = 1
            
            # for real masking
            i = 0
            while i<words2mask_Mask:
                random = np.random.randint(len(ids))
                random_state = ids[random]
                if random_state not in self.tokenizer.all_special_ids and random not in seq_index_mask:
                    i = i+1
                    seq_mask_label.append(ids[random])
                    input_ids[index][random] = self.tokenizer.mask_token_id
                    seq_index_mask.append(random)

            # for masking with real word
            i = 0
            while i<words2mask_realWord:
                random = np.random.randint(len(ids))
                random_state = ids[random]
                if random_state not in self.tokenizer.all_special_ids and random not in seq_index_mask:
                    i = i+1
                    seq_mask_label.append(ids[random])
                    input_ids[index][random] = ids[random]
                    seq_index_mask.append(random)

            # for masking with random word
            i = 0
            while i<words2mask_randomWord:
                random = np.random.randint(len(ids))
                random_state = ids[random]
                if random_state not in self.tokenizer.all_special_ids and random not in seq_index_mask:
                    i = i+1
                    seq_mask_label.append(ids[random])
                    input_ids[index][random] = np.random.randint(max(self.tokenizer.all_special_ids)+1,self.tokenizer.vocab_size-1)
                    seq_index_mask.append(random)

            masked_index_id.append(seq_index_mask)
            masked_label.append(seq_mask_label)
            
        #input ids is masked at the end of the code
        return input_ids,masked_index_id,masked_label
    
    def shuffle_input(self,masked_input_ids,masked_index,masked_label,NSP_Label,token_type):
        
        Shuffle = list(zip(masked_input_ids,masked_index,masked_label,NSP_Label,token_type))
        random.shuffle(Shuffle)
        (masked_input_ids,masked_index,masked_label,NSP_Label,token_type) = zip(*Shuffle)
        
        return (list(masked_input_ids),list(masked_index),list(masked_label),list(NSP_Label),list(token_type))

    
    def __call__(self,data_call,reverse,to_lower):
        self.data = data_call
        (self.reverse,self.to_lower) = (reverse,to_lower)
        
        (self.NSP_Label,self.first_pair,self.second_pair) = self.sentence_split(self.reverse,self.to_lower)
        self.input_ids = self.return_input_ids()
        self.token_type = self.return_token_type_ids(self.input_ids)
        
        (self.masked_input_ids,self.masked_index,self.masked_label) = self.masking_sentences_id(self.input_ids)
        
        (self.shuffled_masked_input_ids,
         self.shuffled_masked_index,
         self.shuffled_masked_label,
         self.shuffled_NSP_Label,
         self.shuffled_token_type) = self.shuffle_input(self.masked_input_ids,
                                                        self.masked_index,
                                                        self.masked_label,
                                                        self.NSP_Label,
                                                        self.token_type)
        
        return (self.shuffled_masked_input_ids,
                self.shuffled_masked_index,
                self.shuffled_masked_label,
                self.shuffled_NSP_Label,
                self.shuffled_token_type)
         
         
         
         
         
         