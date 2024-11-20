import re
import nltk
from tokenizers import Tokenizer
from nltk.corpus import stopwords
from abc import ABC, abstractmethod
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


class PreprocessingPipeline(ABC):   
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__()
        self.tokenizer = tokenizer
        
    def tokenize(self, text):
        """
        Tokenize the input text into words.
        """
        words = word_tokenize(text)
        return words
    
    @abstractmethod
    def preprocess(self, text: str, encode=True) -> str:
        pass

    
class AmharicPreprocessor(PreprocessingPipeline):
    def __init__(self, tokenizer: Tokenizer) -> None:
        super().__init__(tokenizer)
    
    def preprocess(self, text: str, encode=True) -> str:
        # Character level mismatch
        text = self.normalize_char_level_missmatch(text)
        
        # Replace commonly used abbreviations
        text = self.normalize_abbreviations(text)
        
        # Remove punctuations and special characters
        # text = self.remove_punc_and_special_chars(text)
        
        # Remove non-amharic chars and arabic numbers
        # text = self.remove_ascii_and_numbers(text)
        
        if encode:
            return self.tokenizer.encode(
                text,
            ).ids
        else:
            return text
            
    # Remove abbreviations
    def normalize_abbreviations(self, text: str) -> str:
        common_amharic_abbreviations = {
            "ት/ቤት": "ትምህርት ቤት",
            "ት/ርት": "ትምህርት",
            "ት/ክፍል": "ትምህርት ክፍል",
            "ሃ/አለቃ": "ሃምሳ አለቃ",
            "ሃ/ስላሴ": "ሃይለ ስላሴ",
            "ደ/ዘይት": "ደብረ ዘይት",
            "ደ/ታቦር": "ደብረ ታቦር",
            "መ/ር": "መምህር",
            "መ/ቤት": "መስሪያ ቤት",
            "መ/አለቃ": "መቶ አለቃ",
            "ክ/ከተማ": "ክፍለ ከተማ",
            "ክ/ሀገር": "ክፍለ ሀገር",
            "ወ/ር": "",
            "ወ/ሮ": "ወይዘሮ",
            "ወ/ሪት": "ወይዘሪት",
            "ወ/ስላሴ": "ወልደ ስላሴ",
            "ፍ/ስላሴ": "ፍቅረ ስላሴ",
            "ፍ/ቤት": "ፍርድ ቤት",
            "ጽ/ቤት": "ጽህፈት ቤት",
            "ሲ/ር": "",
            "ፕ/ር": "ፕሮፌሰር",
            "ጠ/ሚንስትር": "ጠቅላይ ሚኒስተር",
            "ጠ/ሚ": "ጠቅላይ ሚኒስተር",
            "ዶ/ር": "ዶክተር",
            "ገ/ገዮርጊስ": "ገብረ ገዮርጊስ",
            "ቤ/ክርስትያን": "ቤተ ክርስትያን",
            "ም/ስራ": "",
            "ም/ቤት": "ምክር ቤተ",
            "ተ/ሃይማኖት": "ተክለ ሃይማኖት",
            "ሚ/ር": "ሚኒስትር",
            "ኮ/ል": "ኮሎኔል",
            "ሜ/ጀነራል": "ሜጀር ጀነራል",
            "ብ/ጀነራል": "ብርጋደር ጀነራል",
            "ሌ/ኮለኔል": "ሌተናንት ኮለኔል",
            "ሊ/መንበር": "ሊቀ መንበር",
            "አ/አ": "ኣዲስ ኣበባ",
            "አ.አ": "ኣዲስ ኣበባ",
            "ር/መምህር": "ርዕሰ መምህር",
            "ፕ/ት": "",
            "ዓም": "ዓመተ ምህረት",
            "ዓ.ዓ": "ዓመተ ዓለም",
        }
        for key in common_amharic_abbreviations:
            regex = rf'\b{re.escape(key)}\b'
            text = re.sub(regex, common_amharic_abbreviations[key], text)

        # Remove punctuation, numbers, and extra spaces
        text = re.sub(r'[.\?"\',/#!$%^&*;:፤።{}=\-_`~()፩፪፫፬፭፮፮፰፱፲፳፴፵፵፷፸፹፺፻01-9]', ' ', text)
        text = re.sub(r'\s{2,}', ' ', text)

        return text

    #method to normalize character level missmatch such as ጸሀይ and ፀሐይ
    def normalize_char_level_missmatch(self, text: str) -> str:
        rep1=re.sub('[ሃኅኃሐሓኻ]','ሀ',text)
        rep2=re.sub('[ሑኁዅ]','ሁ',rep1)
        rep3=re.sub('[ኂሒኺ]','ሂ',rep2)
        rep4=re.sub('[ኌሔዄ]','ሄ',rep3)
        rep5=re.sub('[ሕኅ]','ህ',rep4)
        rep6=re.sub('[ኆሖኾ]','ሆ',rep5)
        rep7=re.sub('[ሠ]','ሰ',rep6)
        rep8=re.sub('[ሡ]','ሱ',rep7)
        rep9=re.sub('[ሢ]','ሲ',rep8)
        rep10=re.sub('[ሣ]','ሳ',rep9)
        rep11=re.sub('[ሤ]','ሴ',rep10)
        rep12=re.sub('[ሥ]','ስ',rep11)
        rep13=re.sub('[ሦ]','ሶ',rep12)
        rep14=re.sub('[ዓኣዐ]','አ',rep13)
        rep15=re.sub('[ዑ]','ኡ',rep14)
        rep16=re.sub('[ዒ]','ኢ',rep15)
        rep17=re.sub('[ዔ]','ኤ',rep16)
        rep18=re.sub('[ዕ]','እ',rep17)
        rep19=re.sub('[ዖ]','ኦ',rep18)
        rep20=re.sub('[ጸ]','ፀ',rep19)
        rep21=re.sub('[ጹ]','ፁ',rep20)
        rep22=re.sub('[ጺ]','ፂ',rep21)
        rep23=re.sub('[ጻ]','ፃ',rep22)
        rep24=re.sub('[ጼ]','ፄ',rep23)
        rep25=re.sub('[ጽ]','ፅ',rep24)
        rep26=re.sub('[ጾ]','ፆ',rep25)
        #Normalizing words with Labialized Amharic characters such as በልቱዋል or  በልቱአል to  በልቷል  
        rep27=re.sub('(ሉ[ዋአ])','ሏ',rep26)
        rep28=re.sub('(ሙ[ዋአ])','ሟ',rep27)
        rep29=re.sub('(ቱ[ዋአ])','ቷ',rep28)
        rep30=re.sub('(ሩ[ዋአ])','ሯ',rep29)
        rep31=re.sub('(ሱ[ዋአ])','ሷ',rep30)
        rep32=re.sub('(ሹ[ዋአ])','ሿ',rep31)
        rep33=re.sub('(ቁ[ዋአ])','ቋ',rep32)
        rep34=re.sub('(ቡ[ዋአ])','ቧ',rep33)
        rep35=re.sub('(ቹ[ዋአ])','ቿ',rep34)
        rep36=re.sub('(ሁ[ዋአ])','ኋ',rep35)
        rep37=re.sub('(ኑ[ዋአ])','ኗ',rep36)
        rep38=re.sub('(ኙ[ዋአ])','ኟ',rep37)
        rep39=re.sub('(ኩ[ዋአ])','ኳ',rep38)
        rep40=re.sub('(ዙ[ዋአ])','ዟ',rep39)
        rep41=re.sub('(ጉ[ዋአ])','ጓ',rep40)
        rep42=re.sub('(ደ[ዋአ])','ዷ',rep41)
        rep43=re.sub('(ጡ[ዋአ])','ጧ',rep42)
        rep44=re.sub('(ጩ[ዋአ])','ጯ',rep43)
        rep45=re.sub('(ጹ[ዋአ])','ጿ',rep44)
        rep46=re.sub('(ፉ[ዋአ])','ፏ',rep45)
        rep47=re.sub('[ቊ]','ቁ',rep46) #ቁ can be written as ቊ
        rep48=re.sub('[ኵ]','ኩ',rep47) #ኩ can be also written as ኵ  
        
        return rep48

    #replacing any existance of special character or punctuation to null  
    def remove_punc_and_special_chars(self, text: str) -> str: # puct in amh =፡።፤;፦፧፨፠፣ 
        normalized_text = re.sub('[\!\@\#\$\%\^\«\»\&\*\(\)\…\[\]\{\}\;\“\”\›\’\‘\"\'\:\,\.\‹\/\<\>\?\\\\|\`\´\~\-\=\+\፡\።\፤\;\፦\፥\፧\፨\፠\፣]', '', text) 
        return normalized_text

    #remove all ascii characters and Arabic and Amharic numbers
    def remove_ascii_and_numbers(self, text: str) -> str:
        rm_num_and_ascii=re.sub('[A-Za-z0-9]','',text)
        return re.sub('[^\u1200-\u137F\s]+','',rm_num_and_ascii)
