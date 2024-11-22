import json
from config import DATASET_PATH, VOCAB_SIZE
from preprocessor import AmharicPreprocessor
from tokenizers import Tokenizer, models, pre_tokenizers, decoders, trainers, processors


def train_tokenizer(texts: list[str]) -> Tokenizer:
    preprocessor = AmharicPreprocessor(None)
    all_texts = [preprocessor.preprocess(text, encode=False) for text in texts]

    # Initialize a tokenizer
    tokenizer = Tokenizer(models.BPE(unk_token='[UNK]'))
    
    # Customize pre-tokenization and decoding
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(use_regex=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.post_processor = processors.ByteLevel()
    
    # Amharic chars and the arabic numerals
    alphabet = ["ሀ", "ሁ", "ሂ", "ሃ", "ሄ", "ህ", "ሆ", "ሇ", "ለ", "ሉ", "ሊ", "ላ", "ሌ", "ል", "ሎ", "ሏ", "ሐ", "ሑ", "ሒ", "ሓ", "ሔ", "ሕ", "ሖ", "ሗ", "መ", "ሙ", "ሚ", "ማ", "ሜ", "ም", "ሞ", "ሟ", "ሠ", "ሡ", "ሢ", "ሣ", "ሤ", "ሥ", "ሦ", "ሧ", "ረ", "ሩ", "ሪ", "ራ", "ሬ", "ር", "ሮ", "ሯ", "ሰ", "ሱ", "ሲ", "ሳ", "ሴ", "ስ", "ሶ", "ሷ", "ሸ", "ሹ", "ሺ", "ሻ", "ሼ", "ሽ", "ሾ", "ሿ", "ቀ", "ቁ", "ቂ", "ቃ", "ቄ", "ቅ", "ቆ", "ቇ", "ቈ", "ቊ", "ቋ", "ቌ", "ቍ", "በ", "ቡ", "ቢ", "ባ", "ቤ", "ብ", "ቦ", "ቧ", "ቨ", "ቩ", "ቪ", "ቫ", "ቬ", "ቭ", "ቮ", "ቯ", "ተ", "ቱ", "ቲ", "ታ", "ቴ", "ት", "ቶ", "ቷ", "ቸ", "ቹ", "ቺ", "ቻ", "ቼ", "ች", "ቾ", "ቿ", "ኀ", "ኁ", "ኂ", "ኃ", "ኄ", "ኅ", "ኆ", "ኇ", "ኈ", "ኊ", "ኋ", "ኌ", "ኍ", "ነ", "ኑ", "ኒ", "ና", "ኔ", "ን", "ኖ", "ኗ", "ኘ", "ኙ", "ኚ", "ኛ", "ኜ", "ኝ", "ኞ", "ኟ", "አ", "ኡ", "ኢ", "ኣ", "ኤ", "እ", "ኦ", "ኧ", "ከ", "ኩ", "ኪ", "ካ", "ኬ", "ክ", "ኮ", "ኯ", "ኰ", "ኲ", "ኳ", "ኴ", "ኵ", "ኸ", "ኹ", "ኺ", "ኻ", "ኼ", "ኽ", "ኾ", "ወ", "ዉ", "ዊ", "ዋ", "ዌ", "ው", "ዎ", "ዐ", "ዑ", "ዒ", "ዓ", "ዔ", "ዕ", "ዖ", "ዘ", "ዙ", "ዚ", "ዛ", "ዜ", "ዝ", "ዞ", "ዟ", "ዠ", "ዡ", "ዢ", "ዣ", "ዤ", "ዥ", "ዦ", "ዧ", "የ", "ዩ", "ዪ", "ያ", "ዬ", "ይ", "ዮ", "ደ", "ዱ", "ዲ", "ዳ", "ዴ", "ድ", "ዶ", "ዷ", "ጀ", "ጁ", "ጂ", "ጃ", "ጄ", "ጅ", "ጆ", "ጇ", "ገ", "ጉ", "ጊ", "ጋ", "ጌ", "ግ", "ጎ", "ጏ", "ጠ", "ጡ", "ጢ", "ጣ", "ጤ", "ጥ", "ጦ", "ጧ", "ጨ", "ጩ", "ጪ", "ጫ", "ጬ", "ጭ", "ጮ", "ጯ", "ጰ", "ጱ", "ጲ", "ጳ", "ጴ", "ጵ", "ጶ", "ጷ", "ጸ", "ጹ", "ጺ", "ጻ", "ጼ", "ጽ", "ጾ", "ጿ", "ፀ", "ፁ", "ፂ", "ፃ", "ፄ", "ፅ", "ፆ", "ፇ", "ፈ", "ፉ", "ፊ", "ፋ", "ፌ", "ፍ", "ፎ", "ፏ", "ፐ", "ፑ", "ፒ", "ፓ", "ፔ", "ፕ", "ፖ" , "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H","I","J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        
    # Add punctuations and special symbols
    alphabet += ["!", "@", "#", "$", "%", "^", "«", "»", "&", "?", "*", "(", ")", "…", "[", "]", "{", "}", ";", "“", "”", "›", "’", "‘", '"', "'", ":", ",", ".", "‹", "/", "<", ">", "\\", "\\", "|", "`", "´", "~", "-", "=", "+", "፡", "።", "፤", ";", "፦", "፥", "፧", "፨", "፠", "፣"]
        
    # Train on dataset
    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=["[UNK]", "[PAD]"],
        initial_alphabet=alphabet,
        min_frequency=2,
        show_progress=True
    )
    tokenizer.train_from_iterator(all_texts, trainer=trainer)
    
    return tokenizer


if __name__ == "__main__":
    with open(DATASET_PATH, 'r', encoding='utf-8') as file:
        texts = file.readlines()
    
    tokenizer = train_tokenizer(texts)
    
    tokenizer.save(f"./tokenizers/amharic-bpe-tokenizer-v1-{VOCAB_SIZE // 1000}k.json")