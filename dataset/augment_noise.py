import copy
from typing import List, Tuple, Set, Dict
from collections import defaultdict

class LogAugmenter:
    def __init__(self):
        # Define punctuation hierarchy and valid replacements
        self.punct_hierarchy = {
            '.': [':', ','],  # terminal can become strong pause or weak pause
            ':': [','],       # strong pause can become weak pause
            ',': []          # weak pause stays as is
        }
        self.all_puncts = set(['.', ':', ','])
        
    def is_intermediary_punct(self, 
                            tokens: List[Tuple[str, str]], 
                            idx: int) -> bool:
        """Check if token at idx is an intermediary punctuation"""
        token = tokens[idx][0]
        # Check if it's a punctuation and not at the end of message
        return (token in self.all_puncts and 
                idx < len(tokens) - 1)

    def get_intermediary_puncts(self, 
                              tokens: List[Tuple[str, str]]) -> List[Tuple[int, str]]:
        """Get positions and values of all intermediary punctuations"""
        return [(i, tokens[i][0]) for i in range(len(tokens)) 
                if self.is_intermediary_punct(tokens, i)]

    def remove_punctuation(self, 
                         message: List[Tuple[str, str]]) -> List[Tuple[str, str]]:
        """Remove all intermediary punctuations from the message"""
        # Create a new message without intermediary punctuations
        augmented = []
        for i, (token, tag) in enumerate(message):
            if not self.is_intermediary_punct(message, i):
                augmented.append((token, tag))
        return augmented

    def replace_punctuation(self, 
                          message: List[Tuple[str, str]]) -> List[List[Tuple[str, str]]]:
        """Generate variations by replacing punctuations following linguistic rules"""
        intermediary_puncts = self.get_intermediary_puncts(message)
        if not intermediary_puncts:
            return []

        augmented_messages = []
        
        # For each intermediary punctuation
        for idx, punct in intermediary_puncts:
            # Get valid replacements for this punctuation
            replacements = self.punct_hierarchy.get(punct, [])
            
            # Generate a new message for each valid replacement
            for replacement in replacements:
                new_message = copy.deepcopy(message)
                new_message[idx] = (replacement, new_message[idx][1])
                augmented_messages.append(new_message)

        return augmented_messages

    def augment_message(self, 
                       message: List[Tuple[str, str]], 
                       remove_punct: bool = True,
                       replace_punct: bool = True) -> List[List[Tuple[str, str]]]:
        """Generate all augmentations for a message"""
        augmented_messages = []
        
        # Add original message
        augmented_messages.append(message)
        
        # Add punctuation removal augmentation
        if remove_punct:
            removed = self.remove_punctuation(message)
            if removed != message:
                augmented_messages.append(removed)
        
        # Add punctuation replacement augmentations
        if replace_punct:
            replaced = self.replace_punctuation(message)
            augmented_messages.extend(replaced)
        
        return augmented_messages

def format_message(tokens: List[Tuple[str, str]]) -> str:
    """Format message for readable output"""
    return ' '.join(token for token, _ in tokens)

# Example usage and testing
def test_augmenter():
    # Create test message: [(token, tag), ...]
    test_message = [
        ("System", "O"),
        ("started", "O"),
        (".", "O"),
        ("Processing", "O"),
        ("data", "O"),
        (":", "O"),
        ("analysis", "O"),
        ("complete", "O"),
        (".", "O")
    ]
    
    augmenter = LogAugmenter()
    augmented_messages = augmenter.augment_message(test_message)
    
    print("Original message:")
    print(format_message(test_message))
    print("\nAugmented messages:")
    for idx, aug_message in enumerate(augmented_messages[1:], 1):
        print(f"{idx}. {format_message(aug_message)}")

def augment_dataset(input_file: str, 
                   output_file: str, 
                   sample_ratio: float = 1.0):
    """Augment entire dataset and save to CoNLL format"""
    import random
    
    augmenter = LogAugmenter()
    messages = []
    current_message = []
    
    # Read input file
    with open(input_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line == '':
                if current_message:
                    messages.append(current_message)
                    current_message = []
            else:
                token, tag = line.split()
                current_message.append((token, tag))
        if current_message:
            messages.append(current_message)
    
    # Select messages to augment
    num_to_augment = int(len(messages) * sample_ratio)
    messages_to_augment = random.sample(messages, num_to_augment)
    
    # Augment selected messages
    all_augmented = []
    for message in messages:
        all_augmented.append(message)  # Keep original
        if message in messages_to_augment:
            augmented = augmenter.augment_message(message)[1:]  # Skip original
            all_augmented.extend(augmented)
    
    # Write augmented dataset
    with open(output_file, 'w') as f:
        for message in all_augmented:
            for token, tag in message:
                f.write(f"{token} {tag}\n")
            f.write("\n")  # Empty line between messages

if __name__ == "__main__":
    # Example usage
    # test_augmenter()
    
    # Augment actual dataset
    augment_dataset('train_conll_data.txt', 'train_augmented.txt', sample_ratio=0.2)