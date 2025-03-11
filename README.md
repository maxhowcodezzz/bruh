# bruh
Wordle game
import random
from collections import Counter

class WordleAI:
    def __init__(self, word_list_file="english_words.txt"):
        # Load dictionary of 5-letter words
        self.word_list = self._load_dictionary(word_list_file)
        self.possible_words = self.word_list.copy()
        self.guessed_words = []
        self.correct_positions = {}  # Known correct letters and positions
        self.contains_letters = set()  # Letters known to be in the word
        self.excluded_letters = set()  # Letters known not to be in the word
        self.excluded_positions = {}  # Letters in wrong positions
        
    def _load_dictionary(self, filename):
        """Load dictionary file or use fallback list of common 5-letter words."""
        try:
            with open(filename, 'r') as f:
                words = [word.strip().lower() for word in f if len(word.strip()) == 5]
            return words
        except FileNotFoundError:
            # Fallback to a small list of common 5-letter words
            return [
                "about", "allow", "apple", "beach", "bound", "brain", 
                "brave", "chair", "check", "cloud", "eight", "force", 
                "games", "great", "hotel", "house", "input", "judge", 
                "light", "major", "match", "money", "paper", "party", 
                "plate", "power", "price", "queen", "right", "river", 
                "seven", "sleep", "sound", "south", "space", "table", 
                "thank", "train", "water", "where", "which", "world"
            ]
    
    def update_knowledge(self, guess, feedback):
        """
        Update knowledge based on guess and feedback.
        Feedback should be a string where:
        - 'g' means correct letter in correct position (green)
        - 'y' means correct letter in wrong position (yellow)
        - 'b' means letter not in the word (black/gray)
        """
        self.guessed_words.append(guess)
        
        # Process feedback
        for i, (letter, result) in enumerate(zip(guess, feedback)):
            if result == 'g':
                # Correct letter in correct position
                self.correct_positions[i] = letter
                self.contains_letters.add(letter)
            elif result == 'y':
                # Correct letter in wrong position
                self.contains_letters.add(letter)
                
                # Add to excluded positions
                if i not in self.excluded_positions:
                    self.excluded_positions[i] = set()
                self.excluded_positions[i].add(letter)
            elif result == 'b':
                # Check if this letter appears elsewhere in the guess and is marked as yellow or green
                # If not, it's not in the word at all
                if letter not in [guess[j] for j, res in enumerate(feedback) if res in ['g', 'y']]:
                    self.excluded_letters.add(letter)
        
        # Update possible words based on new knowledge
        self._filter_possible_words()
    
    def _filter_possible_words(self):
        """Filter word list based on current knowledge."""
        filtered_words = []
        
        for word in self.possible_words:
            # Skip already guessed words
            if word in self.guessed_words:
                continue
                
            # Check correct positions
            if not all(word[pos] == letter for pos, letter in self.correct_positions.items()):
                continue
                
            # Check excluded letters
            if any(letter in self.excluded_letters for letter in word):
                continue
                
            # Check that word contains all the required letters
            if not all(letter in word for letter in self.contains_letters):
                continue
                
            # Check excluded positions
            if any(word[pos] in letters for pos, letters in self.excluded_positions.items()):
                continue
                
            filtered_words.append(word)
        
        self.possible_words = filtered_words
    
    def make_guess(self):
        """Make the best guess based on current knowledge."""
        if not self.possible_words:
            return None
            
        if not self.guessed_words:
            # First guess: use a word with common letters
            # "stare", "raise", "roate" are good starters
            starters = ["stare", "raise", "roate"]
            for starter in starters:
                if starter in self.possible_words:
                    return starter
                    
            # If no starters in the word list, use letter frequency
            return self._guess_by_letter_frequency()
        
        if len(self.possible_words) <= 2:
            # If only a few possibilities, just guess one
            return self.possible_words[0]
        
        # Otherwise, use letter frequency of remaining words
        return self._guess_by_letter_frequency()
    
    def _guess_by_letter_frequency(self):
        """Choose word based on letter frequency in possible words."""
        # Count letter frequencies by position
        position_counts = [{} for _ in range(5)]
        
        for word in self.possible_words:
            for i, letter in enumerate(word):
                if i not in self.correct_positions:  # Only count positions we don't know
                    position_counts[i][letter] = position_counts[i].get(letter, 0) + 1
        
        # Score each word
        best_score = -1
        best_word = None
        
        for word in self.possible_words:
            # Skip words with duplicate letters for efficiency in information gain
            if len(set(word)) < 5 and len(self.possible_words) > 10:
                continue
                
            score = 0
            seen_letters = set()
            
            for i, letter in enumerate(word):
                if letter not in seen_letters:
                    if i in self.correct_positions:
                        score += 5  # Bonus for letters we know are correct
                    else:
                        score += position_counts[i].get(letter, 0)
                    seen_letters.add(letter)
            
            if score > best_score:
                best_score = score
                best_word = word
        
        return best_word or random.choice(self.possible_words)
    
    def solve(self, target_word=None):
        """
        Solve the Wordle puzzle, either against a provided target word or interactively.
        If no target word is provided, the function will prompt for feedback after each guess.
        """
        max_attempts = 6
        for attempt in range(1, max_attempts + 1):
            guess = self.make_guess()
            
            if not guess:
                print("No valid words found with current constraints.")
                return False
                
            print(f"Attempt {attempt}: {guess}")
            
            if target_word:
                # Automatic feedback
                feedback = ""
                for i, letter in enumerate(guess):
                    if letter == target_word[i]:
                        feedback += "g"
                    elif letter in target_word:
                        feedback += "y"
                    else:
                        feedback += "b"
                print(f"Feedback: {feedback}")
            else:
                # Get feedback from user
                feedback = input("Enter feedback (g=green, y=yellow, b=black): ").lower()
            
            # If all green, we've won
            if feedback == "ggggg":
                print(f"Solved in {attempt} attempts!")
                return True
                
            # Update knowledge based on feedback
            self.update_knowledge(guess, feedback)
            
            print(f"Possible words remaining: {len(self.possible_words)}")
            if len(self.possible_words) < 10:
                print(f"Possibilities: {', '.join(self.possible_words)}")
            
        print("Failed to solve in 6 attempts.")
        if target_word:
            print(f"The word was: {target_word}")
        return False


# Example usage
if __name__ == "__main__":
    # Uncomment to solve automatically with a known target word
    # ai = WordleAI()
    # ai.solve(target_word="light")
    
    # Interactive mode
    print("Wordle AI Solver")
    print("----------------")
    print("When prompted for feedback, use:")
    print("  g = green (correct letter in correct position)")
    print("  y = yellow (correct letter in wrong position)")
    print("  b = black/gray (letter not in word)")
    print("Example: if you guess 'stare' and feedback is 'gybbg', enter 'gybbg'")
    print("----------------")
    
    ai = WordleAI()
    ai.solve()
Connections
import numpy as np
from collections import Counter
import nltk
from nltk.corpus import wordnet as wn
import gensim.downloader
import random

class ConnectionsAI:
    def __init__(self):
        # Download required NLTK data
        try:
            nltk.data.find('corpora/wordnet')
        except LookupError:
            print("Downloading WordNet...")
            nltk.download('wordnet', quiet=True)
        
        # Load word embeddings - this might take some time on first run
        try:
            print("Loading word embeddings...")
            self.word_vectors = gensim.downloader.load('glove-wiki-gigaword-100')
            print("Word embeddings loaded!")
        except Exception as e:
            print(f"Could not load word embeddings: {e}")
            print("Falling back to simpler methods.")
            self.word_vectors = None
        
        # Dictionary to store categories and their members
        self.categories = {}
        self.difficulty_levels = ["Easy", "Medium", "Hard", "Very Hard"]
        
    def get_semantic_similarity(self, word1, word2):
        """Calculate semantic similarity between two words using word vectors"""
        if self.word_vectors is None:
            return 0
            
        try:
            similarity = self.word_vectors.similarity(word1.lower(), word2.lower())
            return similarity
        except KeyError:
            return 0
    
    def get_wordnet_similarity(self, word1, word2):
        """Calculate semantic similarity using WordNet"""
        # Get synsets for both words
        synsets1 = wn.synsets(word1)
        synsets2 = wn.synsets(word2)
        
        if not synsets1 or not synsets2:
            return 0
            
        # Find the maximum similarity between any synset pair
        max_sim = 0
        for s1 in synsets1:
            for s2 in synsets2:
                try:
                    sim = s1.path_similarity(s2)
                    if sim and sim > max_sim:
                        max_sim = sim
                except:
                    continue
                    
        return max_sim
    
    def calculate_word_similarities(self, words):
        """Calculate pairwise similarities between all words"""
        n = len(words)
        similarity_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i+1, n):
                # Combine different similarity measures
                glove_sim = self.get_semantic_similarity(words[i], words[j])
                wordnet_sim = self.get_wordnet_similarity(words[i], words[j])
                
                # Weight and combine similarities
                similarity = max(glove_sim, wordnet_sim * 0.8)
                similarity_matrix[i, j] = similarity
                similarity_matrix[j, i] = similarity
                
        return similarity_matrix
    
    def identify_potential_groups(self, words, similarity_matrix, threshold=0.3):
        """Identify potential groups based on similarity"""
        n = len(words)
        potential_groups = []
        
        # Try all possible combinations of 4 words
        for i in range(n):
            for j in range(i+1, n):
                for k in range(j+1, n):
                    for l in range(k+1, n):
                        indices = [i, j, k, l]
                        group_words = [words[idx] for idx in indices]
                        
                        # Calculate average similarity within the group
                        similarities = []
                        for a in range(len(indices)):
                            for b in range(a+1, len(indices)):
                                similarities.append(similarity_matrix[indices[a], indices[b]])
                        
                        avg_similarity = sum(similarities) / len(similarities) if similarities else 0
                        
                        if avg_similarity > threshold:
                            potential_groups.append((group_words, avg_similarity))
        
        # Sort groups by similarity score
        potential_groups.sort(key=lambda x: x[1], reverse=True)
        return potential_groups
    
    def find_groups_with_common_features(self, words):
        """Find groups based on common features (prefix, suffix, etc.)"""
        potential_groups = []
        
        # Check for common prefixes
        prefixes = {}
        for word in words:
            for i in range(1, 4):  # Check prefixes of length 1-3
                if len(word) > i:
                    prefix = word[:i]
                    if prefix not in prefixes:
                        prefixes[prefix] = []
                    prefixes[prefix].append(word)
        
        # Check for common suffixes
        suffixes = {}
        for word in words:
            for i in range(1, 4):  # Check suffixes of length 1-3
                if len(word) > i:
                    suffix = word[-i:]
                    if suffix not in suffixes:
                        suffixes[suffix] = []
                    suffixes[suffix].append(word)
        
        # Length-based groups
        length_groups = {}
        for word in words:
            length = len(word)
            if length not in length_groups:
                length_groups[length] = []
            length_groups[length].append(word)
        
        # Character-based patterns
        has_special_char = []
        has_numbers = []
        all_uppercase = []
        
        for word in words:
            if any(not c.isalnum() for c in word):
                has_special_char.append(word)
            if any(c.isdigit() for c in word):
                has_numbers.append(word)
            if word.isupper() and len(word) > 1:
                all_uppercase.append(word)
        
        # Add prefix groups
        for prefix, prefix_words in prefixes.items():
            if len(prefix_words) >= 4:
                potential_groups.append((prefix_words[:4], 0.7))
        
        # Add suffix groups
        for suffix, suffix_words in suffixes.items():
            if len(suffix_words) >= 4:
                potential_groups.append((suffix_words[:4], 0.7))
        
        # Add length groups
        for length, length_words in length_groups.items():
            if len(length_words) >= 4:
                potential_groups.append((length_words[:4], 0.5))
        
        # Add special character/number groups
        if len(has_special_char) >= 4:
            potential_groups.append((has_special_char[:4], 0.6))
        if len(has_numbers) >= 4:
            potential_groups.append((has_numbers[:4], 0.6))
        if len(all_uppercase) >= 4:
            potential_groups.append((all_uppercase[:4], 0.6))
        
        return potential_groups
    
    def find_category_name(self, group):
        """Try to find a common category name for a group of words"""
        # Get all possible hypernyms
        hypernyms = {}
        
        for word in group:
            synsets = wn.synsets(word)
            for synset in synsets:
                for hypernym in synset.hypernyms():
                    name = hypernym.name().split('.')[0].replace('_', ' ')
                    if name not in hypernyms:
                        hypernyms[name] = 0
                    hypernyms[name] += 1
        
        # Sort by frequency
        sorted_hypernyms = sorted(hypernyms.items(), key=lambda x: x[1], reverse=True)
        
        if sorted_hypernyms:
            return sorted_hypernyms[0][0]
        else:
            return "Unknown Category"
    
    def solve(self, words):
        """
        Attempt to solve a Connections puzzle with the given 16 words.
        Returns the proposed 4 categories with 4 words each.
        """
        if len(words) != 16:
            print("Error: Please provide exactly 16 words")
            return None
        
        print(f"Analyzing {len(words)} words...")
        
        # Calculate similarities
        similarity_matrix = self.calculate_word_similarities(words)
        
        # Find potential groups based on semantic similarity
        semantic_groups = self.identify_potential_groups(words, similarity_matrix)
        
        # Find groups based on common features
        feature_groups = self.find_groups_with_common_features(words)
        
        # Combine and sort all potential groups
        all_groups = semantic_groups + feature_groups
        all_groups.sort(key=lambda x: x[1], reverse=True)
        
        print(f"Found {len(all_groups)} potential groups")
        
        # Use a greedy approach to select 4 non-overlapping groups
        selected_groups = []
        words_used = set()
        
        for group, score in all_groups:
            # Check if this group overlaps with already selected groups
            if not any(word in words_used for word in group) and len(group) == 4:
                selected_groups.append((group, score))
                words_used.update(group)
                
                # If we have 4 groups, we're done
                if len(selected_groups) == 4:
                    break
        
        # If we couldn't find 4 non-overlapping groups, we need to be more creative
        if len(selected_groups) < 4:
            print("Couldn't find 4 perfect groups, attempting to create remaining groups...")
            
            remaining_words = [word for word in words if word not in words_used]
            
            while len(selected_groups) < 4 and len(remaining_words) >= 4:
                # Just group the next 4 remaining words
                next_group = remaining_words[:4]
                selected_groups.append((next_group, 0.1))
                words_used.update(next_group)
                remaining_words = remaining_words[4:]
        
        # Assign difficulty levels and attempt to name categories
        result = {}
        for i, (group, score) in enumerate(selected_groups):
            difficulty = self.difficulty_levels[i % len(self.difficulty_levels)]
            category_name = self.find_category_name(group)
            
            result[f"Group {i+1} ({difficulty})"] = {
                "category": category_name,
                "words": group,
                "confidence": score
            }
        
        return result
    
    def interactive_solve(self):
        """Interactive mode for solving Connections puzzles"""
        print("\nWelcome to the Connections AI Solver")
        print("====================================")
        print("Enter the 16 words from the puzzle, separated by commas")
        
        input_words = input("Words: ")
        words = [word.strip() for word in input_words.split(",")]
        
        if len(words) != 16:
            print(f"Warning: You entered {len(words)} words instead of 16. Continuing anyway...")
        
        solution = self.solve(words)
        
        print("\nProposed Solution:")
        print("=================")
        
        for group_name, group_info in solution.items():
            print(f"\n{group_name}: {group_info['category']} (Confidence: {group_info['confidence']:.2f})")
            print(", ".join(group_info['words']))
        
        print("\nDo these groupings look correct? If not, try the following:")
        print("1. Be more specific with your input words")
        print("2. Make sure words are spelled correctly")
        print("3. The game often has clever or punny connections that AI might miss")
        
        return solution


# Example usage
if __name__ == "__main__":
    # Sample Connections puzzle
    sample_puzzle = [
        "BAT", "RACKET", "CLUB", "RECORD",
        "SAVE", "POST", "COPY", "SHARE", 
        "FIELD", "COURT", "PITCH", "TRACK",
        "CAST", "BROADCAST", "BEAM", "PROJECT"
    ]
    
    ai = ConnectionsAI()
    
    # Uncomment to solve a specific puzzle
    # solution = ai.solve(sample_puzzle)
    # for group_name, group_info in solution.items():
    #     print(f"\n{group_name}: {group_info['category']}")
    #     print(", ".join(group_info['words']))
    
    # Interactive mode
    ai.interactive_solve()
