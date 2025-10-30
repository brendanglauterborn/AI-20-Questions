# Brendan Lauterborn
# Lab 3 20 questions
# sources: chatgpt

import os
import math
import pandas as pd
import numpy as np

DEFAULT_DATASET = "games_dataset.xlsx"
MAX_QUESTIONS = 20

def load_dataset(path: str):
    """Load the Excel dataset and handle errors gracefully."""
    try:
        df = pd.read_excel(path)
        df.columns = [c.strip() for c in df.columns]
        return df
    except FileNotFoundError:
        print(f"[!] Error: Dataset not found at '{path}'. Make sure it's in the same directory.")
    except ValueError as e:
        print(f"[!] Error reading Excel file: {e}")
    except Exception as e:
        print(f"[!] Unexpected error while loading dataset: {e}")
    return None


def calculate_entropy(values):
    """Calculate entropy of a set of values (discretized into bins)."""
    if len(values) == 0:
        return 0.0
    # Discretize continuous values into bins
    bins = np.linspace(-1, 1, 11)  # 10 bins from -1 to 1
    hist, _ = np.histogram(values, bins=bins)
    hist = hist[hist > 0]  # remove empty bins
    probs = hist / hist.sum()
    return -np.sum(probs * np.log2(probs + 1e-10))


def calculate_information_gain(df, candidates_idx, feat, threshold=0.0):
    """Calculate expected information gain from asking about a feature.
    
    Uses variance-based approach to measure how well a feature splits candidates.
    Higher variance = more informative question.
    """
    try:
        vals = df.loc[candidates_idx, feat].astype(float)
    except Exception:
        return 0.0
    
    vals = vals[~vals.isna()]
    if len(vals) <= 1:
        return 0.0
    
    # Use variance as a simpler, more effective measure of information
    # High variance = feature values are spread out = more discriminative
    variance = vals.var()
    
    # Also consider how balanced the split would be
    # A balanced split is better for narrowing down candidates
    left = vals[vals <= threshold]
    right = vals[vals > threshold]
    
    if len(left) == 0 or len(right) == 0:
        return variance * 0.5  # Unbalanced split gets penalty
    
    # Balance score: closer to 0.5 is better (perfect balance)
    balance = min(len(left), len(right)) / len(vals)
    
    # Combine variance (discrimination power) with balance
    # Balance is important but variance is more important
    return variance * (0.7 + 0.3 * balance)


def choose_next_question(df, candidates_idx, unused_feats):
    """Pick the feature with the highest information gain among remaining candidates."""
    best_feat, best_gain = None, -1.0
    for feat in unused_feats:
        gain = calculate_information_gain(df, candidates_idx, feat)
        if gain > best_gain:
            best_gain, best_feat = gain, feat
    return best_feat


def score_candidates(df, candidates_idx, asked):
    """Return similarity scores using improved weighted similarity with smart penalties."""
    if not asked or len(candidates_idx) == 0:
        return np.zeros(len(candidates_idx))
    
    feats = [f for f, _ in asked]
    answers = np.array([a for _, a in asked], dtype=float)
    X = (
        df.loc[candidates_idx, feats]
        .apply(pd.to_numeric, errors="coerce")
        .fillna(0.0)
        .to_numpy()
    )
    
    # Start with perfect scores
    scores = np.ones(len(candidates_idx))
    
    # Calculate feature importance based on variance across all remaining candidates
    feature_variances = []
    for i, feat in enumerate(feats):
        var = df.loc[candidates_idx, feat].astype(float).fillna(0.0).var()
        feature_variances.append(var)
    
    # For each feature, apply penalties based on mismatch, answer confidence, and feature importance
    for i, (feat, answer) in enumerate(asked):
        answer_confidence = abs(answer)  # How confident was the user? (0=neutral, 1=very sure)
        feature_var = feature_variances[i]
        
        # Calculate absolute difference (more intuitive than squared)
        game_values = X[:, i]
        abs_diffs = np.abs(game_values - answer)
        
        # More forgiving sigma values - allows for small differences
        # Tolerance increases as confidence decreases
        if answer_confidence > 0.7:  # Strong answer
            base_sigma = 0.35  # Increased from 0.2 to tolerate small differences
        elif answer_confidence > 0.3:  # Medium answer
            base_sigma = 0.45  # Increased from 0.3
        else:  # Weak/neutral answer
            base_sigma = 0.6  # Increased from 0.5
        
        # Adjust sigma based on feature variance (distinctive features get slightly stricter)
        if feature_var > 0.4:  # Very distinctive
            sigma = base_sigma * 0.85  # Less aggressive reduction (was 0.7)
        elif feature_var > 0.2:  # Moderately distinctive
            sigma = base_sigma * 0.92  # Less aggressive reduction (was 0.85)
        else:  # Not very distinctive
            sigma = base_sigma
        
        # Calculate similarity using Gaussian-like penalty
        # Similarity is high when difference is small
        similarity = np.exp(-abs_diffs**2 / (2 * sigma**2))
        
        # Apply sign-aware penalty ONLY for true opposites (not for close values)
        # Example: -1.0 vs 0.8 should be penalized, but -1.0 vs -0.8 should not
        if answer_confidence > 0.6:
            # Only penalize if signs differ AND both values are strong (> 0.5 magnitude)
            opposite_signs = np.sign(answer) != np.sign(game_values)
            both_strong = (abs(answer) > 0.5) & (np.abs(game_values) > 0.5)
            true_opposites = opposite_signs & both_strong
            
            # Less harsh penalty (0.5 instead of 0.3)
            similarity[true_opposites] *= 0.5
        
        # Weight increases with both answer confidence and feature distinctiveness
        # More weight = more impact on final score
        confidence_weight = 0.3 + 0.7 * answer_confidence
        variance_weight = min(1.0, 0.5 + feature_var)
        combined_weight = confidence_weight * variance_weight
        
        # Apply weighted similarity (multiplicative to compound evidence)
        scores = scores * (similarity ** combined_weight)
    
    return scores


def ask_float(prompt: str) -> float:
    """Ask for a value in [-1, 1]. Enter = 0."""
    while True:
        s = input(prompt).strip()
        if s == "":
            return 0.0
        try:
            val = float(s)
            if -1 <= val <= 1:
                return val
            print("Enter a value between -1 and 1.")
        except ValueError:
            print("Enter a number between -1 and 1.")


def update_existing_game(df, dataset_path, game_idx, asked):
    """Update an existing game's features based on user's answers (learning from mistakes)."""
    game_name = df.iloc[game_idx, 0]
    learning_rate = 0.4  # How much to adjust based on new information
    
    print(f"\nUpdating knowledge for '{game_name}' based on your answers...")
    print("(Blending existing data with new information)\n")
    
    updated_count = 0
    for feat, user_ans in asked:
        if feat in df.columns:
            old_val = pd.to_numeric(df.iloc[game_idx][feat], errors='coerce')
            if pd.isna(old_val):
                old_val = 0.0
            
            # Blend old value with new user answer
            new_val = old_val * (1 - learning_rate) + user_ans * learning_rate
            df.iloc[game_idx, df.columns.get_loc(feat)] = new_val
            
            print(f"  {feat}: {old_val:.2f} -> {new_val:.2f}")
            updated_count += 1
    
    if updated_count > 0:
        print(f"\nAttempting to save changes to '{dataset_path}'...")
        try:
            # Make a copy of the dataframe to ensure we're saving the updated version
            df_copy = df.copy()
            df_copy.to_excel(dataset_path, index=False, engine='openpyxl')
            print(f"Successfully saved! Updated {updated_count} feature(s) for '{game_name}'.")
            print("I'll remember this for next time!")
            
            # Verify the save by reading it back
            try:
                test_df = pd.read_excel(dataset_path)
                test_val = pd.to_numeric(test_df.iloc[game_idx][asked[0][0]], errors='coerce')
                expected_val = df.iloc[game_idx, df.columns.get_loc(asked[0][0])]
                if abs(test_val - expected_val) < 0.01:
                    print("Verified: Changes were written to disk successfully!")
                else:
                    print("Warning: File saved but verification shows different values.")
            except:
                pass
            
            return df, True
        except PermissionError:
            print(f"\n[!] ERROR: Permission denied. Close the Excel file if it's open and try again.")
            return df, False
        except Exception as e:
            print(f"\n[!] ERROR: Could not save updated dataset: {e}")
            print(f"    Error type: {type(e).__name__}")
            return df, False
    else:
        print("No features were updated.")
        return df, False


def add_new_game_to_database(df, dataset_path, asked=None):
    """Add a new game with all feature values filled in."""
    print("\n" + "="*50)
    print("ADDING NEW GAME TO DATABASE")
    print("="*50)
    
    game_name = input("\nEnter the game name: ").strip()
    if not game_name:
        print("No game name provided. Skipping.")
        return df, False
    
    # Check if game already exists
    existing_names = df.iloc[:, 0].astype(str).str.lower()
    if game_name.lower() in existing_names.values:
        print(f"'{game_name}' already exists in the dataset!")
        return df, False
    
    new_row = {df.columns[0]: game_name}
    
    # Create a dict from asked questions for easy lookup
    asked_dict = {feat: ans for feat, ans in asked} if asked else {}
    
    # Automatically use answers from game session for those features
    if asked_dict:
        print(f"\nUsing your {len(asked_dict)} answer(s) from the game...")
        for feat, ans in asked_dict.items():
            new_row[feat] = ans
            print(f"  {feat}: {ans:.2f}")
    
    # Get list of features we still need to ask about
    remaining_feats = [f for f in df.columns[1:] if f not in asked_dict]
    
    if remaining_feats:
        print(f"\nPlease provide values for the remaining {len(remaining_feats)} feature(s):")
        print("(Enter values between -1 and 1, or press Enter for 0)")
        for feat in remaining_feats:
            val = ask_float(f"  {feat}: ")
            new_row[feat] = val
    else:
        print("\nAll features have been filled from your game answers!")
    
    # Add the new game
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    print(f"\nAttempting to save '{game_name}' to '{dataset_path}'...")
    try:
        df.to_excel(dataset_path, index=False, engine='openpyxl')
        print(f"✓ Successfully added '{game_name}' to the database!")
        print(f"Dataset now contains {len(df)} games.")
        return df, True
    except PermissionError:
        print(f"\n[!] ERROR: Permission denied. Close the Excel file if it's open and try again.")
        return df, False
    except Exception as e:
        print(f"\n[!] ERROR: Could not save updated dataset: {e}")
        print(f"    Error type: {type(e).__name__}")
        return df, False


def add_new_question_to_database(df, dataset_path):
    """Add a new question/feature column and get values for all games."""
    print("\n" + "="*50)
    print("ADDING NEW QUESTION TO DATABASE")
    print("="*50)
    
    question_name = input("\nEnter the new question/feature name: ").strip()
    if not question_name:
        print("No question name provided. Skipping.")
        return df
    
    # Check if question already exists
    if question_name in df.columns:
        print(f"'{question_name}' already exists in the dataset!")
        return df
    
    print(f"\nAdding new feature: '{question_name}'")
    print("Please provide values for this feature for EACH game:")
    print("(Enter values between -1 and 1, or press Enter for 0)\n")
    
    # Create new column with default values
    new_values = []
    
    for idx, row in df.iterrows():
        game_name = row.iloc[0]
        val = ask_float(f"  {game_name}: ")
        new_values.append(val)
    
    # Add the new column
    df[question_name] = new_values
    
    print(f"\nAttempting to save new question '{question_name}' to '{dataset_path}'...")
    try:
        df.to_excel(dataset_path, index=False, engine='openpyxl')
        print(f"Successfully added new question '{question_name}' to the database!")
        print(f"Dataset now has {len(df.columns) - 1} features.")
        return df
    except PermissionError:
        print(f"\n[!] ERROR: Permission denied. Close the Excel file if it's open and try again.")
        return df
    except Exception as e:
        print(f"\n[!] ERROR: Could not save updated dataset: {e}")
        print(f"    Error type: {type(e).__name__}")
        return df


def show_final_guesses(df, candidates_idx, sims, asked_count):
    """Show top 3 guesses with % certainty using improved confidence calculation."""
    order = np.argsort(-sims)
    top_scores = sims[order]
    
    if len(top_scores) == 0:
        print("\nNo candidates to guess.")
        return None
    
    if np.max(top_scores) == 0:
        # All scores are zero, use uniform distribution
        probs = np.ones_like(top_scores) / len(top_scores)
    else:
        # Use adaptive temperature based on score spread
        score_range = np.max(top_scores) - np.min(top_scores)
        score_std = np.std(top_scores)
        
        # If scores are well-separated, use lower temperature (more confident)
        # If scores are similar, use higher temperature (less confident)
        if score_std > 0.15 and score_range > 0.3:
            temperature = 0.2  # Very confident
        elif score_std > 0.1 and score_range > 0.2:
            temperature = 0.4  # More confident
        elif score_std > 0.05:
            temperature = 0.6  # Moderately confident
        else:
            temperature = 0.8  # Less confident (scores are similar)
        
        # Normalize scores to [0, 1] range for better softmax behavior
        if score_range > 0:
            normalized_scores = (top_scores - np.min(top_scores)) / score_range
        else:
            normalized_scores = top_scores
        
        # Apply softmax with adaptive temperature
        exp_scores = np.exp(normalized_scores / temperature)
        probs = exp_scores / np.sum(exp_scores)
        
        # Boost confidence if top score is significantly higher than second
        if len(probs) > 1:
            score_gap = top_scores[0] - top_scores[1]
            if score_gap > 0.2:  # Lower threshold for boosting
                # Redistribute some probability from others to top
                boost = min(0.25, score_gap * 0.4)  # More aggressive boosting
                probs[1:] *= (1 - boost)
                probs[0] += boost * (len(probs) - 1) * probs[0]
                # Renormalize
                probs = probs / np.sum(probs)

    print(f"\nFinal guesses after {asked_count} question(s):")
    
    # Show up to 5 candidates, or all if fewer than 5
    num_to_show = min(5, len(order))
    for i in range(num_to_show):
        game = df.iloc[candidates_idx[order[i]], 0]
        certainty = probs[i] * 100
        print(f"  {i+1}. {game:<30} — {certainty:.1f}% certainty")

    guess = df.iloc[candidates_idx[order[0]], 0]
    print(f"\n My top guess: **{guess}** ({probs[0]*100:.1f}% confident)")
    return guess


def play_game(df, dataset_path):
    """Play a single round of 20 questions."""
    feats = list(df.columns[1:])
    
    candidates_idx = np.arange(len(df))
    asked = []
    used = set()

    for qnum in range(1, MAX_QUESTIONS + 1):
        unused_feats = [f for f in feats if f not in used]
        if not unused_feats:
            print("\nNo more features to ask about.")
            break

        feat = choose_next_question(df, candidates_idx, unused_feats)
        if feat is None:
            print("\nNo more informative questions left.")
            break

        print(f"\nQ{qnum:02d}: {feat}?")
        ans = ask_float("   (-1 to 1, Enter=0): ")
        asked.append((feat, ans))
        used.add(feat)

        sims = score_candidates(df, candidates_idx, asked)

        # Improved adaptive pruning strategy
        order = np.argsort(-sims)
        sorted_sims = sims[order]
        
        # Special case: If 2 candidates remain and they're clearly distinguished, guess now
        if len(candidates_idx) == 2 and len(sorted_sims) == 2:
            score_ratio = sorted_sims[0] / (sorted_sims[1] + 1e-10)
            # If top score is significantly better (3x or more), we have a clear winner
            if score_ratio >= 3.0:
                candidates_idx = candidates_idx[order[:1]]
                print(f"   → Clear winner identified!")
                break
        
        # Calculate adaptive keep ratio based on question progress
        progress = qnum / MAX_QUESTIONS
        
        # Start more conservative (keep 95%), become more aggressive later (keep 50%)
        base_keep_ratio = 0.95 - 0.45 * progress  # 0.95 -> 0.50
        
        # Use score-based threshold that adapts to score distribution
        if sorted_sims[0] > 0 and len(sorted_sims) > 1:
            # Keep candidates within reasonable range of top score
            # Threshold gets stricter as we ask more questions
            threshold_ratio = 0.7 - 0.3 * progress  # 0.7 -> 0.4
            threshold = sorted_sims[0] * threshold_ratio
            threshold_keep = np.sum(sorted_sims >= threshold)
            
            # Also consider score gaps - if there's a big drop, cut there
            if len(sorted_sims) > 2:
                score_diffs = sorted_sims[:-1] - sorted_sims[1:]
                if len(score_diffs) > 0 and np.max(score_diffs) > 0:
                    # Find the largest gap in scores
                    largest_gap_idx = np.argmax(score_diffs) + 1
                    # Only use gap-based cutoff if it's reasonable
                    if largest_gap_idx >= 2 and score_diffs[largest_gap_idx-1] > 0.1:
                        gap_keep = largest_gap_idx
                    else:
                        gap_keep = len(order)
                else:
                    gap_keep = len(order)
            else:
                gap_keep = len(order)
        else:
            threshold_keep = len(order)
            gap_keep = len(order)
        
        # Combine strategies: use minimum of ratio-based, threshold-based, and gap-based
        # But ensure we keep at least a minimum number
        ratio_keep = int(np.ceil(base_keep_ratio * len(order)))
        keep = max(2, min(ratio_keep, threshold_keep, gap_keep))
        keep = min(keep, len(order))  # Can't keep more than we have
        
        candidates_idx = candidates_idx[order[:keep]]
        
        print(f"   → {len(candidates_idx)} candidate(s) remaining")
        
        # Show top candidates if there are few remaining
        if len(candidates_idx) <= 5 and qnum > 3:
            print(f"   Top candidates: {', '.join([df.iloc[i, 0] for i in candidates_idx[:3]])}")

        # If only 1 candidate remains, we found it!
        if len(candidates_idx) == 1:
            print(f"\nOnly 1 game remaining. I've got it!")
            # Still show the final guesses with percentages
            break

    # Final guesses with certainty
    sims = score_candidates(df, candidates_idx, asked)
    
    if len(sims) > 0:
        guess = show_final_guesses(df, candidates_idx, sims, len(asked))
    else:
        print("\nNo candidates remain to guess.")
        guess = None
    
    return df, guess, asked


def main():
    print("=== 20 Questions — Video Game Guesser ===")
    dataset_path = DEFAULT_DATASET

    df = load_dataset(dataset_path)
    if df is None:
        print("Please check that the Excel file exists in this folder.")
        return

    if df.shape[1] < 2:
        print("[!] Dataset must have at least 2 columns (name + one feature).")
        return

    feats = list(df.columns[1:])
    print(f"Loaded {len(df)} games with {len(feats)} features.\n")

    while True:
        # Play one round
        df, guess, asked = play_game(df, dataset_path)
        
        # Ask if we were correct
        if guess:
            correct = input(f"\nWas I correct? (y/n): ").lower().startswith('y')
        else:
            correct = False
        
        if correct:
            print("I guessed correctly!")
            play_again = input("\nWould you like to play again? (y/n): ").lower().startswith('y')
            if play_again:
                print("\n" + "="*50)
                print("Starting a new game!")
                print("="*50)
                continue
            else:
                print("\nThanks for playing! Goodbye!")
                break
        else:
            # We were wrong - learn from the mistake
            print("\nI was wrong!")
            
            # Ask for the correct game name
            correct_game = input("What was the correct game? ").strip()
            if not correct_game:
                print("No game provided. Exiting.")
                break
            
            # Check if the correct game exists in the database
            existing_names = df.iloc[:, 0].astype(str)
            matching_indices = existing_names[existing_names.str.lower() == correct_game.lower()].index
            
            if len(matching_indices) > 0:
                # Game exists - update its features (learn from mistake)
                game_idx = matching_indices[0]
                print(f"\n'{df.iloc[game_idx, 0]}' is in the database.")
                print("I'll update my knowledge based on your answers...")
                df, updated = update_existing_game(df, dataset_path, game_idx, asked)
                
                if updated:
                    # Ask if they want to add a new question
                    add_question = input("\nWould you like to add a new question/feature? (y/n): ").lower().startswith('y')
                    if add_question:
                        df = add_new_question_to_database(df, dataset_path)
                        print("\nDatabase updated successfully!")
            else:
                # Game doesn't exist - offer to add it
                print(f"\n'{correct_game}' is not in the database.")
                add_game = input("Would you like to add it? (y/n): ").lower().startswith('y')
                
                if add_game:
                    df, added = add_new_game_to_database(df, dataset_path, asked)
                    
                    if added:
                        # Ask if they want to add a new question
                        add_question = input("\nWould you like to add a new question/feature? (y/n): ").lower().startswith('y')
                        
                        if add_question:
                            df = add_new_question_to_database(df, dataset_path)
                            print("\nDatabase updated successfully!")
                    
            print("\nThanks for playing! Goodbye!")
            break


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted. Goodbye.")
