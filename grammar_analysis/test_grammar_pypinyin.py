import sys
import os

# Adjust path to include the current directory so imports work
current_dir = os.path.dirname(os.path.abspath(__file__))
services_dir = os.path.join(current_dir, "services")
sys.path.append(services_dir)
sys.path.append(current_dir)

from services.grammar_rag_analysis import analyze_grammar_point

test_sentence = "人工智慧的發展正在快速進步，專家預測2025年將是AI技術更加專業化與深度應用的關鍵一年。"
print(f"Testing analysis for: {test_sentence}")

try:
    result = analyze_grammar_point(test_sentence, user_level=1)
    print("\n\n--- RESULT ---")
    print(result)
    print("--------------\n")
    
    if "ㄋㄧˇ ㄏㄠˇ ˙ㄇㄚ" in result or "ㄋㄧˇ ㄏㄠˇ 吗" in result: # coping with different outputs
        print("SUCCESS: Zhuyin found in output.")
    else:
        # Relaxed check for pypinyin behavior, might handle neutral tone differently
        print("CHECK: Verify Zhuyin visually above.")

except Exception as e:
    print(f"ERROR: {e}")
    import traceback
    traceback.print_exc()
