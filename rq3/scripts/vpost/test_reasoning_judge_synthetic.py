"""
Test synthetique du judge V_post - AUCUNE dependance a RQ1. Utilise deux
traces de raisonnement fabriquees a la main (style GSM8K, coherent avec le
domaine RQ1) pour verifier, avant l'arrivee des vraies donnees:
  1. Le mecanisme API fonctionne reellement (parametre thinking, format,
     budget de tokens) - jamais teste en conditions reelles jusqu'ici.
  2. Le jugement est sensible sur des cas evidents (une trace propre vs
     une trace avec une contradiction flagrante).

Cout: 4 appels API (2 traces x 2 conditions thinking on/off) - de l'ordre
de quelques centimes, pas de quoi s'inquieter mais un vrai cout.

Usage:
    python test_reasoning_judge_synthetic.py
"""

from reasoning_judge import evaluate_reasoning_trace

# Trace 1: propre, coherente de bout en bout, reponse correcte.
CLEAN_TRACE = """Step 1: The problem states Sarah has 24 apples and gives away 9.
Step 2: 24 - 9 = 15 apples remaining.
Step 3: Sarah then buys 3 more bags of apples, each containing 6 apples.
Step 4: 3 * 6 = 18 new apples.
Step 5: Total apples now: 15 + 18 = 33.
Final answer: Sarah has 33 apples."""
CLEAN_ANSWER = "33"

# Trace 2: contradiction flagrante - le nombre de pommes restantes change
# sans justification entre l'etape 2 et l'etape 4.
CONTRADICTORY_TRACE = """Step 1: The problem states Sarah has 24 apples and gives away 9.
Step 2: 24 - 9 = 15 apples remaining.
Step 3: Sarah then buys 3 more bags of apples, each containing 6 apples.
Step 4: 3 * 6 = 18 new apples. Adding to her 20 remaining apples gives 38.
Final answer: Sarah has 38 apples."""
CONTRADICTORY_ANSWER = "38"


def run_case(label: str, trace: str, answer: str, use_thinking: bool):
    print(f"\n--- {label} | thinking={'on' if use_thinking else 'off'} ---")
    result = evaluate_reasoning_trace(trace, answer, use_thinking=use_thinking)
    print(f"triggered: {result['triggered']}")
    print(f"rationale: {result['judge_rationale']}")
    if "JUDGE_ERROR" in result["judge_rationale"]:
        print("  ATTENTION: echec - le mecanisme API a un probleme a corriger avant d'attendre RQ1.")
    return result


def main():
    print("Test synthetique V_post - verification du mecanisme, pas des vraies donnees.\n")

    r1 = run_case("Trace propre (attendu: triggered=False)", CLEAN_TRACE, CLEAN_ANSWER, use_thinking=False)
    r2 = run_case("Trace propre (attendu: triggered=False)", CLEAN_TRACE, CLEAN_ANSWER, use_thinking=True)
    r3 = run_case("Trace contradictoire (attendu: triggered=True)", CONTRADICTORY_TRACE, CONTRADICTORY_ANSWER, use_thinking=False)
    r4 = run_case("Trace contradictoire (attendu: triggered=True)", CONTRADICTORY_TRACE, CONTRADICTORY_ANSWER, use_thinking=True)

    print("\n=== Resume ===")
    correct = 0
    total = 0
    for label, result, expected in [
        ("clean/thinking_off", r1, False),
        ("clean/thinking_on", r2, False),
        ("contradictory/thinking_off", r3, True),
        ("contradictory/thinking_on", r4, True),
    ]:
        if "JUDGE_ERROR" in result["judge_rationale"]:
            print(f"  {label}: ERREUR MECANIQUE - {result['judge_rationale']}")
            continue
        total += 1
        ok = result["triggered"] == expected
        correct += ok
        print(f"  {label}: {'OK' if ok else 'INATTENDU'} (triggered={result['triggered']}, attendu={expected})")

    if total:
        print(f"\n{correct}/{total} conformes a l'attendu sur ce test minimal.")
    print("\nRappel: ceci ne remplace pas le test A/B avec verite terrain reelle (run_vpost_ab_test.py) - "
          "c'est juste une verification que le mecanisme fonctionne avant d'attendre RQ1.")


if __name__ == "__main__":
    main()