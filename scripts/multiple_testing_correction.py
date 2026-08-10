"""Applique Bonferroni et FDR (Benjamini-Hochberg) aux tests secondaires
deja obtenus, regroupes par famille de tests, pour verifier lesquels
resistent a la correction pour comparaisons multiples."""
from statsmodels.stats.multitest import multipletests

# Resultats deja obtenus, regroupes par famille de tests
families = {
    "Kruskal-Wallis complexite (12 tests)": [
        ("RTD global", 0.8568),
        ("RTD hierarchical", 0.2833),
        ("RTD decentralized", 0.5135),
        ("RTD shared_pool", 0.0150),
        ("Latence global", 0.0000),
        ("Latence hierarchical", 0.0000),
        ("Latence decentralized", 0.0000),
        ("Latence shared_pool", 0.0000),
        ("Reasoning global", 0.0000),
        ("Reasoning hierarchical", 0.0010),
        ("Reasoning decentralized", 0.0000),
        ("Reasoning shared_pool", 0.0000),
    ],
    "Mann-Whitney within-condition (9 tests)": [
        ("RTD hierarchical", 0.7109),
        ("RTD decentralized", 0.6784),
        ("RTD shared_pool", 0.8209),
        ("Latence hierarchical", 0.2872),
        ("Latence decentralized", 0.0689),
        ("Latence shared_pool", 0.1761),
        ("Reasoning hierarchical", 0.5318),
        ("Reasoning decentralized", 0.0430),
        ("Reasoning shared_pool", 0.6613),
    ],
    "Traceurs chi2/Fisher (3 tests)": [
        ("Chi2 categorie x condition", 0.0000),
        ("Chi2 categorie x schema", 0.0316),
        ("Fisher shared_pool+typed racine", 0.0356),
    ],
}

for family_name, tests in families.items():
    print("=" * 70)
    print(family_name)
    print("=" * 70)
    names = [t[0] for t in tests]
    pvals = [t[1] for t in tests]

    bonf_reject, bonf_pvals, _, _ = multipletests(pvals, alpha=0.05, method="bonferroni")
    fdr_reject, fdr_pvals, _, _ = multipletests(pvals, alpha=0.05, method="fdr_bh")

    print(f"{'Test':<35} {'p brut':<10} {'Bonferroni':<12} {'FDR (BH)':<12}")
    for name, p, bp, br, fp, fr in zip(names, pvals, bonf_pvals, bonf_reject, fdr_pvals, fdr_reject):
        bonf_status = "SIG" if br else "n.s."
        fdr_status = "SIG" if fr else "n.s."
        print(f"{name:<35} {p:<10.4f} {bp:<10.4f}({bonf_status}) {fp:<10.4f}({fdr_status})")
    print()