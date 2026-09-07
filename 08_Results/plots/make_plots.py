import pandas as pd
import matplotlib.pyplot as plt


# ==============================
# Plot 1: Acceptance vs pT
# ==============================

pt = pd.read_csv(
    "../tables/conversion_acceptance_by_pT.csv"
)

plt.figure(figsize=(7,5))

plt.bar(
    pt["pT_bin"].astype(str),
    pt["acceptance"]*100
)

plt.xlabel("Photon pT bin")
plt.ylabel("Coplanarity acceptance (%)")
plt.title("Photon Conversion Background Acceptance vs pT")

plt.ylim(0,100)

plt.tight_layout()

plt.savefig(
    "acceptance_vs_pT.png",
    dpi=300
)

plt.close()


# ==============================
# Plot 2: Accepted vs Rejected
# ==============================

results = pd.read_csv(
    "../tables/conversion_coplanarity_results_10000.csv"
)

accepted = results["found"].sum()
rejected = len(results)-accepted


plt.figure(figsize=(6,5))

plt.bar(
    ["Accepted","Rejected"],
    [accepted,rejected]
)

plt.ylabel("Number of conversion events")
plt.title("Coplanarity Classification Result")


plt.tight_layout()

plt.savefig(
    "coplanarity_acceptance.png",
    dpi=300
)

plt.close()


print("Plots saved successfully")