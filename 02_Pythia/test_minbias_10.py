import pythia8
import pandas as pd

pythia = pythia8.Pythia()

pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 13000.")

pythia.readString("SoftQCD:inelastic = on")

pythia.init()

rows = []
nev = 0

for i in range(10):

    if not pythia.next():
        continue

    nev += 1

    for p in pythia.event:

        if not p.isFinal():
            continue

        if not p.isCharged():
            continue

        rows.append({
            "event": nev,
            "id": p.id(),
            "charge": p.charge(),

            "mass": p.m(),

            "px": p.px(),
            "py": p.py(),
            "pz": p.pz(),

            "E": p.e(),
            "pT": p.pT(),
            "eta": p.eta(),
            "phi": p.phi(),

            # Pythia production vertex
            # Pythia lengths are in mm
            "xProd": p.xProd(),
            "yProd": p.yProd(),
            "zProd": p.zProd(),
            "tProd": p.tProd(),
        })

print("Minimum-bias events generated:", nev)
print("Final charged particles:", len(rows))

df = pd.DataFrame(rows)

out = "02_Pythia/minbias_10_test.csv"
df.to_csv(out, index=False)

print("Saved:", out)

print("\nColumns:")
print(df.columns.tolist())

print("\nMass summary [GeV]:")
print(df["mass"].describe())

print("\nProduction vertex summary [mm]:")
print(df[["xProd","yProd","zProd"]].describe())

print("\nCharged particles per event:")
print(df.groupby("event").size())
