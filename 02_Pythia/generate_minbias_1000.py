import pythia8
import pandas as pd

N_EVENTS = 1000
OUT = "02_Pythia/minbias_1000.csv"

pythia = pythia8.Pythia()

pythia.readString("Beams:idA = 2212")
pythia.readString("Beams:idB = 2212")
pythia.readString("Beams:eCM = 13000.")
pythia.readString("SoftQCD:inelastic = on")

pythia.init()

rows = []
nev = 0

for i in range(N_EVENTS):

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

            "xProd": p.xProd(),
            "yProd": p.yProd(),
            "zProd": p.zProd(),
            "tProd": p.tProd(),
        })

    if nev % 100 == 0:
        print(f"Generated {nev}/{N_EVENTS}")

df = pd.DataFrame(rows)

df.to_csv(OUT, index=False)

print("\nSaved:", OUT)
print("Events:", nev)
print("Charged particles:", len(df))
print("Mean charged/event:", len(df) / nev)
print("|eta| < 1.8:", (df["eta"].abs() < 1.8).sum())
