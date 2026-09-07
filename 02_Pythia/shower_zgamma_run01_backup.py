import pythia8
import pandas as pd

LHE = "/mnt/c/Users/HP/Downloads/MG5_aMC_v3_7_2/Zgamma_test/Events/run_01/unweighted_events.lhe"

pythia = pythia8.Pythia()

pythia.readString("Beams:frameType = 4")
pythia.readString(f"Beams:LHEF = {LHE}")

# shower + hadronization
pythia.readString("PartonLevel:ISR = on")
pythia.readString("PartonLevel:FSR = on")
pythia.readString("HadronLevel:all = on")

pythia.init()

particles = []
nev = 0

for i in range(1000):

    if not pythia.next():
        continue

    nev += 1

    for p in pythia.event:
        if p.isFinal():particles.append({
    "event": nev,
    "id": p.id(),

    "status": p.status(),

    "mother1": p.mother1(),
    "mother2": p.mother2(),

    "px": p.px(),
    "py": p.py(),
    "pz": p.pz(),

    "E": p.e(),
    "pT": p.pT(),

    "eta": p.eta(),
    "phi": p.phi()
})



print("Events showered:", nev)
print("Final particles:", len(particles))

df = pd.DataFrame(particles)
df.to_csv("showered_zgammajet.csv", index=False)

print("Saved showered_zgammajet.csv")
