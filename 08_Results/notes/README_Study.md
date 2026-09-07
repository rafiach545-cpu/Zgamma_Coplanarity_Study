\# Zgamma Coplanarity Study



\## Project Overview



This project investigates the performance of a coplanarity-based track classification method for studying quirk-like signal topologies and photon conversion backgrounds.



The main goal is to test whether photon conversion events:



γ → e⁺ + e⁻



can produce track configurations that pass the coplanarity selection criteria.



The complete analysis chain includes:



\- MadGraph event generation

\- Pythia showering

\- Photon selection

\- Photon conversion simulation

\- Curved charged-particle propagation

\- Detector hit generation

\- Coplanarity classification





\---



\# Analysis Pipeline



\## 1. MadGraph Event Generation



\### Purpose



Generate realistic high-energy photon background events.



Process:



pp → Zγ + jet



Software:



MadGraph5\_aMC





Output:



01\_MadGraph/



\- run\_card.dat

\- unweighted\_events.lhe

\- run\_01\_tag\_1\_banner.txt





\---



\## 2. Pythia Showering



\### Purpose



Convert parton-level events into realistic final-state particles.



Included processes:



\- Initial State Radiation (ISR)

\- Final State Radiation (FSR)

\- Hadronization





Output:



02\_Pythia/



\- shower\_zgamma.py

\- showered\_zgammajet.csv





\---



\## 3. Photon Selection



\### Selection Criteria



Photon candidates are selected using:



\- Particle ID = 22

\- Status = 62

\- pT > 0.5 GeV

\- |η| < 2.5





Leading photon selection is performed event-by-event.



Output:



03\_Photon\_Selection/



\- leading\_isolated\_photons\_10000.csv

\---



\# 4. Photon Conversion Simulation



\## Purpose



Study the effect of photon conversion background on the coplanarity selection.



Photon conversion process:



γ → e⁺ + e⁻





\## Conversion Model



Parameters used:



\- Conversion probability = 1%

\- Conversion radius = 25 mm

\- Electron momentum = half of photon momentum

\- Positron momentum = half of photon momentum





Output:



04\_Conversion/



\- converted\_pairs.csv

\- converted\_pairs\_10000.csv





\---



\# 5. Curved Track Propagation



\## Purpose



Propagate charged particles through the detector magnetic field and generate realistic detector hits.





Features:



\- Charged particle helical motion

\- Magnetic field curvature

\- Detector module intersections

\- Real detector geometry

\- Hit generation





Output:



05\_Tracking/



\- charged\_trajectory.py

\- make\_curved\_conversion\_hits.py

\- converted\_photon\_hits\_curved.csv





\---



\# 6. Coplanarity Analysis



\## Purpose



Evaluate whether reconstructed tracks satisfy the expected planar topology.



The coplanarity algorithm calculates:



\- Plane consistency

\- Spatial residuals

\- Track compatibility with a common plane





Output:



06\_Coplanarity/



\- conversion\_coplanarity\_results.csv

\- conversion\_coplanarity\_results\_10000.csv

\- conversion\_coplanarity\_curved\_results.csv

\- conversion\_coplanarity\_relaxedcuts\_results.csv

\- conversion\_acceptance\_by\_pT.csv





\---



\# Current Results



\## Photon Conversion Background Study



MadGraph sample:



10000 generated events





Photon selection:



9560 leading photons





Conversion simulation:



94 photon conversions





Generated charged particles:



188 tracks





\---



\# Coplanarity Acceptance



Overall result:



Accepted events:



64 / 94





Acceptance:



≈ 68%





This shows that photon conversion backgrounds can pass the coplanarity selection and may mimic signal-like topology.





\---



\# pT Dependence



\## Photon pT: 5–20 GeV



Total events:



43





Accepted events:



27





Acceptance:



62.8 ± 7.4 %





\---



\## Photon pT: >20 GeV



Total events:



51





Accepted events:



37





Acceptance:



72.5 ± 6.2 %





\---



\# Physics Interpretation



The results indicate that photon conversion backgrounds are not completely rejected by the coplanarity requirement.



A moderate increase in acceptance is observed at higher photon transverse momentum, which is consistent with reduced track curvature effects.



However, the current statistics are not sufficient to make a strong claim about pT-dependent rejection performance.





\---



\# Current Limitations



\- Conversion background statistics are still limited.

\- More events are required for reliable pT-binned measurements.

\- Multiple quirk signal events are required to estimate signal efficiency.

\- Final background rejection performance requires larger samples.





\---



\# Future Work



\## Planned Studies



1\. Generate larger photon conversion samples.



2\. Produce detailed pT-dependent acceptance curves.



3\. Include statistical uncertainty bands.



4\. Generate multiple quirk benchmark events.



5\. Compare:



Quirk signal efficiency



vs



Photon conversion background acceptance





\---



\# Research Notes



This project keeps the analysis pipeline independent from the original TrackML repository.



Shared physics utilities are stored separately.



The workflow is organized to allow reproducible future studies and thesis/report preparation.

