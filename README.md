# BS30026 Binder Design

Goal of the project is computational design of a **Rapamycin-dependent de novo Heterodimer** using RFdiffusion3, LigandMPNN, and AlphaFold3.

## Project Goal

By utilizing natural heterodimer system of **FKBP12 - Rapamycin - FRB** as a model,
project will design new shorter & efficient target system of **Binder A - Rapamycin - Binder B**

## Design Strategy

### Stage 1 (Binder A)
Binder A is designed against the WT FRB–rapamycin complex.

WT FRB + Rapamycin
    -> RFD3 binder backbone design
    -> Geometry filtering
    -> LigandMPNN sequence design
    -> AF3 at +RAP & -RAP
    -> Geometry filtering
    -> Binder A candidate selection

Binder A replaces natural FKBP component.

### Stage 2 (Binder B)

Selected Binder A candidates will subsequently be used as targets for Binder B design.

Binder A + Rapamycin
    -> RFD3 binder backbone design
    -> Geometry filtering
    -> LigandMPNN sequence design
    -> AF3 at +RAP & -RAP
    -> Geometry filtering
    -> Binder B candidate selection

Binder A replaces natural FRB component.

## Reference Structure

PDB **1FAP**
- FKBP12: chain A, residues 1–107
- Rapamycin: originally chain A, residue 108
- WT FRB: chain B, residues 2018–2112

The **Processed RFD3 target**
- Chain A: generated Binder A
- Chain B: WT FRB
- Chain C: rapamycin

## Repository Structure

```text
.
├── inputs/
│   ├── 1FAP.pdb
│   ├── 1FAP_FRB_RAP.pdb
│   ├── frb_rap_binder_raphotspot.json
│   ├── af3_binderA_pilot/
│   └── af3_binderA_noRAP/
├── scripts/
│   ├── pdb_1FAP_edit.py
│   ├── analyze_binderA_*.py
│   ├── select_binderA_pilot.py
│   ├── make_af3_binderA_pilot_inputs.py
│   ├── analyze_af3_binderA_pilot.py
│   └── make_af3_*overlay.py
├── jobs/
│   ├── rfd3_raphotspot_h200_pilot.slurm
│   ├── ligandmpnn_pilot_22x4.slurm
│   ├── af3_binderA_pilot_88.slurm
│   └── af3_binderA_noRAP_88.slurm
└── README.md
```