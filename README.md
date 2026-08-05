# HENON Mission Viability: CME Evolution Analysis from Distant Retrograde Orbit to L1

## One-Sentence Summary
A Python-based space-weather analysis pipeline that simulates the ESA HENON Distant Retrograde Orbit, builds a proxy CME observation dataset from archival space weather satellite data, and quantifies how CME magnetic-field, velocity, and storm-driving properties evolve between the HENON orbit and L1

## Overview
This repository investigates whether measurements taken from the proposed European Space Agency HENON mission orbit could be used to improve early warning of geomagnetic storms at Earth.

The project first simulates the planned HENON Distant Retrograde Orbit in the Sun–Earth system, then compares that orbit with archival spacecraft trajectories to identify periods where existing spacecraft passed through similar regions of space. These crossover periods are used to construct a proxy dataset representing what a HENON spacecraft might have observed in flight.

The proxy dataset is then cross-referenced with the Richardson & Cane ICME catalogue to isolate crossover events associated with known Coronal Mass Ejections (CMEs). For each event, the repository compares observations of the same CME at two locations: the proxy HENON orbit and the L1 Lagrange point, using both direct in-situ measurements and time-shifted L1 reference data.

A set of analysis modules then quantifies how key CME properties evolve during propagation, including absolute magnetic field, southward magnetic field, solar wind velocity, and predicted geomagnetic storm severity via the Burton Dst model. The overall aim is to test which CME parameters remain reliable over the propagation distance from the HENON orbit to Earth, and whether measurements from that orbit could support earlier and operationally useful space-weather forecasting

## Research Question / Motivation
The central motivation of this project was to assess the feasibility of the proposed ESA HENON mission as an upstream space-weather forecasting platform. Current operational monitoring of Earth-directed solar wind and CME conditions is typically performed near the L1 Lagrange point, which provides only short warning times before CME reaches Earth. By contrast, the planned HENON Distant Retrograde Orbit lies substantially farther upstream, offering the possibility of earlier observations and therefore longer warning times for potentially hazardous space-weather events.

The key scientific question was whether CME measurements made at the HENON orbit could be used directly to predict the properties of the same event when it later reached Earth. In practical terms, the project asked whether important CME parameters are sufficiently conserved during propagation from the HENON DRO to L1, or whether their evolution is large enough that additional physical modelling would be required for useful forecasting.

This led to two main research questions:

1. Are CME magnetic and plasma properties conserved during propagation from the HENON orbit to Earth?
2. Can geomagnetic storm severity at Earth be accurately predicted from measurements taken at the HENON DRO alone?

These questions matter because the value of a more distant upstream mission depends not just on earlier observation, but on whether those earlier measurements remain operationally interpretable by the time the CME reaches Earth. If the key parameters are well preserved, HENON-style measurements could potentially be used directly for forecasting. If not, then the mission would still be scientifically valuable, but would require additional correction or evolution modelling to translate upstream observations into accurate Earth-impact predictions.

In that sense, this project was not only an academic study of CME evolution, but also a practical mission-viability assessment, identifying which quantities remain forecast-relevant at larger upstream distances, and which would need additional modelling before the HENON mission could achieve its full operational forecasting potential

## Key Findings
The absolute magnetic-field structure of CMEs was largely preserved between the HENON orbit and L1.
Analysis of absolute magnetic field showed strong evidence that the large-scale temporal structure of the CME field remains conserved over the propagation distance relevant to HENON. Peak field amplitudes were also statistically consistent between the two observation points, supporting the idea that magnetic measurements from the HENON orbit could be used directly as forecast inputs.

The southward magnetic-field component was also broadly conserved.
Although the southward component is more sensitive to vector structure than absolute field magnitude, the analysis found no statistically significant evidence for systematic degradation between the HENON orbit and L1. This suggests that the most geoeffective magnetic-field component remains operationally useful at the HENON distance.

Solar wind velocity evolved significantly during propagation.
In contrast to the magnetic field, CME velocity measurements were not conserved between the HENON orbit and L1. The HENON-aligned spacecraft generally observed larger shock jumps and higher CME-related velocities than were later seen at L1, consistent with aerodynamic drag slowing the CME as it propagated sunward.

Geomagnetic storm severity was systematically under-predicted when using HENON-orbit measurements directly.
Applying the O’Brien & McPherron Dst model showed that Dst predicted from HENON-orbit measurements was consistently less severe than the equivalent prediction from L1 data. This demonstrated that direct use of HENON plasma measurements would not reproduce Earth-impact storm strength without correction.

The Dst prediction gap was driven primarily by velocity evolution, not magnetic-field evolution.
Decomposition analysis showed that replacing the magnetic-field input had little effect on the Dst discrepancy, whereas replacing the velocity input recovered much closer agreement with the L1 result. This identified velocity evolution as the dominant cause of the forecast gap.

The key physical mechanism was a temporal offset in the velocity profile relative to the magnetic-field structure.
Even when instantaneous velocity and southward magnetic field at minimum Dst appeared broadly similar, the integrated injection history still differed. Cross-correlation analysis showed that shifting the HENON velocity profile relative to the magnetic field was sufficient to recover agreement in predicted minimum Dst, indicating that the main issue was timing evolution of the shock/velocity structure rather than simple amplitude mismatch.

Operationally, magnetic-field measurements at the HENON orbit appear directly useful, but velocity-based storm forecasting does not.
The project’s overall conclusion was that the HENON mission is scientifically and operationally promising, but only partly as a direct forecasting platform. Magnetic measurements appear robust enough to be used directly, whereas accurate geomagnetic storm prediction will require additional physics-based modelling of CME velocity evolution between the HENON orbit and Earth.

## Repository Structure
### `main.py`
The main analysis pipeline.
This script simulates the full HENON-style workflow by:

- generating the DRO-based mission geometry,
- identifying crossover periods between archival spacecraft and the simulated HENON orbit,
- collecting in-situ and L1 comparison data,
- matching crossover periods to known CME events,
- producing event plots and event-level CSV outputs for later analysis.

### `orbit.py`
The orbit and geometry module.
This contains the Circular Restricted Three-Body Problem (CR3BP) implementation used to model the HENON Distant Retrograde Orbit, generate the satellite constellation geometry, and transform the trajectory into the coordinate systems used for event selection and plotting.

### Analysis modules
These scripts operate on the event-level CSV files generated by `main.py` and quantify how different CME properties evolve between the proxy HENON orbit and L1.
In the current project these include:

- absolute magnetic-field analysis
- southward magnetic-field analysis
- velocity analysis
- geomagnetic storm / Dst injection analysis

### `requirements.txt`
Lists the Python dependencies required to run the project.

## Data Sources
### JPL Horizons / SunPy ephemerides
Used to obtain archival spacecraft trajectories and Earth ephemerides, allowing identification of periods where real spacecraft occupied regions comparable to the simulated HENON orbit.

### Archival in-situ spacecraft data
Archival in-situ spacecraft data was used to build the proxy HENON dataset from spacecraft that passed through HENON-like regions.
The main proxy spacecraft used in the project are:

- STEREO-A
- Solar Orbiter

These provide magnetic-field and plasma measurements used as stand-ins for future HENON observations.

### OMNI L1 solar wind data
Used as the near-Earth comparison dataset.
L1 observations act as a proxy for what would later arrive at Earth and provide the baseline against which the HENON-orbit measurements are compared.

### Richardson & Cane ICME catalogue
Used to identify which crossover periods coincide with known CME / ICME events, allowing the construction of a CME-focused proxy dataset rather than a purely geometric orbit-overlap dataset.

## Installation
Install all required packages from `requirements.txt`

Some scripts require you to set local input/output paths in their configuration sections before use, for example:

- the folder containing event CSV files produced by `main.py`
- the folder where analysis figures should be saved

## Usage
For a typical workflow:

1. run `main.py` to generate event-level outputs
2. point the analysis modules at the resulting CSV directory
3. run the chosen analysis scripts

## Outputs
- Event-level CSV files containing proxy HENON / DRO spacecraft measurements, aligned L1 comparison data, CME window metadata, and derived geometry quantities.
- Event plots showing spacecraft–orbit geometry, crossover periods, and in-situ comparisons between proxy HENON observations and L1 data.
- Summary analysis figures including parity plots, correlation-distance trends, RMSE-distance trends, decomposition plots, and storm-prediction comparisons.
- Derived forecasting metrics such as predicted minimum Dst, integrated injection proxies, and hybrid parameter-substitution results.
- Optional saved figure files in PNG or PDF format for use in reports, presentations, or further analysis.

Example Outputs:

![Alt text]('Example Outputs'/'V shock.png')

## Limitations
First, the analysis is based on a proxy dataset constructed from archival spacecraft observations, rather than direct measurements from the HENON mission itself. The spacecraft used in this study were selected because their trajectories passed through regions comparable to the simulated HENON orbit, allowing a useful first-order approximation of what a future HENON spacecraft might observe. However, these proxy measurements are not identical to true mission data and therefore cannot fully capture the exact geometry, instrumentation, or operational cadence of the future HENON constellation.

Second, the study is based on a limited sample of CME events. Although the event-selection pipeline was designed to identify the best available crossover periods, the final dataset is constrained by the availability of suitable archival spacecraft alignments, known CME intervals, and data quality. This limits the statistical size of the sample and means that some findings should be interpreted as strong indications rather than final mission-validation results.

Third, the analysis depends on existing external event catalogues and available spacecraft coverage. The construction of the proxy dataset required both favourable orbital alignment and the presence of a catalogued CME event during the crossover period. As a result, the final sample is shaped not only by the physical questions being asked, but also by the observational limitations of the current heliophysics archive.

These limitations are expected to be addressed in part from 2027 onward, when the first satellite in the HENON project is scheduled to be deployed. Once direct in-situ measurements from the HENON orbit become available, this analysis can be repeated using real mission observations rather than proxy spacecraft data. This will allow the methodology developed in this project to be validated and extended before the full HENON constellation is deployed, providing a much stronger test of the mission’s forecasting capability under real operational conditions.

## Dissertation / Report Context
This repository was developed as part of my MSci research project in Physics at Imperial College London. The project investigates the feasibility of the proposed ESA HENON mission as an upstream space-weather monitoring platform by studying how CME properties evolve between a simulated HENON Distant Retrograde Orbit and the L1 point.

The code in this repository underpins the computational workflow used in the dissertation, including orbit simulation, proxy dataset construction from archival spacecraft measurements, event matching, parameter-evolution analysis, and geomagnetic storm prediction comparisons. The accompanying written dissertation provides the full scientific background, methodology, interpretation, and literature context for the results summarised here.

## Author
**Henry Hodges**  
MSci Physics, Imperial College London  
Contact Email: Henryhodges7@gmail.com
