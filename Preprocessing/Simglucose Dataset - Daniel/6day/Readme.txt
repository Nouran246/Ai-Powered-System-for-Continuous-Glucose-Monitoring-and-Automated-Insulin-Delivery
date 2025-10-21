Preproccessing on data Generated from the FDA approved T1D simulator developed at the University of Virginia

Man CD, Micheletto F, Lv D, Breton M, Kovatchev B, Cobelli C. The UVA/PADOVA Type 1 Diabetes Simulator: New Features. J Diabetes Sci Technol. 2014 Jan;8(1):26-34. doi: 10.1177/1932296813514502. Epub 2014 Jan 1. PMID: 24876534; PMCID: PMC4454102.


The data was generated using the python Wrapper (Simglucose) developed by Jinyu Xie.

Jinyu Xie. Simglucose v0.2.1 (2018) [Online]. Available: https://github.com/jxx123/simglucose. Accessed on: Month-Date-Year.


Data Description:

The simulator includes 30 virtual patients in different age groups (10 adolescents, 10 adults, 10 children)

Read more on how the virtual patients were generated here:
https://github.com/jxx123/simglucose/blob/master/definitions_of_vpatient_parameters.md


The data was cleaned from missing values by forward filling where applicable.

Some feature engineering was done to provide ML/DL friendly formats to Patient Types, hours since last meal and sensor reading capping for distorted values.



Daniel Michel Et al.