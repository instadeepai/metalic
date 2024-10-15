import os

BASEDIR = os.path.dirname((os.path.dirname(__file__)))

PROTEIN_ALPHABET = "ARNDCQEGHILKMFPSTWYV"

VAL_ASSAY_NAMES = [
    "BLAT_ECOLX_Jacquier_2013",
    "CALM1_HUMAN_Weile_2017",
    "DYR_ECOLI_Thompson_2019",
    "DLG4_RAT_McLaughlin_2012",
    "REV_HV1H2_Fernandes_2016",
    "TAT_HV1BR_Fernandes_2016",
    "RL40A_YEAST_Roscoe_2013",
    "P53_HUMAN_Giacomelli_2018_WT_Nutlin",
]

MULTIPLES_ASSAY_NAMES = [
    "PABP_YEAST_Melamed_2013",
    "CAPSD_AAV2S_Sinai_2021",
    "GFP_AEQVI_Sarkisyan_2016",
    "GRB2_HUMAN_Faure_2021",
    "HIS7_YEAST_Pokusaeva_2019",
    # "SPG1_STRSG_Wu_2016",  no zero shot data
]

SUBS_ZERO_SHOT_COLS = [
    "Tranception_L",
    "ESM1v",
    "MSA_Transformer",
    "DeepSequence",
    "TranceptEVE_L",
    "ESM-IF1",
]

SUBS_ZERO_SHOT_COLS_to_index = {  # columns in files have different names
    "Tranception_L": "avg_score",
    "ESM1v": "Ensemble_ESM1v",
    "MSA_Transformer": "esm_msa1b_t12_100M_UR50S_ensemble",
    "DeepSequence": "evol_indices_ensemble",
    "TranceptEVE_L": "avg_score",
    "ESM-IF1": "esmif1_ll",
}
