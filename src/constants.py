
CATEGORIES = ["Membrane","Cytoplasm","Nucleus","Extracellular","Cell membrane","Mitochondrion","Plastid","Endoplasmic reticulum","Lysosome/Vacuole","Golgi apparatus","Peroxisome"]
SS_CATEGORIES = ["NULL", "SP", "TM", "MT", "CH", "TH", "NLS", "NES", "PTS", "GPI"] 

FAST = "Fast"
ACCURATE = "Accurate"
ONEHOT = "OneHot"
BLOSUM = "BLOSUM"
FAST2 = "Fast2"

EMBEDDINGS = {
    FAST: {
        "embeds": "data_files/embeddings/esm1b_swissprot.h5",
        "config": "swissprot_esm1b.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped1k.fasta"
    },
    ACCURATE: {
        "embeds": "data_files/embeddings/prott5_swissprot.h5",
        "config": "swissprot_prott5.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped4k.fasta"
    },
    ONEHOT: {
        "embeds": "data_files/embeddings/onehot_swissprot.h5",
        "config": "swissprot_onehot.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped1k.fasta"
    },
    BLOSUM: {
        "embeds": "data_files/embeddings/blosum_swissprot.h5",
        "config": "swissprot_blosum.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped1k.fasta"
    },
    FAST2: {
        "embeds": "data_files/embeddings/esm2b_swissprot.h5",
        "config": "swissprot_esm2b.yaml",
        "source_fasta": "data_files/deeploc_swissprot_clipped1k.fasta"
    },
}

SIGNAL_DATA = "data_files/multisub_ninesignals.pkl"
LOCALIZATION_DATA = "./data_files/multisub_5_partitions_unique.csv"

BATCH_SIZE = 128
SUP_LOSS_MULT = 0.1
REG_LOSS_MULT = 0.1

