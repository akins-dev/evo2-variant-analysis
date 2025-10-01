import base64
import sys
from venv import create

import modal

# Docker image with evo2 installed
evo2_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.0-devel-ubuntu22.04",
        add_python="3.12",
        # force_build=True,
    )
    .apt_install(
        ["build-essential", "cmake", "ninja-build", "libcudnn8", "libcudnn8-dev", "git", "gcc", "g++"]
    )
    .env({
        "CC": "/usr/bin/gcc",
        "CXX": "/usr/bin/g++",
    })
    # Upgrade pip, setuptools, wheel
    .run_commands("pip install --upgrade pip setuptools wheel")
    # Install Torch (must match CUDA 12.4 runtime)
    .run_commands("pip install torch==2.8.0")
    .run_commands("pip install flash-attn --no-build-isolation")

    .run_commands("git clone https://github.com/ArcInstitute/evo2.git && cd evo2 && pip install -e .")
    .run_commands("pip uninstall -y transformer-engine transformer_engine") 
    .run_commands("pip install 'transformer_engine[pytorch]>=1.13' --no-build-isolation")
    .pip_install_from_requirements("requirements.txt")
)

app = modal.App("variants-analysis-evo2", image=evo2_image)
volume = modal.Volume.from_name("hf_cache", create_if_missing=True)
mount_path = "/root/.cache/huggingface"

@app.function(gpu="H100", volumes={mount_path: volume}, timeout=1000)
def run_brca1_analysis():
    import base64
    from io import BytesIO
    from Bio import SeqIO
    import gzip
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd
    import os
    import seaborn as sns
    from sklearn.metrics import roc_auc_score

    from evo2 import Evo2


    WINDOW_SIZE = 8192

    print("Loading evo2 model...")
    model = Evo2('evo2_7b')
    print("Evo2 model loaded.")

    brca1_df = pd.read_excel(
        '/evo2/notebooks/brca1/41586_2018_461_MOESM3_ESM.xlsx',
        header=2,
    )
    brca1_df = brca1_df[[
        'chromosome', 'position (hg19)', 'reference', 'alt', 'function.score.mean', 'func.class',
    ]]

    # Rename columns
    brca1_df.rename(columns={
        'chromosome': 'chrom',
        'position (hg19)': 'pos',
        'reference': 'ref',
        'alt': 'alt',
        'function.score.mean': 'score',
        'func.class': 'class',
    }, inplace=True)

    # Convert to two-class system
    brca1_df['class'] = brca1_df['class'].replace(['FUNC', 'INT'], 'FUNC/INT')

    # Read the reference genome sequence of chromosome 17
    with gzip.open('/evo2/notebooks/brca1/GRCh37.p13_chr17.fna.gz', "rt") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            seq_chr17 = str(record.seq)
            break

    # Build mappings of unique reference sequences
    ref_seqs = []
    ref_seq_to_index = {}

    # Parse sequences and store indexes
    ref_seq_indexes = []
    var_seqs = []

    # Let's just run a subset of the dataset for speed
    brca1_subset = brca1_df.iloc[:500].copy()

    for _, row in brca1_subset.iterrows():
        # Parse reference and variant sequences from the reference genome sequence.
        p = row["pos"] - 1 # Convert to 0-indexed position
        full_seq = seq_chr17

        ref_seq_start = max(0, p - WINDOW_SIZE//2)
        ref_seq_end = min(len(full_seq), p + WINDOW_SIZE//2)
        ref_seq = seq_chr17[ref_seq_start:ref_seq_end]
        snv_pos_in_ref = min(WINDOW_SIZE//2, p)
        var_seq = ref_seq[:snv_pos_in_ref] + row["alt"] + ref_seq[snv_pos_in_ref+1:]

        # Get or create index for reference sequence
        if ref_seq not in ref_seq_to_index:
            ref_seq_to_index[ref_seq] = len(ref_seqs)
            ref_seqs.append(ref_seq)
        
        ref_seq_indexes.append(ref_seq_to_index[ref_seq])
        var_seqs.append(var_seq)


    ref_seq_indexes = np.array(ref_seq_indexes)

    print(f'Scoring likelihoods of {len(ref_seqs)} reference sequences with Evo 2...')
    ref_scores = model.score_sequences(ref_seqs)

    print(f'Scoring likelihoods of {len(var_seqs)} variant sequences with Evo 2...')
    var_scores = model.score_sequences(var_seqs)

    # Subtract score of corresponding reference sequences from scores of variant sequences
    delta_scores = np.array(var_scores) - np.array(ref_scores)[ref_seq_indexes]

    # Add delta scores to dataframe
    brca1_subset[f'evo2_delta_score'] = delta_scores

    # Calculate AUROC of zero-shot predictions
    y_true = (brca1_subset['class'] == 'LOF')
    auroc = roc_auc_score(y_true, -brca1_subset['evo2_delta_score'])
    print(f'BRCA1 zero-shot variant effect prediction AUROC: {auroc:.4f}')
    

    plt.figure(figsize=(4, 2))

    # Plot stripplot of distributions
    p = sns.stripplot(
        data=brca1_subset,
        x='evo2_delta_score',
        y='class',
        hue='class',
        order=['FUNC/INT', 'LOF'],
        palette=['#777777', 'C3'],
        size=2,
        jitter=0.3,
    )

    # Mark medians from each distribution
    sns.boxplot(showmeans=True,
                meanline=True,
                meanprops={'visible': False},
                medianprops={'color': 'k', 'ls': '-', 'lw': 2},
                whiskerprops={'visible': False},
                zorder=10,
                x="evo2_delta_score",
                y="class",
                data=brca1_subset,
                showfliers=False,
                showbox=False,
                showcaps=False,
                ax=p)
    plt.xlabel('Delta likelihood score, Evo 2')
    plt.ylabel('BRCA1 SNV class')
    plt.tight_layout()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plot_data = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return {
        "variants": brca1_subset.to_dict(orient='records'),
        "plot": plot_data,
        "auroc": auroc,
    }


@app.function()
def brca1_example():
    import base64
    from io import BytesIO
    import matplotlib.pyplot as plt
    import matplotlib.image as mpimg
    print("Running BRCA1 variant analysis with Evo2...")

    # Run inference on BRCA1 variants
    result = run_brca1_analysis.remote()

    if "plot" in result:
        plot_data = base64.b64decode(result["plot"])
        with open("brca1_analysis_plot.png", "wb") as f:
            f.write(plot_data)

        print("Plot saved to brca1_analysis_plot.png")

        # Show plots on returned data
        img = mpimg.imread(BytesIO(plot_data))
        plt.figure(figsize=(10, 5))
        plt.imshow(img)
        plt.axis('off')
        plt.show()


@app.local_entrypoint()
def main():
    brca1_example.local()
