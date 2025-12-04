#!/bin/bash
# Pipeline for VB Gene Clustering, Homology Search,
# Sequence Retrieval, Re-clustering, MSA, and HMM Construction
# Author: <Your Name>
# Supplementary Script for: <Your Study Name>

### Activate analysis environment
conda activate metaphlan4

### 1. First-round clustering of original VB gene sequences
# Input:  All VB gene FASTA files stored in: VB_SEQ/
# Output: Clustered sequences stored in: VB_SEQ_cluster/

mkdir -p VB_SEQ_cluster

# Generate CD-HIT commands for each gene file
ls VB_SEQ/* | while read i; do
    name=$(basename "$i" | cut -d'.' -f1)
    echo "cd-hit -i ${i} -o VB_SEQ_cluster/${name}.fa -c 0.95 -G 0 -aS 0.9 -g 1 -d 0"
done > run_cluster.sh

# Run the clustering script
sh run_cluster.sh

# Remove .clstr files (cluster reports), not needed
rm VB_SEQ_cluster/*.clstr


### 2. Homology search against UniProt database
# DIAMOND database has been built already:
# diamond makedb --in uniprot.fa --db uniprot

mkdir -p diamond_results

# Loop through clustered VB genes and run DIAMOND BLASTp
for file in VB_SEQ_cluster/*.fa; do
  diamond blastp \
    --more-sensitive \
    -p 200 \
    -q "$file" \
    -d uniprot.dmnd \
    -f 6 qseqid sseqid pident length mismatch gapopen qstart qend sstart send evalue bitscore slen \
    -o diamond_results/$(basename "$file" .fa).out \
    --id 60 \
    --evalue 1e-10 \
    -b12 -c1
done


### 3. Extract UniProt IDs from DIAMOND results
mkdir -p uniprot_seq_id
mkdir -p uniprot_seq_id_for_rename

# Extract UniProt IDs (column 2)
for file in diamond_results/*.out; do
    awk '{print $2}' "$file" > uniprot_seq_id/"$(basename "${file%.out}.txt")"
done

# Extract “UniProt_ID  Identity%ID” for renaming later
for file in diamond_results/*.out; do
    awk '{print $2 "\t" $3 "%" $2}' "$file" > uniprot_seq_id_for_rename/"$(basename "${file%.out}.txt")"
done


### 4. Retrieve homologous protein sequences from UniProt
mkdir -p uniprot_hit_seqs

# Use seqtk to extract matching UniProt sequences
for file in uniprot_seq_id/*.txt; do
    output_file="uniprot_hit_seqs/$(basename ${file%.*}.fa)"
    seqtk subseq uniprot.fa $file > $output_file
done


### 5. Replace sequence IDs based on identity mapping
mkdir -p uniprot_hit_seqs_final

# Map IDs using the identity mapping table
for file in uniprot_hit_seqs/*.fa; do
    gene_name_file="uniprot_seq_id_for_rename/$(basename $file)"
    output_file="uniprot_hit_seqs_final/$(basename $file)"
    
    awk 'NR==FNR { map[$1]=$2; next }
         {
            if (substr($1,2) in map)
                print ">" map[substr($1,2)];
            else
                print;
         }' "$gene_name_file" "$file" > "$output_file"
done

# Prefix each sequence ID with the gene filename: VB_gene~identity%~UniProtID
for file in uniprot_hit_seqs_final/*.fa; do
    sed -i "s/^>/>$(basename "${file%.fa}")~/" "$file"
done


### 6. Second-round clustering
mkdir -p VB_SEQ_cluster_final

# Generate CD-HIT commands
ls uniprot_hit_seqs_final/* | while read i; do
    name=$(basename "$i" | cut -d'.' -f1)
    echo "cd-hit -i ${i} -o VB_SEQ_cluster_final/${name}.fa -c 0.95 -G 0 -aS 0.9 -g 1 -d 0"
done > run_cluster_final.sh

# Run clustering
sh run_cluster_final.sh

# Remove .clstr files
rm VB_SEQ_cluster_final/*.clstr


### 7. Multiple Sequence Alignment using MAFFT
mkdir -p VB_mafft

# Generate MAFFT commands
ls VB_SEQ_cluster_final/* | while read i; do
    name=$(basename "$i" | cut -d'.' -f1)
    echo "mafft --auto ${i} > VB_mafft/${name}.fa"
done > run_mafft.sh

sh run_mafft.sh


### 8. Build HMM profiles using HMMER
mkdir -p VB_hmm

# Generate hmmbuild commands
ls VB_mafft/* | while read i; do
    name=$(basename "$i" | cut -d'.' -f1)
    echo "hmmbuild VB_hmm/${name}.hmm ${i}"
done > run_hmm.sh

# Run HMM building
sh run_hmm.sh


### 9. Merge all HMM models into a single database

cat VB_hmm/*.hmm > VB.hmm

echo "Pipeline completed successfully!"
