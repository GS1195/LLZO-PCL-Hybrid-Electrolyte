#!/usr/bin/env python3
# file: summarize_hops_by_side_chunks_from_by_side_csv.py
import pandas as pd

# 1) Load the chunked, by-side counts CSV (made by the chunked analyzer)
df = pd.read_csv("li_transitions_by_side_chunks.csv")
# expected columns: chunk_id,species,direction,side,count,time_ns

# 2) Totals across ALL species per chunk
tot = (df.groupby(["chunk_id","direction","side"])["count"]
         .sum()
         .reset_index())

pivot_all = (tot.pivot_table(index="chunk_id",
                             columns=["direction","side"],
                             values="count",
                             fill_value=0)
               .sort_index(axis=1))
pivot_all.to_csv("hop_counts_by_side_per_chunk_ALLspecies.csv")
print("\nALL species — counts per chunk × direction × side")
print(pivot_all)

# 3) Per-species table (one row per chunk×species)
per_species = (df.groupby(["chunk_id","species","direction","side"])["count"]
                 .sum()
                 .reset_index())

pivot_species = (per_species.pivot_table(index=["chunk_id","species"],
                                         columns=["direction","side"],
                                         values="count",
                                         fill_value=0)
                               .sort_index(axis=1))
pivot_species.to_csv("hop_counts_by_side_per_chunk_PERspecies.csv")
print("\nPer species — counts per chunk × direction × side")
print(pivot_species)
