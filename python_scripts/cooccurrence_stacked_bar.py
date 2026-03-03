#!/usr/bin/env python3
"""
Co-occurrence stacked bar plots
"""

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import os

matplotlib.rcParams['svg.fonttype'] = 'none'  # Embed fonts in SVG

input_file = "{input_file}.txt"
output_dir = ""

os.makedirs(output_dir, exist_ok=True)

df = pd.read_csv(input_file, sep='\t')

df['prop_a_both'] = df['a_both'] / df['N']
df['prop_b_intestini_only'] = df['b_intestini_only'] / df['N']
df['prop_c_smithii_only'] = df['c_smithii_only'] / df['N']

colors = {
    'a_both': '#8CC63F',
    'b_intestini_only': '#4A6A90',
    'c_smithii_only': '#EDBF1B'
}

fig, ax = plt.subplots(figsize=(10, max(6, len(df) * 0.5)))

y_positions = range(len(df))

bar1 = ax.barh(y_positions, df['prop_a_both'], 
               color=colors['a_both'], label='Both species', edgecolor='white', linewidth=0.5)
bar2 = ax.barh(y_positions, df['prop_b_intestini_only'], 
               left=df['prop_a_both'],
               color=colors['b_intestini_only'], label='M. intestini only', edgecolor='white', linewidth=0.5)
bar3 = ax.barh(y_positions, df['prop_c_smithii_only'], 
               left=df['prop_a_both'] + df['prop_b_intestini_only'],
               color=colors['c_smithii_only'], label='M. smithii only', edgecolor='white', linewidth=0.5)

ax.set_yticks(y_positions)
ax.set_yticklabels(df['country'])

ax.set_xlim(0, 1)
ax.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax.set_xticklabels(['0%', '25%', '50%', '75%', '100%'])

ax.set_xlabel('Proportion', fontsize=12, fontweight='bold')
ax.set_ylabel('Country', fontsize=12, fontweight='bold')
ax.set_title('Co-occurrence of Methanogen Species by Country', fontsize=14, fontweight='bold', pad=20)

ax.legend(loc='upper right', frameon=True, fancybox=False, shadow=False)

ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.set_axisbelow(True)

plt.tight_layout()

png_output = os.path.join(output_dir, 'cooccurrence_stacked_bar.png')
plt.savefig(png_output, dpi=300, bbox_inches='tight', facecolor='white')
print(f"PNG saved to: {png_output}")

svg_output = os.path.join(output_dir, 'cooccurrence_stacked_bar.svg')
plt.savefig(svg_output, format='svg', bbox_inches='tight', facecolor='white')
print(f"SVG saved to: {svg_output}")

print("\n=== Summary Statistics ===")
print(f"Total countries: {len(df)}")
print(f"\nTotal samples by country:")
for _, row in df.iterrows():
    print(f"  {row['country']}: N = {row['N']}")

print("\n=== Proportions ===")
for _, row in df.iterrows():
    print(f"\n{row['country']}:")
    print(f"  Both species: {row['prop_a_both']:.2%}")
    print(f"  M. intestini only: {row['prop_b_intestini_only']:.2%}")
    print(f"  M. smithii only: {row['prop_c_smithii_only']:.2%}")

plt.close()
print("\nDone")