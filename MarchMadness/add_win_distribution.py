import json
import os

# Path to the notebook file
notebook_path = 'March_Madness_Prediction.ipynb'
output_path = 'March_Madness_Prediction_updated.ipynb'

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Define the new code cell with the display_win_distribution function
new_code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Add the display_win_distribution function to the MarchMadnessPredictor class\n",
        "def display_win_distribution(self, min_season=None, max_season=None, title=None):\n",
        "    \"\"\"\n",
        "    Displays a heatmap of win distribution by seed matchup.\n",
        "    \n",
        "    Parameters:\n",
        "    -----------\n",
        "    min_season : int, optional\n",
        "        Minimum season to include in the analysis\n",
        "    max_season : int, optional\n",
        "        Maximum season to include in the analysis\n",
        "    title : str, optional\n",
        "        Custom title for the plot\n",
        "    \n",
        "    Returns:\n",
        "    --------\n",
        "    None, displays the plot\n",
        "    \"\"\"\n",
        "    # Filter tournament data by season if specified\n",
        "    df = self.tourney_compact_results.copy()\n",
        "    if min_season is not None:\n",
        "        df = df[df['Season'] >= min_season]\n",
        "    if max_season is not None:\n",
        "        df = df[df['Season'] <= max_season]\n",
        "    \n",
        "    # Create a crosstab (matrix) of winner seed vs loser seed\n",
        "    win_matrix = pd.crosstab(\n",
        "        index=df['LSeedValue'],  # Y-axis: losing seeds\n",
        "        columns=df['WSeedValue'],  # X-axis: winning seeds\n",
        "        values=df['Season'],\n",
        "        aggfunc='count'\n",
        "    ).fillna(0)\n",
        "    \n",
        "    # Ensure all seeds from 1-16 are represented\n",
        "    all_seeds = list(range(1, 17))\n",
        "    for seed in all_seeds:\n",
        "        if seed not in win_matrix.index:\n",
        "            win_matrix.loc[seed] = 0\n",
        "        if seed not in win_matrix.columns:\n",
        "            win_matrix[seed] = 0\n",
        "    \n",
        "    # Sort the indices to ensure they're in order from 1-16\n",
        "    win_matrix = win_matrix.reindex(index=all_seeds, columns=all_seeds)\n",
        "    \n",
        "    # Create the heatmap\n",
        "    fig = sp.make_subplots(\n",
        "        rows=1, cols=1,\n",
        "        subplot_titles=[title or f\"March Madness Win Distribution by Seed ({min_season or 'All'}-{max_season or 'Present'})\"]\n",
        "    )\n",
        "    \n",
        "    # Add the heatmap\n",
        "    heatmap = go.Heatmap(\n",
        "        z=win_matrix.values,\n",
        "        x=win_matrix.columns,\n",
        "        y=win_matrix.index,\n",
        "        colorscale='Viridis',\n",
        "        showscale=True,\n",
        "        colorbar=dict(title='Count'),\n",
        "        text=[[f\"{int(val)} wins\" if val > 0 else \"\" for val in row] for row in win_matrix.values],\n",
        "        hoverinfo='text',\n",
        "        hoverongaps=False\n",
        "    )\n",
        "    \n",
        "    fig.add_trace(heatmap)\n",
        "    \n",
        "    # Update layout\n",
        "    fig.update_layout(\n",
        "        height=600,\n",
        "        width=700,\n",
        "        xaxis=dict(\n",
        "            title='Winning Seed',\n",
        "            tickmode='linear',\n",
        "            tick0=1,\n",
        "            dtick=1\n",
        "        ),\n",
        "        yaxis=dict(\n",
        "            title='Losing Seed',\n",
        "            tickmode='linear',\n",
        "            tick0=1,\n",
        "            dtick=1,\n",
        "            autorange='reversed'  # Reverse y-axis to have 1 at the top\n",
        "        )\n",
        "    )\n",
        "    \n",
        "    # Add highlighting for upset cells (higher seed beats lower seed)\n",
        "    for i in range(1, 17):\n",
        "        for j in range(1, 17):\n",
        "            if i == j:\n",
        "                continue\n",
        "            if i < j:  # \"Expected\" outcome: lower seed (i) beats higher seed (j)\n",
        "                # No need to highlight as this is expected\n",
        "                pass\n",
        "            else:  # \"Upset\": higher seed (i) loses to lower seed (j)\n",
        "                if win_matrix.loc[i, j] > 0:\n",
        "                    # Add a rectangle around upset cells\n",
        "                    fig.add_shape(\n",
        "                        type=\"rect\",\n",
        "                        x0=j-0.5, x1=j+0.5,\n",
        "                        y0=i-0.5, y1=i+0.5,\n",
        "                        line=dict(color=\"red\", width=1),\n",
        "                        fillcolor=\"rgba(0,0,0,0)\"\n",
        "                    )\n",
        "    \n",
        "    # Show the plot\n",
        "    fig.show()\n",
        "    \n",
        "    # Calculate and display some statistics\n",
        "    total_games = win_matrix.sum().sum()\n",
        "    expected_wins = sum(win_matrix.iloc[i-1:, i].sum() for i in range(1, 17))  # Sum of lower seeds beating higher seeds\n",
        "    upset_wins = total_games - expected_wins\n",
        "    \n",
        "    print(f\"Total Tournament Games: {int(total_games)}\")\n",
        "    print(f\"Expected Outcomes (lower seed beats higher seed): {int(expected_wins)} ({expected_wins/total_games:.1%})\")\n",
        "    print(f\"Upset Outcomes (higher seed beats lower seed): {int(upset_wins)} ({upset_wins/total_games:.1%})\")\n",
        "    \n",
        "    # Show the most common upsets\n",
        "    upset_matrix = win_matrix.copy()\n",
        "    for i in range(1, 17):\n",
        "        for j in range(1, 17):\n",
        "            if i <= j:  # Not an upset\n",
        "                upset_matrix.iloc[i-1, j-1] = 0\n",
        "    \n",
        "    if upset_matrix.sum().sum() > 0:\n",
        "        top_upsets = []\n",
        "        for i in range(1, 17):\n",
        "            for j in range(1, 17):\n",
        "                if i > j and upset_matrix.iloc[i-1, j-1] > 0:\n",
        "                    top_upsets.append((j, i, int(upset_matrix.iloc[i-1, j-1])))\n",
        "        \n",
        "        top_upsets.sort(key=lambda x: x[2], reverse=True)\n",
        "        \n",
        "        print(\"\\nTop 5 Most Common Upsets:\")\n",
        "        for w_seed, l_seed, count in top_upsets[:5]:\n",
        "            print(f\"Seed #{w_seed} beating Seed #{l_seed}: {count} times\")\n",
        "\n",
        "# Add the function to the MarchMadnessPredictor class\n",
        "MarchMadnessPredictor.display_win_distribution = display_win_distribution"
    ]
}

# Define the usage example code cell
usage_example_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "# Example of using the display_win_distribution function\n",
        "predictor.display_win_distribution()\n",
        "\n",
        "# You can also filter by seasons\n",
        "# predictor.display_win_distribution(min_season=2010, title=\"Tournament Results Since 2010\")"
    ]
}

# Add a markdown cell explaining the function
explanation_cell = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "## Win Distribution Visualization\n",
        "\n",
        "The following function visualizes the win distribution in tournament games based on seed matchups. \n",
        "It creates a heatmap where:\n",
        "- The x-axis represents the winning seed\n",
        "- The y-axis represents the losing seed\n",
        "- The color intensity represents the count of games\n",
        "- Upset games (where a higher seed number beats a lower seed number) are highlighted with red borders\n",
        "\n",
        "This visualization helps identify patterns in tournament outcomes and commonly occurring upsets."
    ]
}

# Find where to insert the new cells - after the last code cell
insert_idx = len(notebook['cells'])
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        insert_idx = i + 1

# Insert the cells at the determined position
notebook['cells'].insert(insert_idx, explanation_cell)
notebook['cells'].insert(insert_idx + 1, new_code_cell)
notebook['cells'].insert(insert_idx + 2, usage_example_cell)

# Write the updated notebook
with open(output_path, 'w', encoding='utf-8') as f:
    json.dump(notebook, f, indent=1)

print(f"Successfully created {output_path} with the display_win_distribution function added.") 