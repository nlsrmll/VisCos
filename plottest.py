import matplotlib.pyplot as plt

# Subplots erstellen: 3 Reihen, 2 Spalten
fig, axes = plt.subplots(3, 2, figsize=(10, 8))

# Titel für jede Reihe
row_titles = ["Row 1 Title", "Row 2 Title", "Row 3 Title"]

# Füge Daten und individuelle Subplot-Titel hinzu
for i, row_axes in enumerate(axes):  # Iteriere durch jede Reihe
    for ax in row_axes:  # Iteriere durch die Spalten in der Reihe
        ax.plot([1, 2, 3], [4, 5, 6])
        ax.set_title(f"Subplot ({i+1})", fontsize=10)

    # Füge den Titel für die gesamte Reihe hinzu
    fig.text(0.5, 0.9 - i * 0.3, row_titles[i], ha="center", fontsize=12, weight="bold")

plt.tight_layout(rect=[0, 0, 1, 0.92])  # Passt Platz für die Zeilentitel an
plt.show()
