import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PROGRAM 2 : SUMMARY, STATISTICS AND VISUALIZATION
# ============================================================

# ------------------------------------------------------------
# 1. LOAD DATASET
# ------------------------------------------------------------

df = pd.read_csv("music library songs.csv")

# Create duplicate dataset
music = df.copy()

print("=" * 90)
print("PROGRAM 2 : SUMMARY, STATISTICS AND VISUALIZATION")
print("=" * 90)


# ------------------------------------------------------------
# 2. DATASET SUMMARY
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("DATASET SUMMARY")
print("=" * 90)

print("\nTotal Songs   :", len(music))
print("Total Columns :", music.shape[1])
print("Dataset Shape :", music.shape)


# ------------------------------------------------------------
# 3. COLUMN DATA TYPES
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("COLUMN DATA TYPES")
print("=" * 90)

print(music.dtypes.to_string())


# ------------------------------------------------------------
# 4. FIRST 3 SONG TITLES
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("FIRST 3 SONG TITLES")
print("=" * 90)

print(music["Song Title"].head(3).to_string(index=False))


# ------------------------------------------------------------
# 5. LAST 3 RECORDS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("LAST 3 RECORDS")
print("=" * 90)

print(music.tail(3).to_string(index=False))


# ------------------------------------------------------------
# 6. UNIQUE VALUES
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("NUMBER OF UNIQUE VALUES")
print("=" * 90)

print(music.nunique().to_string())


# ------------------------------------------------------------
# 7. NULL VALUES
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("NULL VALUES")
print("=" * 90)

print(music.isnull().sum().to_string())


# ------------------------------------------------------------
# 8. SUMMARY STATISTICS
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("SUMMARY STATISTICS")
print("=" * 90)

print(music.describe(include="all").T.to_string())


# ------------------------------------------------------------
# 9. MEMORY USAGE
# ------------------------------------------------------------

print("\n" + "=" * 90)
print("MEMORY USAGE")
print("=" * 90)

print(music.memory_usage(deep=True).sum(), "bytes")


# ============================================================
# GRAPH 1
# TOP 10 ARTISTS - HORIZONTAL BAR CHART
# ============================================================

top_artists = music["Artist Name 1"].value_counts().head(10)

plt.figure(figsize=(9, 5))

top_artists.sort_values().plot(kind="barh")

plt.title("Top 10 Artists in Music Library")
plt.xlabel("Number of Songs")
plt.ylabel("Artist")

# Display values on bars
for i, v in enumerate(top_artists.sort_values()):
    plt.text(v + 0.1, i, str(v), va="center")

plt.tight_layout()

plt.savefig(
    "output/graph_1_top_artists_horizontal.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 2
# TOP 10 ARTISTS - VERTICAL BAR CHART
# ============================================================

top_artists = music["Artist Name 1"].value_counts().head(10)

plt.figure(figsize=(11, 6))

plt.bar(
    top_artists.index,
    top_artists.values,
    edgecolor="black"
)

plt.title("Top 10 Artists")
plt.xlabel("Artist")
plt.ylabel("Songs")

plt.xticks(rotation=45)

plt.grid(
    axis="y",
    linestyle="--",
    alpha=0.5
)

plt.tight_layout()

plt.savefig(
    "output/graph_2_top_artists_vertical.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 3
# ALBUM DISTRIBUTION - PIE CHART
# ============================================================

top_albums = music["Album Title"].value_counts().head(6)

plt.figure(figsize=(8, 8))

plt.pie(
    top_albums,
    labels=top_albums.index,
    autopct="%1.1f%%",
    startangle=120
)

plt.title("Album Distribution")

plt.tight_layout()

plt.savefig(
    "output/graph_3_album_distribution.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 4
# DISTRIBUTION OF SONGS PER ARTIST - HISTOGRAM
# ============================================================

artist_freq = music["Artist Name 1"].value_counts()

plt.figure(figsize=(8, 5))

plt.hist(
    artist_freq,
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Songs per Artist")
plt.xlabel("Songs")
plt.ylabel("Frequency")

plt.tight_layout()

plt.savefig(
    "output/graph_4_songs_per_artist_histogram.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 5
# TOP 10 ALBUMS - HORIZONTAL BAR CHART
# ============================================================

albums = music["Album Title"].value_counts().head(10)

plt.figure(figsize=(10, 6))

plt.barh(
    albums.index,
    albums.values
)

plt.title("Top Albums")
plt.xlabel("Song Count")

plt.grid(
    axis="x",
    linestyle=":"
)

plt.tight_layout()

plt.savefig(
    "output/graph_5_top_albums.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 6
# ARTIST POPULARITY TREND - LINE GRAPH
# ============================================================

artists = music["Artist Name 1"].value_counts().head(10)

plt.figure(figsize=(11, 5))

plt.plot(
    artists.index,
    artists.values,
    marker="o",
    linewidth=3
)

plt.xticks(rotation=45)

plt.title("Artist Popularity Trend")
plt.xlabel("Artist")
plt.ylabel("Songs")

plt.grid(True)

plt.tight_layout()

plt.savefig(
    "output/graph_6_artist_popularity.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# GRAPH 7
# ARTIST SONG COUNT SPREAD - BOXPLOT
# ============================================================

plt.figure(figsize=(6, 5))

plt.boxplot(
    music["Artist Name 1"].value_counts(),
    patch_artist=True
)

plt.title("Artist Song Count Spread")
plt.ylabel("Songs")

plt.tight_layout()

plt.savefig(
    "output/graph_7_artist_song_count_boxplot.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()
plt.close()


# ============================================================
# COMPLETION
# ============================================================

print("\n" + "=" * 90)
print("ALL GRAPHS GENERATED SUCCESSFULLY")
print("=" * 90)

print("\nOutput files saved in the 'output' folder:")
print("1. graph_1_top_artists_horizontal.png")
print("2. graph_2_top_artists_vertical.png")
print("3. graph_3_album_distribution.png")
print("4. graph_4_songs_per_artist_histogram.png")
print("5. graph_5_top_albums.png")
print("6. graph_6_artist_popularity.png")
print("7. graph_7_artist_song_count_boxplot.png")

print("\n" + "=" * 90)
print("PROGRAM EXECUTED SUCCESSFULLY")
print("=" * 90)