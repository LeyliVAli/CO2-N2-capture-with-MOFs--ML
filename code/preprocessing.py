# Input and settings

INPUT_FILE="PATH_TO_DATASET.csv"
TARGET_COL="TARGET_COLUMN_NAME"
GROUP_COL="filename"

SEED=42
TRAIN_SIZE=0.80
ZERO_THRESHOLD=90
VAR_THRESHOLD=1e-10
CORR_THRESHOLD=0.90


# Load data

df=pd.read_csv(INPUT_FILE)
df=df.loc[:,~df.columns.astype(str).str.startswith("Unnamed:")].copy()


# Basic data checks

if TARGET_COL not in df.columns:
    raise ValueError(f"Target column '{TARGET_COL}' not found.")

if GROUP_COL not in df.columns:
    raise ValueError(f"Grouping column '{GROUP_COL}' not found.")

if df.columns.duplicated().any():
    raise ValueError("Duplicate column names detected.")

print("Duplicate rows:",df.duplicated().sum())

missing=df.isna().sum()

if missing.any():
    print("\nColumns with missing values:")
    print(missing[missing>0])
else:
    print("No missing values detected.")

print("\nNon-numeric columns:")
print(df.select_dtypes(exclude=np.number).columns.tolist())


# High-zero feature removal

numeric_cols=[
    c for c in df.select_dtypes(include=np.number).columns
    if c!=TARGET_COL
]

zero_pct=df[numeric_cols].eq(0).mean()*100

high_zero_cols=zero_pct[
    zero_pct>=ZERO_THRESHOLD
].index.tolist()

print(
    f"\nColumns with ≥{ZERO_THRESHOLD}% zeros:",
    high_zero_cols
)

df=df.drop(
    columns=high_zero_cols,
    errors="ignore"
)

print("High-zero columns removed:",len(high_zero_cols))


# Constant feature removal

predictor_cols=[
    c for c in df.columns
    if c not in [TARGET_COL,GROUP_COL]
]

constant_cols=[
    c for c in predictor_cols
    if df[c].nunique(dropna=False)<=1
]

print("\nConstant columns:",constant_cols)

df=df.drop(
    columns=constant_cols,
    errors="ignore"
)

print("Constant columns removed:",len(constant_cols))


# Remove mc and D_mc descriptors

mc_cols=[
    "mc-chi-0-all","mc-chi-1-all","mc-chi-2-all","mc-chi-3-all",
    "mc-Z-0-all","mc-Z-1-all","mc-Z-2-all","mc-Z-3-all",
    "mc-I-0-all","mc-I-1-all","mc-I-2-all","mc-I-3-all",
    "mc-T-0-all","mc-T-1-all","mc-T-2-all","mc-T-3-all",
    "mc-S-0-all","mc-S-1-all","mc-S-2-all","mc-S-3-all",
    "D_mc-chi-0-all","D_mc-chi-1-all","D_mc-chi-2-all","D_mc-chi-3-all",
    "D_mc-Z-0-all","D_mc-Z-1-all","D_mc-Z-2-all","D_mc-Z-3-all",
    "D_mc-I-0-all","D_mc-I-1-all","D_mc-I-2-all","D_mc-I-3-all",
    "D_mc-T-0-all","D_mc-T-1-all","D_mc-T-2-all","D_mc-T-3-all",
    "D_mc-S-0-all","D_mc-S-1-all","D_mc-S-2-all","D_mc-S-3-all"
]

existing_mc=[
    c for c in mc_cols
    if c in df.columns
]

df=df.drop(
    columns=existing_mc,
    errors="ignore"
)

print("\nmc/D_mc columns removed:",len(existing_mc))


# Define predictors and target

X=df.drop(columns=[TARGET_COL])
y=df[TARGET_COL].astype(float)

groups=X[GROUP_COL].astype(str).to_numpy()


# Group-based train/test split

gss=GroupShuffleSplit(
    n_splits=1,
    train_size=TRAIN_SIZE,
    random_state=SEED
)

train_idx,test_idx=next(
    gss.split(
        X,
        y,
        groups=groups
    )
)

X_train=X.iloc[
    train_idx
].reset_index(drop=True)

X_test=X.iloc[
    test_idx
].reset_index(drop=True)

y_train=y.iloc[
    train_idx
].reset_index(drop=True)

y_test=y.iloc[
    test_idx
].reset_index(drop=True)


# Train/test leakage check

overlap=set(
    X_train[GROUP_COL]
)&set(
    X_test[GROUP_COL]
)

if overlap:
    raise RuntimeError(
        f"Data leakage detected: "
        f"{len(overlap)} structures occur in both sets."
    )

print("\nTrain rows:",len(X_train))
print("Test rows:",len(X_test))

print(
    "Train structures:",
    X_train[GROUP_COL].nunique()
)

print(
    "Test structures:",
    X_test[GROUP_COL].nunique()
)

print(
    "Overlapping structures:",
    len(overlap)
)


# Robust scaling

numeric_cols=X_train.select_dtypes(
    include=np.number
).columns.tolist()

scaler=RobustScaler()

X_train[numeric_cols]=scaler.fit_transform(
    X_train[numeric_cols]
)

X_test[numeric_cols]=scaler.transform(
    X_test[numeric_cols]
)

print("\nRobustScaler fitted on train and applied to test.")


# Near-zero variance filtering

variance=X_train[
    numeric_cols
].var()

near_zero_cols=variance[
    variance<=VAR_THRESHOLD
].index.tolist()

print(
    "\nNear-zero variance columns:",
    near_zero_cols
)

X_train=X_train.drop(
    columns=near_zero_cols,
    errors="ignore"
)

X_test=X_test.drop(
    columns=near_zero_cols,
    errors="ignore"
)

print(
    "Near-zero variance columns removed:",
    len(near_zero_cols)
)


# Spearman correlation filtering

numeric_cols=X_train.select_dtypes(
    include=np.number
).columns.tolist()

corr=X_train[
    numeric_cols
].corr(
    method="spearman"
).abs()

upper=corr.where(
    np.triu(
        np.ones(corr.shape),
        k=1
    ).astype(bool)
)

corr_drop=[
    c for c in upper.columns
    if (upper[c]>CORR_THRESHOLD).any()
]

print(
    f"\nColumns removed for "
    f"|Spearman rho|>{CORR_THRESHOLD}:"
)

print(corr_drop)

X_train=X_train.drop(
    columns=corr_drop,
    errors="ignore"
)

X_test=X_test.drop(
    columns=corr_drop,
    errors="ignore"
)

print(
    "Correlation-filtered columns removed:",
    len(corr_drop)
)


# Final data check

if X_train.select_dtypes(
    include=np.number
).isna().any().any():

    raise ValueError(
        "Missing values remain in training predictors."
    )

if X_test.select_dtypes(
    include=np.number
).isna().any().any():

    raise ValueError(
        "Missing values remain in test predictors."
    )

print("\nPreprocessing completed.")

print(
    "Final number of predictors:",
    X_train.shape[1]-1
)


# Visualization: Spearman correlation heatmap

numeric_final=X_train.select_dtypes(
    include=np.number
)

corr_final=numeric_final.corr(
    method="spearman"
)

fig,ax=plt.subplots(
    figsize=(14,12)
)

sns.heatmap(
    corr_final,
    cmap="coolwarm",
    annot=False,
    square=True,
    linewidths=0.1,
    cbar_kws={"shrink":0.75},
    ax=ax
)

ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=90,
    fontsize=9
)

ax.set_yticklabels(
    ax.get_yticklabels(),
    rotation=0,
    fontsize=9
)

plt.tight_layout()
plt.show()


# Visualization: Boxplots

numeric_cols=numeric_final.columns.tolist()

n_cols=4
n_rows=math.ceil(
    len(numeric_cols)/n_cols
)

fig,axes=plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        4*n_cols,
        3*n_rows
    )
)

axes=np.asarray(
    axes
).ravel()

for i,col in enumerate(
    numeric_cols
):
    data=X_train[col].dropna()

    sns.boxplot(
        y=data,
        ax=axes[i],
        width=0.5
    )

    axes[i].set_title(
        col,
        fontsize=9
    )

    axes[i].set_xlabel("")
    axes[i].set_ylabel("")

for ax in axes[
    len(numeric_cols):
]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()


# Visualization: Histogram and KDE

fig,axes=plt.subplots(
    n_rows,
    n_cols,
    figsize=(
        4*n_cols,
        3.2*n_rows
    )
)

axes=np.asarray(
    axes
).ravel()

for i,col in enumerate(
    numeric_cols
):
    data=X_train[col].dropna()

    if data.empty:
        axes[i].axis("off")
        continue

    if data.nunique()<=1:
        axes[i].text(
            0.5,
            0.5,
            "Constant feature",
            ha="center",
            va="center"
        )

        axes[i].axis("off")
        continue

    sns.histplot(
        data=data,
        bins=20,
        kde=True,
        stat="density",
        ax=axes[i]
    )

    axes[i].set_title(
        col,
        fontsize=9
    )

    axes[i].set_xlabel("")
    axes[i].set_ylabel("Density")

for ax in axes[
    len(numeric_cols):
]:
    fig.delaxes(ax)

plt.tight_layout()
plt.show()
