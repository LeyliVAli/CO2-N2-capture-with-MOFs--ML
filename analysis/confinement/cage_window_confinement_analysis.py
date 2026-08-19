# Settings

INPUT_FILE="..."

ID="filename"
P="p/bar"

DI="Di"
DF="Df"
DENSITY="Density"

Q="mmol/g"
S="S(g1)"
HOA="hoa/kcal/mol"

ACCESS_COL="Reference_probe_accessible"

P_LOW=0.015
P_HIGH=0.150
ALPHA=0.05


# Load data

df=pd.read_csv(INPUT_FILE)
df.columns=df.columns.astype(str).str.strip()
df=df.loc[:,~df.columns.str.startswith("Unnamed:")].copy()

required=[ID,P,DI,DF,DENSITY,Q,S,HOA]

missing=[
    c for c in required
    if c not in df.columns
]

if missing:
    raise KeyError(f"Missing columns: {missing}")

for c in [P,DI,DF,DENSITY,Q,S,HOA]:
    df[c]=pd.to_numeric(
        df[c],
        errors="coerce"
    )


# Confinement descriptors

df["Gamma_CW"]=df[DI]/df[DF]

df["DeltaD"]=(
    df[DI]-df[DF]
)

df["P_scale"]=np.sqrt(
    df[DI]*df[DF]
)

df["C_CW"]=np.log(
    df["Gamma_CW"]
)


# Benjamini-Hochberg correction

def add_bh(table,p_col="p_value"):

    table=table.copy()

    valid=table[p_col].notna()

    table["FDR_q"]=np.nan

    if valid.any():

        table.loc[
            valid,
            "FDR_q"
        ]=multipletests(
            table.loc[valid,p_col],
            alpha=ALPHA,
            method="fdr_bh"
        )[1]

    return table


# Structure-level table

low=df[
    np.isclose(df[P],P_LOW)
].set_index(ID).sort_index()

high=df[
    np.isclose(df[P],P_HIGH)
].set_index(ID).sort_index()

if set(low.index)!=set(high.index):
    raise ValueError(
        "Low/high structure sets differ."
    )

st=pd.DataFrame(index=low.index)

for c in [
    DI,
    DF,
    DENSITY,
    "Gamma_CW",
    "DeltaD",
    "P_scale",
    "C_CW"
]:
    st[c]=low[c]

st["q_low"]=low[Q]
st["q_high"]=high[Q]

st["HoA_low"]=low[HOA]
st["HoA_high"]=high[HOA]

st["S_low"]=low[S]
st["S_high"]=high[S]

st=st.reset_index()


# Gamma-CW association with Di and Df

rows=[]

for descriptor in [DI,DF]:

    temp=st[
        ["Gamma_CW",descriptor]
    ].dropna()

    rho,pval=spearmanr(
        temp["Gamma_CW"],
        temp[descriptor]
    )

    rows.append({
        "Relationship":
            f"Gamma_CW vs {descriptor}",
        "N":len(temp),
        "Spearman_rho":rho,
        "p_value":pval
    })

gamma_components=pd.DataFrame(rows)

gamma_components.to_csv(
    "Gamma_Di_Df_Correlations.csv",
    index=False
)


# Pressure-resolved Gamma-CW and DeltaD correlations

rows=[]

targets={
    Q:"Uptake",
    HOA:"HoA",
    S:"Selectivity"
}

for pressure in [P_LOW,P_HIGH]:

    d=df[
        np.isclose(df[P],pressure)
    ]

    for descriptor in [
        "Gamma_CW",
        "DeltaD"
    ]:

        for target,target_name in targets.items():

            temp=d[
                [descriptor,target]
            ].dropna()

            rho,pval=spearmanr(
                temp[descriptor],
                temp[target]
            )

            rows.append({
                "Descriptor":descriptor,
                "Target":target_name,
                "Pressure_bar":pressure,
                "N":len(temp),
                "Spearman_rho":rho,
                "p_value":pval
            })

pressure_corr=pd.DataFrame(rows)

pressure_corr=add_bh(
    pressure_corr
)

pressure_corr.to_csv(
    "Gamma_DeltaD_PressureResolved.csv",
    index=False
)


# Partial Spearman function

def partial_spearman(x,y,controls):

    data=pd.concat(
        [
            x.rename("x"),
            y.rename("y"),
            controls
        ],
        axis=1
    ).replace(
        [np.inf,-np.inf],
        np.nan
    ).dropna()

    ranked=data.rank()

    C=ranked[
        controls.columns
    ].to_numpy()

    C=np.column_stack([
        np.ones(len(C)),
        C
    ])

    xr=ranked["x"].to_numpy()
    yr=ranked["y"].to_numpy()

    bx=np.linalg.lstsq(
        C,xr,rcond=None
    )[0]

    by=np.linalg.lstsq(
        C,yr,rcond=None
    )[0]

    rx=xr-C@bx
    ry=yr-C@by

    rho,pval=pearsonr(
        rx,ry
    )

    return rho,pval,len(data)


# Df-controlled Gamma-CW correlations

target_map={
    "q_low":P_LOW,
    "q_high":P_HIGH,
    "HoA_low":P_LOW,
    "HoA_high":P_HIGH,
    "S_low":P_LOW,
    "S_high":P_HIGH
}

rows=[]

for target,pressure in target_map.items():

    rho,pval,n=partial_spearman(
        st["C_CW"],
        st[target],
        st[[DF]]
    )

    rows.append({
        "Analysis":"Gamma_CW | Df",
        "Target":target,
        "Pressure_bar":pressure,
        "N":n,
        "Partial_rho":rho,
        "p_value":pval
    })

gamma_df_controlled=add_bh(
    pd.DataFrame(rows)
)

gamma_df_controlled.to_csv(
    "Gamma_Df_Controlled.csv",
    index=False
)


# Reciprocal Df-controlled-for-Gamma test

rows=[]

for target,pressure in target_map.items():

    rho,pval,n=partial_spearman(
        st[DF],
        st[target],
        st[["C_CW"]]
    )

    rows.append({
        "Analysis":"Df | Gamma_CW",
        "Target":target,
        "Pressure_bar":pressure,
        "N":n,
        "Partial_rho":rho,
        "p_value":pval
    })

df_gamma_controlled=add_bh(
    pd.DataFrame(rows)
)

df_gamma_controlled.to_csv(
    "Df_Gamma_Controlled.csv",
    index=False
)


# Pore-scale and density controlled Gamma-CW correlations

rows=[]

controls=st[
    ["P_scale",DENSITY]
]

for target,pressure in target_map.items():

    rho,pval,n=partial_spearman(
        st["C_CW"],
        st[target],
        controls
    )

    rows.append({
        "Target":target,
        "Pressure_bar":pressure,
        "Controls":"P_scale + Density",
        "N":n,
        "Partial_rho":rho,
        "p_value":pval
    })

structural_control=add_bh(
    pd.DataFrame(rows)
)

structural_control.to_csv(
    "Gamma_PoreScale_Density_Controlled.csv",
    index=False
)


# HoA-controlled Gamma-CW correlations

hoa_controls={
    "q_low":"HoA_low",
    "q_high":"HoA_high",
    "S_low":"HoA_low",
    "S_high":"HoA_high"
}

rows=[]

for target,hoa_col in hoa_controls.items():

    pressure=(
        P_LOW
        if target.endswith("_low")
        else P_HIGH
    )

    rho,pval,n=partial_spearman(
        st["C_CW"],
        st[target],
        st[[hoa_col]]
    )

    rows.append({
        "Target":target,
        "Pressure_bar":pressure,
        "Control":hoa_col,
        "N":n,
        "Partial_rho":rho,
        "p_value":pval
    })

hoa_controlled=add_bh(
    pd.DataFrame(rows)
)

hoa_controlled.to_csv(
    "Gamma_HoA_Controlled.csv",
    index=False
)


# Df-HoA linear relationship

hoa_structure=(
    st["HoA_low"]+
    st["HoA_high"]
)/2

x=st[DF].to_numpy()
y=hoa_structure.to_numpy()*4.184

mask=np.isfinite(x)&np.isfinite(y)

linear=np.polyfit(
    x[mask],
    y[mask],
    1
)

pred=np.polyval(
    linear,
    x[mask]
)

r2_linear=r2_score(
    y[mask],
    pred
)


# Reciprocal Df-HoA relationship

x_inv=1/x[mask]

reciprocal=np.polyfit(
    x_inv,
    y[mask],
    1
)

pred_inv=np.polyval(
    reciprocal,
    x_inv
)

r2_reciprocal=r2_score(
    y[mask],
    pred_inv
)

df_hoa_models=pd.DataFrame([
    {
        "Model":"HoA ~ Df",
        "Slope":linear[0],
        "Intercept":linear[1],
        "R2":r2_linear
    },
    {
        "Model":"HoA ~ 1/Df",
        "Slope":reciprocal[0],
        "Intercept":reciprocal[1],
        "R2":r2_reciprocal
    }
])

df_hoa_models.to_csv(
    "Df_HoA_Models.csv",
    index=False
)


# Gamma-CW × pressure interaction

df["High_pressure"]=np.where(
    np.isclose(df[P],P_HIGH),
    1,
    0
)

interaction_targets=[
    (Q,"Uptake","log"),
    (HOA,"HoA","raw"),
    (S,"Selectivity","log")
]

rows=[]

for target,target_name,scale in interaction_targets:

    d=df[
        [
            ID,
            "Gamma_CW",
            "High_pressure",
            target
        ]
    ].dropna().copy()

    if scale=="log":

        d=d[
            d[target]>0
        ].copy()

        d["Y"]=np.log(
            d[target]
        )

    else:
        d["Y"]=d[target]

    fit=smf.mixedlm(
        "Y ~ Gamma_CW * High_pressure",
        data=d,
        groups=d[ID]
    ).fit(
        reml=False
    )

    term="Gamma_CW:High_pressure"

    ci=fit.conf_int().loc[term]

    rows.append({
        "Target":target_name,
        "beta3":fit.params[term],
        "CI_low":ci.iloc[0],
        "CI_high":ci.iloc[1],
        "p_value":fit.pvalues[term]
    })

interaction_results=add_bh(
    pd.DataFrame(rows)
)

interaction_results.to_csv(
    "Gamma_Pressure_Interaction.csv",
    index=False
)


# Gamma-CW quartile analysis

st["Gamma_quartile"]=pd.qcut(
    st["Gamma_CW"],
    q=4,
    labels=[
        "Q1",
        "Q2",
        "Q3",
        "Q4"
    ]
)

quartile_targets=[
    "q_low",
    "q_high",
    "HoA_low",
    "HoA_high",
    "S_low",
    "S_high"
]

rows=[]

for quartile,g in st.groupby(
    "Gamma_quartile",
    observed=True
):

    for target in quartile_targets:

        x=g[target].dropna()

        rows.append({
            "Gamma_quartile":quartile,
            "Target":target,
            "N":len(x),
            "Median":x.median(),
            "Q1":x.quantile(.25),
            "Q3":x.quantile(.75)
        })

quartile_profile=pd.DataFrame(rows)

quartile_profile.to_csv(
    "Gamma_Quartile_Profile.csv",
    index=False
)


# Accessibility-subset Df correlations

if ACCESS_COL in df.columns:

    accessible=df[
        df[ACCESS_COL].astype(str)
        .str.lower()
        .isin([
            "yes",
            "true",
            "1"
        ])
    ].copy()

    rows=[]

    for pressure in [
        P_LOW,
        P_HIGH
    ]:

        d=accessible[
            np.isclose(
                accessible[P],
                pressure
            )
        ]

        for target,target_name in targets.items():

            temp=d[
                [DF,target]
            ].dropna()

            rho,pval=spearmanr(
                temp[DF],
                temp[target]
            )

            rows.append({
                "Target":target_name,
                "Pressure_bar":pressure,
                "N":len(temp),
                "Df_Spearman_rho":rho,
                "p_value":pval
            })

    accessibility_results=pd.DataFrame(
        rows
    )

    accessibility_results.to_csv(
        "Accessible_Subset_Df_Correlations.csv",
        index=False
    )


# LOWESS plots for Gamma-CW and DeltaD

plot_targets=[
    (
        Q,
        "CO2 uptake (mmol/g)"
    ),
    (
        HOA,
        "Heat of adsorption (kcal/mol)"
    ),
    (
        S,
        "CO2/N2 selectivity"
    )
]

for descriptor in [
    "Gamma_CW",
    "DeltaD"
]:

    for target,ylabel in plot_targets:

        fig,ax=plt.subplots(
            figsize=(6.5,5.2)
        )

        for pressure,label in [
            (P_LOW,"0.015 bar"),
            (P_HIGH,"0.150 bar")
        ]:

            d=df[
                np.isclose(
                    df[P],
                    pressure
                )
            ][
                [descriptor,target]
            ].dropna().sort_values(
                descriptor
            )

            smooth=lowess(
                d[target],
                d[descriptor],
                frac=.35,
                return_sorted=True
            )

            ax.scatter(
                d[descriptor],
                d[target],
                alpha=.4,
                label=label
            )

            ax.plot(
                smooth[:,0],
                smooth[:,1]
            )

        ax.set_xlabel(descriptor)
        ax.set_ylabel(ylabel)
        ax.legend(frameon=False)

        plt.tight_layout()
        plt.show()
