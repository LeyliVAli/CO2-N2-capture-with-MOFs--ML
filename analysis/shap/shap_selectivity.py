TARGET="SEL"
SEED=42
N_BACKGROUND=100
TOP_N=10
N_INTERACTION=400

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"selectivity"
MODEL_DIR=ROOT/"models"/"selectivity"
OUT=ROOT/"outputs"/"shap"/"selectivity"
OUT.mkdir(parents=True,exist_ok=True)

MODEL_FILE=MODEL_DIR/"Final_Model_SEL.joblib"
XTRAIN_FILE=DATA/"X_train_model.csv"
XTEST_FILE=DATA/"X_test_model.csv"
RAW_TRAIN_FILE=DATA/"X_train_physical.csv"
RAW_TEST_FILE=DATA/"X_test_physical.csv"

def read(p):
    d=pd.read_csv(p)
    return d.loc[:,~d.columns.astype(str).str.startswith("Unnamed:")]

def label(f):
    return {
        "p/bar":"Pressure (bar)",
        "Di":r"$D_i$ ($\AA$)",
        "Df":r"$D_f$ ($\AA$)",
        "Density":r"Density (g cm$^{-3}$)",
        "AVA":r"AVA ($\AA^3$)",
        "ASA":r"ASA ($\AA^2$)"
    }.get(f,f)

bundle=joblib.load(MODEL_FILE)
model=bundle["model"]
features=list(bundle["selected_features"])

train=read(XTRAIN_FILE)
test=read(XTEST_FILE)
raw_train=read(RAW_TRAIN_FILE)
raw_test=read(RAW_TEST_FILE)

Xtrain=train[features].apply(pd.to_numeric)
Xtest=test[features].apply(pd.to_numeric)

rng=np.random.default_rng(SEED)
idx=rng.choice(
    len(Xtrain),
    min(N_BACKGROUND,len(Xtrain)),
    replace=False
)
background=Xtrain.iloc[idx]

explainer=shap.TreeExplainer(
    model,
    data=background,
    feature_perturbation="interventional"
)

shap_train=explainer(Xtrain,check_additivity=False)
shap_test=explainer(Xtest,check_additivity=False)

# Global SHAP
def importance(values):
    m=np.asarray(values.values)
    out=pd.DataFrame({
        "Feature":features,
        "Mean_Abs_SHAP":np.abs(m).mean(axis=0)
    }).sort_values(
        "Mean_Abs_SHAP",
        ascending=False
    )
    out.insert(0,"Rank",range(1,len(out)+1))
    return out.reset_index(drop=True)

imp_train=importance(shap_train)
imp_test=importance(shap_test)

imp_train.to_csv(
    OUT/"SHAP_Global_Train_SEL.csv",
    index=False
)

imp_test.to_csv(
    OUT/"SHAP_Global_Test_SEL.csv",
    index=False
)

shap.plots.bar(
    shap_test,
    max_display=TOP_N,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUT/"SHAP_Bar_SEL.png",
    dpi=400,
    bbox_inches="tight"
)
plt.close()

shap.plots.beeswarm(
    shap_test,
    max_display=TOP_N,
    show=False
)
plt.tight_layout()
plt.savefig(
    OUT/"SHAP_Beeswarm_SEL.png",
    dpi=400,
    bbox_inches="tight"
)
plt.close()

# Dependence plots on physical scale
M=np.asarray(shap_test.values)
top_features=imp_test["Feature"].head(3)

for f in top_features:
    x=pd.to_numeric(raw_test[f])
    y=M[:,features.index(f)]

    plt.figure(figsize=(6,5))
    plt.scatter(x,y,alpha=.7)
    plt.axhline(0,ls="--",lw=.8)
    plt.xlabel(label(f))
    plt.ylabel("SHAP value")
    plt.tight_layout()

    plt.savefig(
        OUT/f"SHAP_Dependence_{f}_SEL.png",
        dpi=400,
        bbox_inches="tight"
    )
    plt.close()

# Pressure-resolved SHAP
pressure=pd.to_numeric(raw_train["p/bar"])
Mtrain=np.asarray(shap_train.values)

rows=[]

for p in sorted(pressure.unique()):
    mask=pressure.eq(p).to_numpy()

    imp=pd.DataFrame({
        "Feature":features,
        "Pressure_bar":p,
        "Mean_Abs_SHAP":
            np.abs(Mtrain[mask]).mean(axis=0)
    })

    rows.append(imp)

pd.concat(rows).to_csv(
    OUT/"SHAP_Pressure_Resolved_SEL.csv",
    index=False
)

# SHAP interactions
idx=rng.choice(
    len(Xtrain),
    min(N_INTERACTION,len(Xtrain)),
    replace=False
)

Xi=Xtrain.iloc[idx]

interaction_explainer=shap.TreeExplainer(model)

V=np.asarray(
    interaction_explainer.shap_interaction_values(Xi)
)

if V.ndim==4 and V.shape[-1]==1:
    V=V[...,0]

rows=[]

for i in range(len(features)):
    for j in range(i+1,len(features)):

        rows.append({
            "Feature_1":features[i],
            "Feature_2":features[j],
            "Mean_Abs_SHAP_Interaction":
                np.abs(V[:,i,j]).mean()
        })

interactions=pd.DataFrame(rows).sort_values(
    "Mean_Abs_SHAP_Interaction",
    ascending=False
).reset_index(drop=True)

interactions.insert(
    0,
    "Rank",
    range(1,len(interactions)+1)
)

interactions.to_csv(
    OUT/"SHAP_Interactions_SEL.csv",
    index=False
)

print("SHAP analysis completed:",OUT)
