TARGET="SEL"
GROUP="filename"
SEED=42
OUTER=5
INNER=4
N_CANDIDATES=120
MIN_FEATURES=5
N_BOOTSTRAP=5000

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"data"/"selectivity"
OUT=ROOT/"outputs"/"selectivity"
MODEL_DIR=ROOT/"models"/"selectivity"

OUT.mkdir(parents=True,exist_ok=True)
MODEL_DIR.mkdir(parents=True,exist_ok=True)

XTR_FILE=DATA/"X_train.csv"
XTE_FILE=DATA/"X_test.csv"
YTR_FILE=DATA/"y_train.csv"
YTE_FILE=DATA/"y_test.csv"

def read(p):
    d=pd.read_csv(p)
    return d.loc[:,~d.columns.astype(str).str.startswith("Unnamed:")].reset_index(drop=True)

def load_target(p):
    d=read(p)
    c=d.select_dtypes(include=np.number).columns.tolist()
    if len(c)!=1:
        raise ValueError(f"{p} must contain exactly one numeric target column.")
    y=d[c[0]].astype(float).reset_index(drop=True)
    if not np.isfinite(y).all():
        raise ValueError(f"{p} contains non-finite target values.")
    return y,c[0]

def key(x):
    x=str(x).strip().lower().replace(" ","")
    for s in ("_log1p","_yeo"):
        if x.endswith(s):
            x=x[:-len(s)]
    return x

PROPERTY_COLUMNS={
    "mmol/g","c_v","s(g1)","hoa/kcal/mol","kh_mmolg_per_bar",
    "mmol/g_n2","c_v_n2","s(g1)_n2","hoa/kcal/mol_n2",
    "s(g1)_err","s(g1)_err_n2","f/bar","f/bar_n2",
    "p(g0)","p(g1)","p(g0)_n2","p(g1)_n2",
    "stdev","stdev.1","stdev.2","stdev.3","stdev.4",
    "stdev_n2","stdev.1_n2","stdev.2_n2","stdev.3_n2","stdev.4_n2"
}

def metrics(y,p):
    return {
        "R2":float(r2_score(y,p)),
        "MAE":float(mean_absolute_error(y,p)),
        "RMSE":float(np.sqrt(mean_squared_error(y,p)))
    }

def rfecv_model(seed):
    return ExtraTreesRegressor(
        n_estimators=400,
        criterion="squared_error",
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features=.80,
        max_leaf_nodes=60,
        bootstrap=True,
        max_samples=.85,
        ccp_alpha=1e-5,
        min_impurity_decrease=0.0,
        random_state=seed,
        n_jobs=1
    )

def make_model(params,seed):
    return ExtraTreesRegressor(random_state=seed,n_jobs=1,**params)

SPACE={
    "n_estimators":[500,700,900,1200],
    "criterion":["squared_error","friedman_mse"],
    "max_depth":[8,10,12,15,18,20],
    "min_samples_split":[4,6,8,10,12],
    "min_samples_leaf":[1,2,3,4,5],
    "max_features":[.50,.65,.80],
    "max_leaf_nodes":[30,40,50,60,80,100],
    "bootstrap":[True],
    "max_samples":[.70,.80,.85,.90],
    "ccp_alpha":[0.0,1e-6,1e-5,5e-5,1e-4],
    "min_impurity_decrease":[0.0,1e-5,5e-5]
}

CANDIDATES=list(ParameterSampler(
    SPACE,
    n_iter=N_CANDIDATES,
    random_state=SEED
))

def select_features(X,y,g,seed):
    cv=GroupKFold(n_splits=INNER)

    selector=RFECV(
        estimator=rfecv_model(seed),
        step=1,
        min_features_to_select=MIN_FEATURES,
        cv=cv,
        scoring="r2",
        n_jobs=-1
    )
    selector.fit(X,y,groups=g)

    r=selector.cv_results_
    mean=np.asarray(r["mean_test_score"],float)
    std=np.asarray(r["std_test_score"],float)

    n=np.asarray(
        r.get(
            "n_features",
            np.linspace(MIN_FEATURES,X.shape[1],len(mean),dtype=int)
        ),
        dtype=int
    )

    best=np.argmax(mean)
    threshold=mean[best]-std[best]/np.sqrt(INNER)
    n_selected=int(n[mean>=threshold].min())

    rfe=RFE(
        estimator=clone(rfecv_model(seed)),
        n_features_to_select=n_selected,
        step=1
    )
    rfe.fit(X,y)

    return X.columns[rfe.support_].tolist()

def evaluate(params,X,y,g,seed):
    cv=GroupKFold(n_splits=INNER)
    scores=[]

    for k,(a,b) in enumerate(cv.split(X,y,g),start=1):
        m=make_model(params,seed+k)
        m.fit(X.iloc[a],y.iloc[a])
        scores.append(r2_score(y.iloc[b],m.predict(X.iloc[b])))

    return float(np.mean(scores)),float(np.std(scores,ddof=0))

def complexity(p):
    return (
        p["max_depth"],
        p["max_leaf_nodes"],
        -p["min_samples_leaf"],
        -p["min_samples_split"],
        p["max_features"],
        p["max_samples"],
        -p["ccp_alpha"],
        p["n_estimators"]
    )

def tune(X,y,g,seed):
    results=[]

    for i,p in enumerate(CANDIDATES):
        mean,std=evaluate(p,X,y,g,seed+i*1000)
        results.append((p,mean,std))

    best=max(results,key=lambda z:z[1])
    threshold=best[1]-best[2]/np.sqrt(INNER)

    eligible=[z for z in results if z[1]>=threshold]

    chosen=sorted(
        eligible,
        key=lambda z:(complexity(z[0]),-z[1],z[2])
    )[0]

    return chosen[0]

def bootstrap_r2(y,p,g):
    y=np.asarray(y,float)
    p=np.asarray(p,float)
    g=np.asarray(g,str)

    groups=np.unique(g)
    rows={x:np.flatnonzero(g==x) for x in groups}
    rng=np.random.default_rng(SEED)
    values=[]

    for _ in range(N_BOOTSTRAP):
        sampled=rng.choice(groups,len(groups),replace=True)
        idx=np.concatenate([rows[x] for x in sampled])

        if np.unique(y[idx]).size>1:
            values.append(r2_score(y[idx],p[idx]))

    return np.asarray(values,float)

ytr,target_name=load_target(YTR_FILE)
yte,target_name_test=load_target(YTE_FILE)

if target_name!=target_name_test:
    raise ValueError("Train/test target names differ.")

tr=read(XTR_FILE)
te=read(XTE_FILE)

if GROUP not in tr.columns or GROUP not in te.columns:
    raise ValueError(f"{GROUP} must exist in both X files.")

if len(tr)!=len(ytr) or len(te)!=len(yte):
    raise ValueError("X/y row counts do not match.")

gtr=tr[GROUP].astype(str).to_numpy()
gte=te[GROUP].astype(str).to_numpy()

if set(gtr)&set(gte):
    raise ValueError("Train/test group leakage detected.")

properties=PROPERTY_COLUMNS|{key(target_name)}
common=[c for c in tr.columns if c in te.columns and c!=GROUP]
features=[c for c in common if key(c) not in properties]

Xtr=tr[features].select_dtypes(include=np.number).copy()
Xte=te[Xtr.columns].copy()

if Xtr.empty:
    raise ValueError("No numeric predictors remain.")

if not np.isfinite(Xtr.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values in training predictors.")

if not np.isfinite(Xte.to_numpy(dtype=float)).all():
    raise ValueError("Non-finite values in test predictors.")

outer=GroupKFold(n_splits=OUTER)
oof=np.full(len(ytr),np.nan)
fold_results=[]

for fold,(a,b) in enumerate(outer.split(Xtr,ytr,gtr),start=1):
    print(f"Outer fold {fold}/{OUTER}")

    Xa=Xtr.iloc[a].reset_index(drop=True)
    Xb=Xtr.iloc[b].reset_index(drop=True)
    ya=ytr.iloc[a].reset_index(drop=True)
    yb=ytr.iloc[b].reset_index(drop=True)
    ga=gtr[a]

    selected=select_features(Xa,ya,ga,SEED+fold*10000)
    params=tune(Xa[selected],ya,ga,SEED+fold*100000)

    m=make_model(params,SEED+fold)
    m.fit(Xa[selected],ya)

    pred=m.predict(Xb[selected])
    oof[b]=pred

    s=metrics(yb,pred)

    fold_results.append({
        "Fold":fold,
        "N_Features":len(selected),
        **s
    })

if np.isnan(oof).any():
    raise RuntimeError("Missing OOF predictions.")

outer_results=pd.DataFrame(fold_results)
oof_metrics=metrics(ytr,oof)

final_features=select_features(
    Xtr,ytr,gtr,
    SEED+900000
)

final_params=tune(
    Xtr[final_features],
    ytr,
    gtr,
    SEED+990000
)

final_model=make_model(final_params,SEED)
final_model.fit(Xtr[final_features],ytr)

pred_test=final_model.predict(Xte[final_features])
test_metrics=metrics(yte,pred_test)

boot=bootstrap_r2(yte,pred_test,gte)
ci_low,ci_high=np.quantile(boot,[.025,.975])

q1,q3=np.quantile(ytr,[.25,.75])
iqr=float(q3-q1)

if iqr<=0:
    raise ValueError("Training target IQR is zero.")

bundle={
    "model":final_model,
    "model_family":"ET",
    "target":TARGET,
    "target_column":target_name,
    "selected_features":final_features,
    "final_parameters":final_params,
    "group_column":GROUP
}

joblib.dump(
    bundle,
    MODEL_DIR/"Final_Model_SEL.joblib"
)

pd.concat([
    tr[[GROUP]].reset_index(drop=True),
    Xtr[final_features].reset_index(drop=True)
],axis=1).to_csv(
    DATA/"X_train_model.csv",
    index=False
)

pd.concat([
    te[[GROUP]].reset_index(drop=True),
    Xte[final_features].reset_index(drop=True)
],axis=1).to_csv(
    DATA/"X_test_model.csv",
    index=False
)

pd.DataFrame({
    "Selected_Feature":final_features
}).to_csv(
    OUT/"Selected_Features_SEL.csv",
    index=False
)

pd.DataFrame({
    GROUP:te[GROUP].astype(str),
    "Actual":yte,
    "Predicted":pred_test,
    "Residual":yte.to_numpy()-pred_test,
    "Absolute_Error":np.abs(yte.to_numpy()-pred_test)
}).to_csv(
    OUT/"Test_Predictions_SEL.csv",
    index=False
)

summary=pd.DataFrame([{
    "Target":TARGET,
    "Final_Features":len(final_features),
    "Outer_CV_R2_Mean":outer_results["R2"].mean(),
    "Outer_CV_R2_STD":outer_results["R2"].std(ddof=0),
    "OOF_R2":oof_metrics["R2"],
    "OOF_MAE":oof_metrics["MAE"],
    "OOF_RMSE":oof_metrics["RMSE"],
    "Test_R2":test_metrics["R2"],
    "Test_MAE":test_metrics["MAE"],
    "Test_RMSE":test_metrics["RMSE"],
    "Test_nMAE_IQR":test_metrics["MAE"]/iqr,
    "Test_nRMSE_IQR":test_metrics["RMSE"]/iqr,
    "Test_R2_CI_Low":ci_low,
    "Test_R2_CI_High":ci_high,
    "Final_Parameters":json.dumps(final_params,sort_keys=True)
}])

summary.to_csv(
    OUT/"Performance_SEL.csv",
    index=False
)

print("\nCompleted.")
print(summary.to_string(index=False))
