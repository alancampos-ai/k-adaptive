import re, ast 
from typing import List, Tuple, Optional, Dict
import pandas as pd
import numpy as np

def parse_list_int(s: str) -> List[int]:
    return [int(x.strip()) for x in s.split(",") if x.strip()]

def parse_list_float(s: str) -> List[float]:
    return [float(x.strip()) for x in s.split(",") if x.strip()]

def _case_map(cols: List[str]) -> Dict[str,str]:
    return {c.lower(): c for c in cols}

def _resolve_name(df_tr: pd.DataFrame, df_va: pd.DataFrame, name: Optional[str], candidates: List[str]) -> Optional[str]:
    tmap = _case_map(list(df_tr.columns)); vmap = _case_map(list(df_va.columns))
    if name:
        n = name.lower()
        if n in tmap and n in vmap: return tmap[n]
        tr_hits = [tmap[c] for c in tmap if n == c or n in c]
        va_hits = [vmap[c] for c in vmap if n == c or n in c]
        inter = set([h.lower() for h in tr_hits]) & set([h.lower() for h in va_hits])
        if len(inter)==1:
            key = list(inter)[0]; return tmap.get(key, vmap.get(key))
    for cand in candidates:
        c = cand.lower()
        if c in tmap and c in vmap: return tmap[c]
    for cand in candidates:
        c = cand.lower()
        tr_hits = [tmap[x] for x in tmap if c in x]
        va_hits = [vmap[x] for x in vmap if c in x]
        inter = set([h.lower() for h in tr_hits]) & set([h.lower() for h in va_hits])
        if len(inter)==1:
            key = list(inter)[0]; return tmap.get(key, vmap.get(key))
    return None

def read_csv_auto(path: str) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python")

def _force_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    out = {}
    for c in cols:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            out[c] = s.astype(float); continue
        s2 = s.astype(str)
        s2 = s2.str.replace(",", ".", regex=False)
        s2 = s2.str.replace(r"[^\d\+\-eE\.]", " ", regex=True)
        out[c] = pd.to_numeric(s2, errors="coerce")
    return pd.DataFrame(out)

def _expand_vector_col_to_matrix(s: pd.Series) -> np.ndarray:
    txt = s.astype(str).str.replace(",", ".", regex=False).tolist()
    lists = []
    for v in txt:
        try:
            if v.strip().startswith("[") or v.strip().startswith("("):
                arr = ast.literal_eval(v)
                if isinstance(arr, (list, tuple)):
                    lists.append([str(x) for x in arr]); continue
        except Exception:
            pass
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", v)
        lists.append(nums if nums else [])
    L = max((len(x) for x in lists), default=0)
    if L == 0: return np.empty((len(lists), 0))
    arr = np.vstack([pd.to_numeric(x + (["nan"]*(L-len(x))), errors="coerce") for x in lists])
    return arr

def _expand_vector_pair(df_tr: pd.DataFrame, df_va: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cand_tr = [c for c in df_tr.columns if c not in exclude]
    cand_va = [c for c in df_va.columns if c not in exclude]
    common = [c for c in cand_tr if c in cand_va]
    for c in common:
        At = _expand_vector_col_to_matrix(df_tr[c])
        Av = _expand_vector_col_to_matrix(df_va[c])
        L = max(At.shape[1], Av.shape[1])
        if L >= 3:
            if At.shape[1] != L:
                At = np.hstack([At, np.full((At.shape[0], L-At.shape[1]), np.nan)])
            if Av.shape[1] != L:
                Av = np.hstack([Av, np.full((Av.shape[0], L-Av.shape[1]), np.nan)])
            cols = [f"{c}[{i}]" for i in range(L)]
            for i,name in enumerate(cols):
                df_tr[name] = At[:,i]
                df_va[name] = Av[:,i]
            del df_tr[c]; del df_va[c]
    new_exclude = [c for c in exclude if c in df_tr.columns and c in df_va.columns]
    return df_tr, df_va, new_exclude

def _expand_any_numeric_pair(df_tr: pd.DataFrame, df_va: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    cand_tr = [c for c in df_tr.columns if c not in exclude]
    cand_va = [c for c in df_va.columns if c not in exclude]
    common = [c for c in cand_tr if c in cand_va]
    any_added = False
    for c in common:
        st = df_tr[c].astype(str).str.replace(",", ".", regex=False)
        sv = df_va[c].astype(str).str.replace(",", ".", regex=False)
        s2t = st.str.replace(r"[^\d\+\-eE\.]", " ", regex=True)
        s2v = sv.str.replace(r"[^\d\+\-eE\.]", " ", regex=True)
        nt = pd.to_numeric(s2t, errors="coerce")
        nv = pd.to_numeric(s2v, errors="coerce")
        if nt.notna().sum() > 0 and nv.notna().sum() > 0:
            df_tr[c] = nt; df_va[c] = nv; any_added = True
    new_exclude = [c for c in exclude if c in df_tr.columns and c in df_va.columns]
    if any_added: return df_tr, df_va, new_exclude
    return df_tr, df_va, exclude

def _fallback_concat_parse_pair(df_tr: pd.DataFrame, df_va: pd.DataFrame, exclude: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    xt = df_tr[[c for c in df_tr.columns if c not in exclude]].astype(str).agg(" ".join, axis=1)
    xv = df_va[[c for c in df_va.columns if c not in exclude]].astype(str).agg(" ".join, axis=1)
    At = _expand_vector_col_to_matrix(xt)
    Av = _expand_vector_col_to_matrix(xv)
    L = max(At.shape[1], Av.shape[1])
    if L == 0: return pd.DataFrame(index=df_tr.index), pd.DataFrame(index=df_va.index)
    if At.shape[1] != L: At = np.hstack([At, np.full((At.shape[0], L-At.shape[1]), np.nan)])
    if Av.shape[1] != L: Av = np.hstack([Av, np.full((Av.shape[0], L-Av.shape[1]), np.nan)])
    cols = [f"f[{i}]" for i in range(L)]
    return pd.DataFrame(At, columns=cols, index=df_tr.index), pd.DataFrame(Av, columns=cols, index=df_va.index)

def coerce_numeric_pair(df_tr: pd.DataFrame, df_va: pd.DataFrame, label_col: Optional[str], subject_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, Optional[str], Optional[str]]:
    lc = _resolve_name(df_tr, df_va, label_col, ["label","y","class","target","outcome","response"])
    sc = _resolve_name(df_tr, df_va, subject_col, ["subject","subj","subject_id","sid","pid","patient","user","id","record"])
    exclude = [x for x in [lc, sc] if x is not None]
    df_tr2, df_va2, exclude2 = _expand_vector_pair(df_tr.copy(), df_va.copy(), exclude)
    df_tr3, df_va3, exclude3 = _expand_any_numeric_pair(df_tr2, df_va2, exclude2)
    cols_tr = [c for c in df_tr3.columns if c not in exclude3]
    cols_va = [c for c in df_va3.columns if c not in exclude3]
    common = [c for c in cols_tr if c in cols_va]
    num_tr = _force_numeric(df_tr3, common)
    num_va = _force_numeric(df_va3, common)
    chk1 = num_tr.dropna(axis=1, how="all").columns.tolist()
    chk2 = num_va.dropna(axis=1, how="all").columns.tolist()
    common_num = [c for c in chk1 if c in chk2]
    if not common_num:
        num_tr, num_va = _fallback_concat_parse_pair(df_tr.copy(), df_va.copy(), exclude)
        chk1 = num_tr.dropna(axis=1, how="all").columns.tolist()
        chk2 = num_va.dropna(axis=1, how="all").columns.tolist()
        common_num = [c for c in chk1 if c in chk2]
        if not common_num: raise ValueError("no numeric feature columns found")
    return num_tr[common_num], num_va[common_num], lc, sc

def load_pair_csvs(train_csv: str, val_csv: str, label_col: Optional[str], subject_col: Optional[str]):
    df_tr = read_csv_auto(train_csv); df_va = read_csv_auto(val_csv)
    X_tr, X_va, lc, sc = coerce_numeric_pair(df_tr, df_va, label_col, subject_col)
    y_tr = df_tr[lc] if lc else None
    y_va = df_va[lc] if lc else None
    subj_tr = df_tr[sc] if sc else None
    subj_va = df_va[sc] if sc else None
    return X_tr, X_va, y_tr, y_va, subj_tr, subj_va
