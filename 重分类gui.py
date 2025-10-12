# -*- coding: utf-8 -*-
"""
分类人工调整 Web GUI（持久化 Session + 服务端保存 Excel + 内存优化）
运行：
  pip install fastapi uvicorn[standard] pandas openpyxl requests python-multipart scikit-learn sentence-transformers python-dotenv
  python app.py
  打开 http://127.0.0.1:8000
"""
import io, os, gc, re, json, uuid, time, math, traceback
from datetime import datetime
from threading import Lock
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from fastapi import FastAPI, UploadFile, File, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA

try:
    import hdbscan  # type: ignore
except Exception:  # pragma: no cover - 可选依赖
    hdbscan = None
from dotenv import load_dotenv, set_key
import uvicorn

# ============== 可选：sentence-transformers ==============
_ST_MODEL = None
_ST_MODEL_NAME = None
_ST_MODEL_FALLBACKS = [
    "text2vec-base-multilingual",
    "paraphrase-multilingual-MiniLM-L12-v2",
    "uer/sbert-base-chinese-nli",
    "all-MiniLM-L6-v2",
]
try:
    from sentence_transformers import SentenceTransformer
except Exception:
    SentenceTransformer = None

load_dotenv(dotenv_path=".env")

# ================== 常量与目录 ==================
APP_VERSION = "4.0.0"
SESS_DIR    = os.getenv("SESS_DIR", "sessions")
OUTPUT_DIR  = os.getenv("OUTPUT_DIR", "Output")
STATIC_DIR  = "static"

# 限制常驻内存中的 DataFrame 会话数量，LRU 超限自动卸载（磁盘已持久化，不丢）
MAX_MEMORY_SESSIONS = int(os.getenv("MAX_MEMORY_SESSIONS", "3"))

os.makedirs(SESS_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)

# ============== 全局内存（LRU + 锁） ==============
DATASTORE: Dict[str, pd.DataFrame] = {}
PROGRESS: Dict[str, Dict[str, Any]] = {}
PROGRESS_LOCK = Lock()

SESSION_LOCKS: Dict[str, Lock] = {}
SESSION_TOUCH: Dict[str, float] = {}  # 最近访问时间戳（LRU）
SESSION_LOCKS_LOCK = Lock()

SCATTER_CACHE: Dict[str, Dict[str, Any]] = {}

def _get_lock(sid: str) -> Lock:
    with SESSION_LOCKS_LOCK:
        if sid not in SESSION_LOCKS:
            SESSION_LOCKS[sid] = Lock()
        return SESSION_LOCKS[sid]

def _touch(sid: str):
    SESSION_TOUCH[sid] = time.time()

def _evict_if_needed():
    """LRU：将超过 MAX_MEMORY_SESSIONS 的最久未用会话从内存卸载（磁盘已有持久化）。"""
    try:
        if len(DATASTORE) <= MAX_MEMORY_SESSIONS:
            return
        # 依据最新访问时间排序，保留最近的
        alive = sorted(SESSION_TOUCH.items(), key=lambda kv: kv[1], reverse=True)
        keep = set([sid for sid, _ in alive[:MAX_MEMORY_SESSIONS]])
        remove = [sid for sid in DATASTORE.keys() if sid not in keep]
        for sid in remove:
            # 不删除磁盘文件，只释放内存
            try:
                del DATASTORE[sid]
            except Exception:
                pass
        gc.collect()
    except Exception:
        pass

# ================== 参考列表 ==================
TOPIC_LIST = [
    "健康议题","经济议题","政治议题","环境议题","传播模式与行为",
    "媒介制度与平台治理","科技议题","文化议题","宗教议题","其他议题"
]
FIELD_LIST = [
    "政治传播","健康传播","大众传播","新闻学","跨文化传播","人际传播","科学传播",
    "法律与政策","公共关系","组织传播","媒介效果","传播心理学","传播伦理",
    "传播理论","广告学","传播研究方法","媒介史","媒介技术","言语传播",
    "教育传播","计算传播","网络与新媒体","其他领域"
]
REQUIRED_COLS = [
    "Article Title","Abstract","结构化总结",
    "研究主题（议题）分类","研究领域分类",
    "Publication Year","DOI"
]
ADJUST_TOPIC_COL = "研究主题（议题）分类_调整"
ADJUST_FIELD_COL = "研究领域分类_调整"
RANK_SCORE_COL   = "智能排序分数"
RANK_GROUP_COL   = "智能排序分组"
RANK_OUTLIER_COL = "智能排序_离群"
RANK_ALGO_COL    = "智能排序算法"

# ================== FastAPI 应用 ==================
app = FastAPI(title="分类人工调整 GUI", version=APP_VERSION)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

@app.get("/", response_class=FileResponse)
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

# ================== 进度条 ==================
def _progress_init(sid:str):
    with PROGRESS_LOCK:
        PROGRESS[sid] = {"status":"idle","message":"","percent":0,"emb_current":0,"emb_total":0,"groups_done":0,"groups_total":0,"outliers":0}

def _progress_update(sid:str, **kw):
    with PROGRESS_LOCK:
        st = PROGRESS.setdefault(sid, {})
        st.update(kw)
        emb_total = st.get("emb_total",0) or 0
        emb_cur   = st.get("emb_current",0) or 0
        g_total   = st.get("groups_total",0) or 0
        g_done    = st.get("groups_done",0) or 0
        emb_prog  = (emb_cur/emb_total) if emb_total else 0
        grp_prog  = (g_done/g_total) if g_total else 0
        st["percent"] = int(min(100, round(emb_prog*70 + grp_prog*30)))
        PROGRESS[sid] = st

def _progress_finish(sid:str, ok=True, msg=""):
    with PROGRESS_LOCK:
        st = PROGRESS.setdefault(sid,{})
        st["status"] = "done" if ok else "error"
        st["message"]= msg
        if ok: st["percent"]=100
        PROGRESS[sid] = st

@app.get("/progress")
def progress(session_id: str = Query(...)):
    with PROGRESS_LOCK:
        st = PROGRESS.get(session_id, {"status":"idle","percent":0,"message":""})
    return JSONResponse(st)

# ================== 工具函数 ==================
def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    for c in REQUIRED_COLS:
        if c not in df.columns: df[c] = ""
    if ADJUST_TOPIC_COL not in df.columns: df[ADJUST_TOPIC_COL] = ""
    if ADJUST_FIELD_COL not in df.columns: df[ADJUST_FIELD_COL] = ""
    for c in [RANK_SCORE_COL,RANK_GROUP_COL,RANK_OUTLIER_COL,RANK_ALGO_COL]:
        if c not in df.columns: df[c] = "" if c != RANK_OUTLIER_COL else False
    # 统一类型，避免写盘/读盘后类型漂移
    df[ADJUST_TOPIC_COL] = df[ADJUST_TOPIC_COL].astype(str).replace("nan","")
    df[ADJUST_FIELD_COL] = df[ADJUST_FIELD_COL].astype(str).replace("nan","")
    if RANK_OUTLIER_COL in df.columns:
        df[RANK_OUTLIER_COL] = df[RANK_OUTLIER_COL].astype(bool)
    return df

def read_table(file: UploadFile) -> pd.DataFrame:
    name = (file.filename or "").lower()
    bio = io.BytesIO(file.file.read())
    if name.endswith((".xlsx",".xls")):
        df = pd.read_excel(bio)
    elif name.endswith(".csv"):
        try: df = pd.read_csv(bio, encoding="utf-8-sig")
        except Exception:
            bio.seek(0); df = pd.read_csv(bio, encoding="gb18030")
    else:
        raise ValueError("仅支持 .xlsx/.xls/.csv")
    return normalize_columns(df)

def safe_get(row: pd.Series, col: str) -> str:
    v = row.get(col, "")
    return "" if pd.isna(v) else str(v)

def _counts(ser: pd.Series) -> List[Dict[str,Any]]:
    s = ser.fillna("").astype(str).str.strip()
    s = s[s!=""]
    total = int(s.shape[0]) if s.shape[0] else 1
    vc = s.value_counts()
    return [{"label":str(k),"count":int(v),"percent":round(v*100.0/total,2)} for k,v in vc.items()]


def _sanitize_blacklist(raw, allowed: List[str]) -> List[str]:
    """清理来自前端的黑名单，过滤非法值并保持顺序。"""
    if not raw:
        return []
    if isinstance(raw, str):
        candidates = [raw]
    else:
        try:
            candidates = list(raw)
        except TypeError:
            candidates = []
    seen = set()
    cleaned: List[str] = []
    for item in candidates:
        if not isinstance(item, str):
            continue
        name = item.strip()
        if not name or name not in allowed or name in seen:
            continue
        cleaned.append(name)
        seen.add(name)
    return cleaned

def slice_df(df: pd.DataFrame, page:int, page_size:int) -> pd.DataFrame:
    start = max(0,(page-1)*page_size); end = start + page_size
    return df.iloc[start:end].copy()

def _now_str():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def _session_dir(sid:str) -> str:
    d = os.path.join(SESS_DIR, sid)
    os.makedirs(d, exist_ok=True)
    return d

def _session_meta_path(sid:str) -> str:
    return os.path.join(_session_dir(sid), "meta.json")

def _session_data_path(sid:str) -> str:
    # 使用 pickle 避免额外依赖（pyarrow/feather）
    return os.path.join(_session_dir(sid), "data.pkl")

def _read_meta(sid:str) -> Dict[str,Any]:
    p = _session_meta_path(sid)
    if not os.path.exists(p): return {}
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def _write_meta(sid:str, meta:Dict[str,Any]):
    p = _session_meta_path(sid)
    try:
        with open(p, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print("[meta write failed]", e)

def save_session_to_disk(sid:str, origin_filename:str=None) -> Dict[str,Any]:
    """将当前会话的 DataFrame + 元信息写盘"""
    if sid not in DATASTORE:
        raise RuntimeError("会话不存在")
    df = DATASTORE[sid]
    dpath = _session_data_path(sid)
    df.to_pickle(dpath)
    meta = _read_meta(sid)
    meta.update({
        "session_id": sid,
        "rows": int(df.shape[0]),
        "updated_at": time.time(),
        "updated_text": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "origin_filename": origin_filename or meta.get("origin_filename",""),
        "app_version": APP_VERSION,
    })
    _write_meta(sid, meta)
    return meta

def load_session_from_disk(sid:str) -> pd.DataFrame:
    p = _session_data_path(sid)
    if not os.path.exists(p):
        raise RuntimeError("会话数据不存在或未保存")
    df = pd.read_pickle(p)
    return normalize_columns(df)

def list_sessions() -> List[Dict[str,Any]]:
    out=[]
    for sid in os.listdir(SESS_DIR):
        sp = os.path.join(SESS_DIR, sid)
        if not os.path.isdir(sp): continue
        meta = _read_meta(sid)
        if not meta:
            # 尝试补充 rows
            dpath = _session_data_path(sid)
            rows = 0
            if os.path.exists(dpath):
                try: rows = pd.read_pickle(dpath).shape[0]
                except: rows = 0
            meta={"session_id":sid,"rows":rows,"updated_at":0}
        out.append({
            "session_id": sid,
            "rows": meta.get("rows",0),
            "updated_text": meta.get("updated_text",""),
            "origin_filename": meta.get("origin_filename",""),
            "last_export_path": meta.get("last_export_path",""),
        })
    # 依据更新时间文本近似排序（无则靠文件ctime）
    def _key(m):
        txt = m.get("updated_text","")
        try:
            return datetime.strptime(txt, "%Y-%m-%d %H:%M:%S")
        except:
            return datetime.fromtimestamp(os.path.getctime(_session_dir(m["session_id"])))
    out.sort(key=_key, reverse=True)
    return out

def safe_filename(name:str) -> str:
    name = re.sub(r"[\\/:*?\"<>|]+", "_", name)
    name = name.strip()
    if not name: name = "export"
    return name

# ================== 前端配置 ==================
@app.get("/frontend_config")
def frontend_config():
    return JSONResponse({
        "topic_list": TOPIC_LIST,
        "field_list": FIELD_LIST,
        "adj_topic_key": ADJUST_TOPIC_COL,
        "adj_field_key": ADJUST_FIELD_COL,
        "app_version": APP_VERSION
    })

# ================== 上传 / 统计 / 分页 ==================
@app.post("/upload")
def upload(file: UploadFile = File(...)):
    try:
        df = read_table(file)
        sid = str(uuid.uuid4())
        with _get_lock(sid):
            DATASTORE[sid] = df
            _touch(sid)
            save_session_to_disk(sid, origin_filename=file.filename or "")
            _evict_if_needed()
        return JSONResponse({"session_id": sid, "total": int(df.shape[0])})
    except Exception as e:
        return PlainTextResponse(f"读取失败：{e}", status_code=400)

def _timeline(df: pd.DataFrame, bin_years: int) -> Dict[str, Any]:
    try:
        bin_size = int(bin_years)
    except Exception:
        bin_size = 5
    if bin_size < 1:
        bin_size = 1

    if "Publication Year" not in df.columns:
        return {
            "bin_size": bin_size,
            "total": 0,
            "min_year": None,
            "max_year": None,
            "bins": [],
            "stack_topic": {"series": []},
            "stack_field": {"series": []},
        }

    timeline_df = df[["Publication Year", ADJUST_TOPIC_COL, ADJUST_FIELD_COL]].copy()
    timeline_df["Publication Year"] = pd.to_numeric(timeline_df["Publication Year"], errors="coerce")
    timeline_df = timeline_df.dropna(subset=["Publication Year"])
    if timeline_df.empty:
        return {
            "bin_size": bin_size,
            "total": 0,
            "min_year": None,
            "max_year": None,
            "bins": [],
            "stack_topic": {"series": []},
            "stack_field": {"series": []},
        }

    timeline_df["Publication Year"] = timeline_df["Publication Year"].astype(int)
    years = timeline_df["Publication Year"]
    min_year = int(years.min())
    max_year = int(years.max())
    # 将区间对齐到 bin_size 的整数倍，便于展示
    start_year = int(math.floor(min_year / bin_size) * bin_size)
    end_year = int(math.ceil((max_year + 1) / bin_size) * bin_size - 1)

    labels: List[str] = []
    total = int(timeline_df.shape[0])

    # 预先计算每行所属的时间区间，便于后续统计
    def _assign_bin(year: int) -> Tuple[int, int]:
        offset = year - start_year
        if offset < 0:
            bucket_index = 0
        else:
            bucket_index = offset // bin_size
        bucket_start = start_year + bucket_index * bin_size
        bucket_end = bucket_start + bin_size - 1
        return bucket_start, bucket_end

    bin_starts: List[int] = []
    bin_ends: List[int] = []
    for start in range(start_year, end_year + 1, bin_size):
        end = start + bin_size - 1
        labels.append(f"{start}-{end}")
        bin_starts.append(start)
        bin_ends.append(end)

    # 将区间标签加入数据帧
    start_series = []
    end_series = []
    label_series = []
    for year in timeline_df["Publication Year"].tolist():
        s, e = _assign_bin(int(year))
        start_series.append(s)
        end_series.append(e)
        label_series.append(f"{s}-{e}")
    timeline_df = timeline_df.assign(
        bin_start=start_series,
        bin_end=end_series,
        bin_label=label_series,
    )

    count_by_bin = timeline_df.groupby("bin_label").size()
    bins: List[Dict[str, Any]] = []
    cumulative = 0
    for idx, label in enumerate(labels):
        start = bin_starts[idx]
        end = bin_ends[idx]
        count = int(count_by_bin.get(label, 0))
        cumulative += count
        percent = (count / total * 100.0) if total else 0.0
        cumulative_percent = (cumulative / total * 100.0) if total else 0.0
        bins.append({
            "label": label,
            "start": int(start),
            "end": int(end),
            "count": count,
            "percent": round(percent, 4),
            "cumulative_count": cumulative,
            "cumulative_percent": round(min(cumulative_percent, 100.0), 4),
        })

    def _build_stack(column: str, empty_label: str) -> Dict[str, Any]:
        cat_series = timeline_df[column].fillna("").astype(str).str.strip()
        cat_series = cat_series.replace("", empty_label)
        stack_df = timeline_df.assign(stack_category=cat_series)
        totals = stack_df.groupby("stack_category").size().sort_values(ascending=False)
        if totals.empty:
            return {"series": []}

        pivot = (
            stack_df.pivot_table(
                index="bin_label",
                columns="stack_category",
                values="Publication Year",
                aggfunc="count",
                fill_value=0,
            )
        )
        pivot = pivot.reindex(labels, fill_value=0)

        series: List[Dict[str, Any]] = []
        for cat, total_count in totals.items():
            if total_count <= 0:
                continue
            counts = pivot[cat].tolist() if cat in pivot.columns else [0] * len(labels)
            series.append({
                "label": str(cat),
                "counts": [int(v) for v in counts],
                "total": int(total_count),
            })

        return {
            "series": series,
            "category_total": int(totals.sum()),
            "unique_categories": int(totals.shape[0]),
        }

    return {
        "bin_size": bin_size,
        "total": total,
        "min_year": min_year,
        "max_year": max_year,
        "bins": bins,
        "stack_topic": _build_stack(ADJUST_TOPIC_COL, "未分类"),
        "stack_field": _build_stack(ADJUST_FIELD_COL, "未分类"),
    }


@app.get("/stats")
def stats(session_id: str = Query(...), bin_years: int = Query(5, ge=1, le=50)):
    if session_id not in DATASTORE:
        # 尝试从磁盘加载
        try:
            with _get_lock(session_id):
                DATASTORE[session_id] = load_session_from_disk(session_id)
                _touch(session_id); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", status_code=400)
    df = DATASTORE[session_id]
    return JSONResponse({
        "total": int(df.shape[0]),
        "topic_orig": _counts(df["研究主题（议题）分类"]),
        "field_orig": _counts(df["研究领域分类"]),
        "topic_adj": _counts(df[ADJUST_TOPIC_COL]),
        "field_adj": _counts(df[ADJUST_FIELD_COL]),
        "timeline": _timeline(df, bin_years),
    })

@app.get("/data")
def get_data(
    session_id: str = Query(...),
    page: int = Query(1, ge=1),
    page_size: int = Query(50, ge=1, le=5000),
    filter_target: Optional[str] = Query(None),
    filter_value: Optional[str] = Query(None),
    search_scope: Optional[str] = Query(None),
    search_keyword: Optional[str] = Query(None),
    sort_by: Optional[str] = Query(None),        # "rank"
    sort_order: Optional[str] = Query("asc"),    # "asc"|"desc"
    only_outliers: Optional[int] = Query(0)      # 1|0
):
    if session_id not in DATASTORE:
        try:
            with _get_lock(session_id):
                DATASTORE[session_id] = load_session_from_disk(session_id)
                _touch(session_id); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", status_code=400)
    df = DATASTORE[session_id]

    # 过滤
    dfv = df
    if filter_target and filter_value:
        if   filter_target=="topic_orig": m = df["研究主题（议题）分类"].astype(str) == filter_value
        elif filter_target=="field_orig": m = df["研究领域分类"].astype(str) == filter_value
        elif filter_target=="topic_adj":  m = df[ADJUST_TOPIC_COL].astype(str)   == filter_value
        elif filter_target=="field_adj":  m = df[ADJUST_FIELD_COL].astype(str)   == filter_value
        else: m = pd.Series([True]*df.shape[0], index=df.index)
        dfv = df[m]

    # 关键词搜索
    keyword = (search_keyword or "").strip()
    if keyword:
        scope = (search_scope or "title").lower()
        if scope == "summary":
            column = "结构化总结"
        else:
            column = "Article Title"
        if column in dfv.columns:
            series = dfv[column].fillna("").astype(str)
            mask = pd.Series(True, index=dfv.index)
            parts = [p for p in re.split(r"\s+", keyword) if p]
            if not parts:
                parts = [keyword]
            for part in parts:
                mask &= series.str.contains(part, case=False, na=False, regex=False)
            dfv = dfv[mask]

    # 只看离群
    if only_outliers and RANK_OUTLIER_COL in dfv.columns:
        dfv = dfv[dfv[RANK_OUTLIER_COL]==True]

    # 排序
    if sort_by=="rank" and RANK_SCORE_COL in dfv.columns:
        dfv = dfv.sort_values(by=[RANK_SCORE_COL], ascending=(str(sort_order).lower()!="desc"), na_position="last")

    total = int(dfv.shape[0])
    total_pages = max(1, math.ceil(total / page_size))
    page = max(1, min(page, total_pages))  # 夹逼

    sub = slice_df(dfv, page, page_size)
    cols = REQUIRED_COLS + [ADJUST_TOPIC_COL, ADJUST_FIELD_COL, RANK_SCORE_COL, RANK_OUTLIER_COL]
    rows = []
    for i, row in sub.iterrows():
        r = {"_index": int(i)}
        for c in cols: r[c] = safe_get(row, c)
        rows.append(r)

    _touch(session_id)
    return JSONResponse({"rows": rows,"page": page,"page_size": page_size,"total": total,"total_pages": total_pages})

# ================== 更新（只改一列）/ 批量 ==================
@app.post("/update")
def update_row(payload: Dict[str, Any]):
    try:
        sid = payload["session_id"]; idx = int(payload["index"])
        if sid not in DATASTORE:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        df = DATASTORE[sid]
        if idx<0 or idx>=df.shape[0]: return PlainTextResponse("行号越界", 400)

        # 结构化总结可独立保存
        if "structured" in payload:
            df.at[idx, "结构化总结"] = payload.get("structured","")

        which = (payload.get("which_adjust","") or "").strip()  # "topic"|"field"
        val   = payload.get("target_value","")

        # 向后兼容（避免同时改两列）
        if not which:
            t_old = payload.get("topic_adjust", None)
            f_old = payload.get("field_adjust", None)
            if (t_old is not None and str(t_old)!="") and (f_old is not None and str(f_old)!=""):
                return PlainTextResponse("一次只能修改一个调整列（topic 或 field）", 400)
            if t_old is not None and str(t_old)!="": which, val = "topic", str(t_old)
            elif f_old is not None and str(f_old)!="": which, val = "field", str(f_old)
            else:
                save_session_to_disk(sid)  # 即便只改了结构化总结也落盘
                _touch(sid)
                return JSONResponse({"ok": True})

        if which=="topic":
            if val not in TOPIC_LIST: return PlainTextResponse("目标值不在主题参考列表中", 400)
            df.at[idx, ADJUST_TOPIC_COL] = val
        elif which=="field":
            if val not in FIELD_LIST: return PlainTextResponse("目标值不在领域参考列表中", 400)
            df.at[idx, ADJUST_FIELD_COL] = val
        else:
            return PlainTextResponse("which_adjust 必须是 topic 或 field", 400)

        DATASTORE[sid] = df
        save_session_to_disk(sid)
        _touch(sid)
        return JSONResponse({"ok": True})
    except Exception as e:
        return PlainTextResponse(f"更新失败：{e}", 400)

@app.post("/bulk_update_indices")
def bulk_update_indices(payload: Dict[str, Any]):
    sid = payload.get("session_id"); idxs = payload.get("indices",[])
    which = payload.get("which_adjust"); val = payload.get("target_value","")
    if sid not in DATASTORE:
        try:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", 400)
    df = DATASTORE[sid]
    if which not in ("topic","field"): return PlainTextResponse("which_adjust 必须为 topic 或 field", 400)
    if which=="topic" and val not in TOPIC_LIST: return PlainTextResponse("目标值不在主题参考列表中", 400)
    if which=="field" and val not in FIELD_LIST: return PlainTextResponse("目标值不在领域参考列表中", 400)
    n=0
    for i in idxs:
        ii = int(i)
        if 0<=ii<df.shape[0]:
            if which=="topic": df.at[ii, ADJUST_TOPIC_COL] = val
            else: df.at[ii, ADJUST_FIELD_COL] = val
            n+=1
    DATASTORE[sid] = df
    save_session_to_disk(sid)
    _touch(sid)
    return JSONResponse({"ok": True, "updated": n})

@app.post("/bulk_update_filtered")
def bulk_update_filtered(payload: Dict[str, Any]):
    sid = payload.get("session_id"); which = payload.get("which_adjust"); val = payload.get("target_value")
    ft  = payload.get("filter_target"); fv  = payload.get("filter_value")
    if sid not in DATASTORE:
        try:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", 400)
    if which not in ("topic","field"): return PlainTextResponse("which_adjust 应为 topic 或 field", 400)
    if which=="topic" and val not in TOPIC_LIST: return PlainTextResponse("目标值不在主题参考列表中", 400)
    if which=="field" and val not in FIELD_LIST: return PlainTextResponse("目标值不在领域参考列表中", 400)
    df = DATASTORE[sid]
    if   ft=="topic_orig": m = df["研究主题（议题）分类"].astype(str)==fv
    elif ft=="field_orig": m = df["研究领域分类"].astype(str)==fv
    elif ft=="topic_adj":  m = df[ADJUST_TOPIC_COL].astype(str)==fv
    elif ft=="field_adj":  m = df[ADJUST_FIELD_COL].astype(str)==fv
    else: return PlainTextResponse("filter_target 参数不合法", 400)
    idxs = df.index[m].tolist()
    for i in idxs:
        if which=="topic": df.at[i, ADJUST_TOPIC_COL] = val
        else: df.at[i, ADJUST_FIELD_COL] = val
    DATASTORE[sid] = df
    save_session_to_disk(sid)
    _touch(sid)
    return JSONResponse({"ok": True, "updated": len(idxs)})

@app.post("/bulk_copy")
def bulk_copy(session_id: str = Query(...)):
    if session_id not in DATASTORE:
        try:
            with _get_lock(session_id):
                DATASTORE[session_id] = load_session_from_disk(session_id)
                _touch(session_id); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", 400)
    df = DATASTORE[session_id]
    mt = df[ADJUST_TOPIC_COL].astype(str).str.strip().replace("nan","")== ""
    mf = df[ADJUST_FIELD_COL].astype(str).str.strip().replace("nan","")== ""
    df.loc[mt, ADJUST_TOPIC_COL]  = df.loc[mt, "研究主题（议题）分类"].where(df["研究主题（议题）分类"].isin(TOPIC_LIST), "")
    df.loc[mf, ADJUST_FIELD_COL]  = df.loc[mf, "研究领域分类"].where(df["研究领域分类"].isin(FIELD_LIST), "")
    DATASTORE[session_id] = df
    save_session_to_disk(session_id)
    _touch(session_id)
    return JSONResponse({"ok": True})

# ================== 嵌入 / 排序 ==================
def _ensure_st_model(name: Optional[str]=None):
    global _ST_MODEL, _ST_MODEL_NAME
    if SentenceTransformer is None:
        raise RuntimeError("未安装 sentence-transformers；请改用 OpenAI Embeddings 或安装依赖。")
    if _ST_MODEL is None:
        candidates: List[str] = []
        if name:
            candidates.append(name)
        env_model = os.getenv("SENTENCE_MODEL", "").strip()
        if env_model:
            candidates.append(env_model)
        candidates.extend(_ST_MODEL_FALLBACKS)
        tried = set()
        last_err: Optional[Exception] = None
        for nm in candidates:
            if not nm or nm in tried:
                continue
            tried.add(nm)
            try:
                _ST_MODEL = SentenceTransformer(nm)
                _ST_MODEL_NAME = nm
                break
            except Exception as exc:
                last_err = exc
        if _ST_MODEL is None:
            raise RuntimeError(
                "无法加载可用的 SentenceTransformer 模型，请检查网络或手动设置 SENTENCE_MODEL 环境变量"
            ) from last_err
    return _ST_MODEL, _ST_MODEL_NAME

def embed_local(texts: List[str], model_name: Optional[str], cb=None) -> np.ndarray:
    model,_ = _ensure_st_model(model_name)
    bs = int(os.getenv("EMB_BATCH","64")); total=len(texts); out=[]
    for i in range(0,total,bs):
        v = model.encode(texts[i:i+bs], batch_size=bs, show_progress_bar=False, convert_to_numpy=True, normalize_embeddings=True)
        out.append(v)
        if cb: cb(i+len(v), total)
    return np.vstack(out) if out else np.zeros((0,384),dtype=np.float32)

def embed_openai(texts: List[str], base_url:str, api_key:str, model:str, cb=None) -> np.ndarray:
    url = base_url.rstrip("/") + "/v1/embeddings"
    headers={"Authorization":f"Bearer {api_key}","Content-Type":"application/json"}
    bs = int(os.getenv("OPENAI_EMB_BATCH","100")); total=len(texts); out=[]; done=0
    for i in range(0,total,bs):
        chunk = texts[i:i+bs]
        r = requests.post(url, headers=headers, json={"model":model,"input":chunk}, timeout=120); r.raise_for_status()
        data = r.json()["data"]
        vecs = np.array([np.array(d["embedding"],dtype=np.float32) for d in data])
        vecs = vecs / (np.linalg.norm(vecs,axis=1,keepdims=True)+1e-12)
        out.append(vecs); done += len(chunk)
        if cb: cb(done, total)
    return np.vstack(out) if out else np.zeros((0,1536),dtype=np.float32)

def build_texts(df: pd.DataFrame, fields: Dict[str,bool]) -> List[str]:
    parts = []
    if fields.get("title",True):    parts.append(df["Article Title"].fillna("").astype(str))
    if fields.get("abstract",True): parts.append(df["Abstract"].fillna("").astype(str))
    if fields.get("summary",True):  parts.append(df["结构化总结"].fillna("").astype(str))
    if not parts: parts=[df["Article Title"].fillna("").astype(str)]
    s = parts[0]
    for p in parts[1:]: s = s + " || " + p
    return s.fillna("").astype(str).tolist()

def compute_group_ranking(df: pd.DataFrame, group_by:str, fields:Dict[str,bool], algorithm:str, k:int,
                          src:str, base_url:str, api_key:str, emb_model:str,
                          local_model_name:str, sigma:float, min_cluster_size:int, sid:str) -> Tuple[int,int]:
    gcol = ADJUST_TOPIC_COL if group_by=="topic_adj" else "研究主题（议题）分类"
    mask = df[gcol].astype(str).str.strip()!=""; dfv = df[mask].copy()
    if dfv.empty: return (0,0)

    texts = build_texts(dfv, fields)
    def _emb_cb(done,total): _progress_update(sid,status="embedding",emb_current=done,emb_total=total, message=f"嵌入 {done}/{total}")
    _progress_update(sid,status="embedding",emb_current=0,emb_total=len(texts),groups_done=0,groups_total=0,outliers=0, message="准备嵌入")

    if src=="local":
        emb = embed_local(texts, local_model_name, _emb_cb)
    else:
        if not base_url or not api_key: raise RuntimeError("OpenAI Embeddings 未配置")
        emb = embed_openai(texts, base_url, api_key, emb_model, _emb_cb)

    labels = dfv[gcol].astype(str).tolist()
    idxs   = dfv.index.tolist()
    groups = sorted(set(labels))
    _progress_update(sid,status="grouping",groups_done=0,groups_total=len(groups), message=f"共 {len(groups)} 组")

    total_out = 0
    scores_all = np.zeros(len(idxs), dtype=float)
    outliers_all = np.zeros(len(idxs), dtype=bool)
    cluster_all = np.full(len(idxs), -1, dtype=int)

    for gi, lab in enumerate(groups, start=1):
        sel = [i for i,lb in enumerate(labels) if lb==lab]
        X = emb[sel,:]
        if X.shape[0]==1:
            d = np.array([0.0])
            labs = np.zeros(1, dtype=int)
            outs = np.array([False])
        else:
            if algorithm=="kmeans":
                kk = max(1, min(k, X.shape[0]))
                km = KMeans(n_clusters=kk, n_init=10, random_state=42)
                labs = km.fit_predict(X); centers = km.cluster_centers_
                sims = [float(cosine_similarity(X[i:i+1], centers[labs[i]].reshape(1,-1))[0][0]) for i in range(X.shape[0])]
                d = 1.0 - np.array(sims)
                outs = d > (np.mean(d) + sigma*(np.std(d)+1e-12))
            elif algorithm=="hdbscan":
                if hdbscan is None:
                    raise RuntimeError("未安装 hdbscan，请先 pip install hdbscan")
                mcs = max(2, min_cluster_size)
                clusterer = hdbscan.HDBSCAN(min_cluster_size=mcs, metric="euclidean")
                labs = clusterer.fit_predict(X)
                probs = getattr(clusterer, "probabilities_", np.ones(X.shape[0], dtype=float))
                d = 1.0 - np.array(probs, dtype=float)
                outs = labs == -1
                out_scores = getattr(clusterer, "outlier_scores_", None)
                if out_scores is not None:
                    d = np.maximum(d, np.array(out_scores, dtype=float))
            else:
                cen = X.mean(axis=0, keepdims=True)
                sims= cosine_similarity(X, cen).reshape(-1)
                d = 1.0 - sims
                labs = np.zeros(X.shape[0], dtype=int)
                outs = d > (np.mean(d) + sigma*(np.std(d)+1e-12))

        mu, sd = float(np.mean(d)), float(np.std(d)+1e-12)
        thr = mu + sigma*sd
        if algorithm != "hdbscan":
            outs = d > thr
        total_out += int(np.sum(outs))

        for j, li in enumerate(sel):
            ridx = idxs[li]
            df.at[ridx, RANK_SCORE_COL]   = float(d[j])
            df.at[ridx, RANK_GROUP_COL]   = lab
            df.at[ridx, RANK_OUTLIER_COL] = bool(outs[j])
            if algorithm=="kmeans":
                algo_desc = f"kmeans(k={k})"
            elif algorithm=="hdbscan":
                algo_desc = f"hdbscan(min_cluster_size={mcs})"
            else:
                algo_desc = "centroid"
            df.at[ridx, RANK_ALGO_COL]    = algo_desc
            scores_all[li] = float(d[j])
            outliers_all[li] = bool(outs[j])
            cluster_all[li] = int(labs[j]) if algorithm != "centroid" else 0

        _progress_update(sid,status="grouping",groups_done=gi, message=f"分组 {lab} 完成（{gi}/{len(groups)}）", outliers=total_out)

    try:
        dims = 2
        comps = min(dims, emb.shape[1] if emb.shape[1]>0 else dims, emb.shape[0] if emb.shape[0]>0 else dims)
        coords = np.zeros((emb.shape[0], dims), dtype=float)
        if emb.shape[0] > 0 and comps > 0:
            pca = PCA(n_components=comps)
            trans = pca.fit_transform(emb)
            coords[:, :comps] = trans
        SCATTER_CACHE[sid] = {
            "coords": coords.tolist(),
            "indices": idxs,
            "group_labels": labels,
            "algorithm": algorithm,
            "cluster_labels": cluster_all.tolist(),
            "scores": scores_all.tolist(),
            "outliers": outliers_all.tolist(),
            "group_by": group_by,
            "timestamp": _now_str(),
        }
    except Exception:
        SCATTER_CACHE.pop(sid, None)

    return (int(dfv.shape[0]), int(total_out))

@app.post("/compute_ranking")
def compute_rank(payload: Dict[str,Any]):
    sid = payload.get("session_id")
    try:
        if sid not in DATASTORE:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        df = DATASTORE[sid]
        _progress_init(sid); _progress_update(sid,status="embedding",message="开始计算…")
        SCATTER_CACHE.pop(sid, None)

        total, outs = compute_group_ranking(
            df=df,
            group_by=payload.get("group_by","topic_orig"),
            fields=payload.get("fields",{"title":True,"abstract":True,"summary":True}),
            algorithm=payload.get("algorithm","centroid"),
            k=int(payload.get("k",3)),
            src=payload.get("embedding_source","local"),
            base_url=payload.get("openai_base_url","") or os.getenv("OPENAI_BASE_URL",""),
            api_key =payload.get("openai_api_key","")  or os.getenv("OPENAI_API_KEY",""),
            emb_model=payload.get("openai_emb_model","") or os.getenv("OPENAI_EMBEDDINGS_MODEL","text-embedding-3-large"),
            local_model_name=os.getenv("SENTENCE_MODEL", None),
            sigma=float(payload.get("sigma",2.0)),
            min_cluster_size=int(payload.get("min_cluster_size",5)),
            sid=sid
        )
        DATASTORE[sid] = df
        save_session_to_disk(sid)
        _progress_finish(sid, True, f"完成：样本 {total}，离群 {outs}")
        _touch(sid)
        return JSONResponse({"ok": True, "total": total, "outliers": outs})
    except Exception as e:
        _progress_finish(sid, False, f"失败：{e}")
        return JSONResponse({"detail": f"排序失败：{e}"}, status_code=500)

@app.get("/ranking_scatter")
def ranking_scatter(session_id: str = Query(...)):
    data = SCATTER_CACHE.get(session_id)
    if not data:
        return JSONResponse({"detail": "当前会话暂无排序可视化数据"}, status_code=404)
    return JSONResponse({"ok": True, **data})

# ================== 会话持久化 API ==================
@app.post("/session/save")
def session_save(payload: Dict[str,Any]):
    sid = payload.get("session_id","")
    if not sid: return PlainTextResponse("缺少 session_id", 400)
    if sid not in DATASTORE:
        try:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", 400)
    meta = save_session_to_disk(sid)
    _touch(sid)
    return JSONResponse({"ok": True, "meta": meta})

@app.get("/session/list")
def session_list():
    return JSONResponse({"sessions": list_sessions()})

@app.get("/session/load")
def session_load(session_id: str = Query(...)):
    try:
        with _get_lock(session_id):
            DATASTORE[session_id] = load_session_from_disk(session_id)
            _touch(session_id); _evict_if_needed()
        df = DATASTORE[session_id]
        meta = _read_meta(session_id)
        return JSONResponse({"ok": True, "session_id": session_id, "total": int(df.shape[0]), "meta": meta})
    except Exception as e:
        return PlainTextResponse(f"加载失败：{e}", 400)

@app.get("/session/info")
def session_info(session_id: str = Query(...)):
    meta = _read_meta(session_id)
    if not meta: return PlainTextResponse("未找到会话信息", 404)
    return JSONResponse({"meta": meta})

# ================== 导出 Excel（服务端保存） ==================
def export_excel_to_server(sid:str, file_name:Optional[str]=None) -> str:
    if sid not in DATASTORE:
        with _get_lock(sid):
            DATASTORE[sid] = load_session_from_disk(sid)
            _touch(sid); _evict_if_needed()
    df = DATASTORE[sid]

    # 目标目录
    out_dir = os.path.join(OUTPUT_DIR, sid)
    os.makedirs(out_dir, exist_ok=True)

    # 文件名
    base = safe_filename(file_name or f"export_{_now_str()}")
    if not base.lower().endswith(".xlsx"):
        base += ".xlsx"
    path = os.path.join(out_dir, base)

    # 直接写盘（避免大 BytesIO 占内存）
    with pd.ExcelWriter(path, engine="openpyxl") as w:
        df.to_excel(w, index=False, sheet_name="Sheet1")
        ws = w.book["Sheet1"]
        from openpyxl.styles import PatternFill
        fill_topic = PatternFill("solid", fgColor="FFF3BF")
        fill_field = PatternFill("solid", fgColor="CDEAFE")
        fill_out   = PatternFill("solid", fgColor="FFDAD6")
        col_idx = {c:i+1 for i,c in enumerate(df.columns.tolist())}
        # 行上色（注意：这里是写盘阶段，不再把整本工作簿放在内存二次复制）
        for r, (_, row) in enumerate(df.iterrows(), start=2):
            if str(row.get(ADJUST_TOPIC_COL,"")) and row.get(ADJUST_TOPIC_COL)!=row.get("研究主题（议题）分类"):
                ws.cell(r, col_idx[ADJUST_TOPIC_COL]).fill = fill_topic
            if str(row.get(ADJUST_FIELD_COL,"")) and row.get(ADJUST_FIELD_COL)!=row.get("研究领域分类"):
                ws.cell(r, col_idx[ADJUST_FIELD_COL]).fill = fill_field
            if bool(row.get(RANK_OUTLIER_COL, False)) and "Article Title" in col_idx:
                ws.cell(r, col_idx["Article Title"]).fill = fill_out

    # 写入 meta
    meta = _read_meta(sid)
    meta["last_export_path"] = path.replace("\\","/")
    meta["last_export_time"] = _now_str()
    _write_meta(sid, meta)

    # 明确触发垃圾回收，降低长时交互内存峰值
    gc.collect()
    return path

@app.post("/export_excel_save")
def export_excel_save(payload: Dict[str,Any]):
    sid = payload.get("session_id","")
    fname = payload.get("file_name","") or None
    if not sid: return PlainTextResponse("缺少 session_id", 400)
    try:
        path = export_excel_to_server(sid, fname)
        # 返回相对路径以及一个可选下载 url
        rel = os.path.relpath(path, start=os.getcwd()).replace("\\","/")
        return JSONResponse({
            "ok": True,
            "saved_path": rel,
            "download_url": f"/file?path={rel}"
        })
    except Exception as e:
        return PlainTextResponse(f"导出失败：{e}", 500)

@app.get("/file")
def file_serve(path: str = Query(...)):
    """
    仅允许读取位于 OUTPUT_DIR 或 SESS_DIR 下的文件，防止任意路径下载。
    使用 FileResponse 流式传输，避免内存溢出。
    """
    # 规范化路径
    apath = os.path.abspath(path)
    base_out = os.path.abspath(OUTPUT_DIR)
    base_sess= os.path.abspath(SESS_DIR)
    if not (apath.startswith(base_out) or apath.startswith(base_sess)):
        return PlainTextResponse("路径不允许访问", 403)
    if not os.path.exists(apath):
        return PlainTextResponse("文件不存在", 404)
    filename = os.path.basename(apath)
    return FileResponse(apath, filename=filename, media_type="application/octet-stream")

# ================== LLM 配置 / 健康检查 ==================
@app.get("/default_llm_config")
def default_llm_config():
    return JSONResponse({
        "base_url": os.getenv("OPENAI_BASE_URL",""),
        "api_key":  os.getenv("OPENAI_API_KEY",""),
        "model":    os.getenv("OPENAI_MODEL",""),
        "temp":     0.2
    })

@app.post("/save_env")
def save_env(payload: Dict[str,Any]):
    base=payload.get("base_url","").strip()
    key =payload.get("api_key","").strip()
    model=payload.get("model","").strip()
    try:
        if base:  set_key(".env","OPENAI_BASE_URL",base)
        if key:   set_key(".env","OPENAI_API_KEY",key)
        if model: set_key(".env","OPENAI_MODEL",model)
        if not os.getenv("OPENAI_EMBEDDINGS_MODEL",""):
            set_key(".env","OPENAI_EMBEDDINGS_MODEL","text-embedding-3-large")
        return JSONResponse({"ok": True})
    except Exception as e:
        return PlainTextResponse(f".env 写入失败：{e}", 500)

@app.get("/healthz")
def healthz():
    return PlainTextResponse("ok")

# ================== 可选：智能建议占位（防止前端报错） ==================
def _format_topic_prompt(
    row: pd.Series,
    available_topics: List[str],
    retry_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    """生成用于请求议题分类的提示词。"""
    title = safe_get(row, "Article Title")
    abstract = safe_get(row, "Abstract")
    structured = safe_get(row, "结构化总结")
    content_lines = [
        "请你扮演中文学术分类助手。根据题目、摘要与结构化总结判断研究主题（议题）分类，并在需要时补全结构化总结。",
        "必须从候选列表中严格选择一个研究主题（议题），不要创造或输出列表外内容。",
        "输出 JSON，字段定义如下：",
        "{",
        "  \"topic_suggestion\": \"候选列表中的研究主题（议题）名称\",",
        "  \"structured_summary\": \"150字以内的中文结构化总结，如已有可适当润色后返回\"",
        "}",
        "研究主题（议题）候选列表：" + "、".join(available_topics),
    ]
    if retry_hint:
        content_lines.append(f"上一次回答未通过校验，原因：{retry_hint}。请务必从候选列表中选择正确的主题，并严格返回 JSON。")
    content_lines.extend(
        [
            "请根据以下文献内容完成任务：",
            f"标题：{title or '（无）'}",
            f"摘要：{abstract or '（无）'}",
            f"结构化总结：{structured or '（无）'}",
        ]
    )
    user_prompt = "\n".join(content_lines)
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns strict JSON responses for academic topic classification.",
        },
        {"role": "user", "content": user_prompt},
    ]


def _format_field_prompt(
    row: pd.Series,
    structured_summary: str,
    available_fields: List[str],
    retry_hint: Optional[str] = None,
) -> List[Dict[str, str]]:
    """生成用于请求领域分类的提示词。"""
    title = safe_get(row, "Article Title")
    abstract = safe_get(row, "Abstract")
    structured = structured_summary or safe_get(row, "结构化总结")
    content_lines = [
        "请你扮演中文学术分类助手。根据题目、摘要与结构化总结判断研究领域分类。",
        "必须从候选列表中严格选择一个研究领域，不要创造或输出列表外内容。",
        "输出 JSON，字段定义如下：",
        "{",
        "  \"field_suggestion\": \"候选列表中的研究领域名称\"",
        "}",
        "研究领域候选列表：" + "、".join(available_fields),
    ]
    if retry_hint:
        content_lines.append(f"上一次回答未通过校验，原因：{retry_hint}。请重新从候选列表中选择正确的领域，并严格返回 JSON。")
    content_lines.extend(
        [
            "请根据以下文献内容完成任务：",
            f"标题：{title or '（无）'}",
            f"摘要：{abstract or '（无）'}",
            f"结构化总结：{structured or '（无）'}",
        ]
    )
    user_prompt = "\n".join(content_lines)
    return [
        {
            "role": "system",
            "content": "You are a helpful assistant that returns strict JSON responses for academic field classification.",
        },
        {"role": "user", "content": user_prompt},
    ]


def _prepare_llm_config() -> Tuple[str, str, str]:
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
    model = (os.getenv("OPENAI_MODEL") or "").strip()
    if not (base_url and api_key and model):
        raise RuntimeError(
            "未配置 LLM 访问参数，请在环境变量或 .env 文件中设置 OPENAI_BASE_URL / OPENAI_API_KEY / OPENAI_MODEL。"
        )
    return base_url, api_key, model


def _invoke_llm(
    messages: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    try:
        resp = requests.post(url, headers=headers, data=json.dumps(payload).encode("utf-8"), timeout=60)
    except Exception as e:
        raise RuntimeError(f"请求 LLM 失败：{e}") from e
    if resp.status_code >= 400:
        raise RuntimeError(f"LLM 接口返回错误：{resp.status_code} {resp.text}")
    try:
        data = resp.json()
    except Exception as e:
        raise RuntimeError(f"解析 LLM 返回值失败：{e}") from e
    try:
        content = data["choices"][0]["message"]["content"].strip()
    except Exception as e:
        raise RuntimeError(f"LLM 返回格式不符合预期：{data}") from e
    return content


def _invoke_llm_json(
    messages: List[Dict[str, str]],
    base_url: str,
    api_key: str,
    model: str,
) -> Dict[str, Any]:
    content = _invoke_llm(messages, base_url, api_key, model)
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        raise ValueError(f"LLM 返回的内容不是合法 JSON：{content}") from e


def _llm_autosuggest(
    row: pd.Series,
    topic_blacklist: Optional[List[str]] = None,
    field_blacklist: Optional[List[str]] = None,
) -> Dict[str, Any]:
    base_url, api_key, model = _prepare_llm_config()

    topic_blacklist_clean = _sanitize_blacklist(topic_blacklist, TOPIC_LIST)
    field_blacklist_clean = _sanitize_blacklist(field_blacklist, FIELD_LIST)
    topic_blacklist_set = set(topic_blacklist_clean)
    field_blacklist_set = set(field_blacklist_clean)
    available_topics = [t for t in TOPIC_LIST if t not in topic_blacklist_set]
    available_fields = [f for f in FIELD_LIST if f not in field_blacklist_set]
    if not available_topics:
        raise RuntimeError("议题分类黑名单排除了所有候选项，请调整设置。")
    if not available_fields:
        raise RuntimeError("领域分类黑名单排除了所有候选项，请调整设置。")

    topic_retry_hint: Optional[str] = None
    topic_last_error: Optional[str] = None
    topic_suggestion: Optional[str] = None
    structured_summary: str = safe_get(row, "结构化总结")
    for _ in range(2):
        try:
            topic_resp = _invoke_llm_json(
                _format_topic_prompt(row, available_topics, topic_retry_hint),
                base_url,
                api_key,
                model,
            )
        except ValueError as ve:
            topic_last_error = str(ve)
            topic_retry_hint = str(ve)
            continue
        except RuntimeError:
            raise
        topic_candidate = str(topic_resp.get("topic_suggestion", "")).strip()
        summary_candidate = str(topic_resp.get("structured_summary", "")).strip()
        if summary_candidate:
            structured_summary = summary_candidate
        if topic_candidate in available_topics:
            topic_suggestion = topic_candidate
            break
        topic_last_error = f"LLM 议题分类不在候选列表中：{topic_candidate or '（空）'}"
        topic_retry_hint = f"返回的议题“{topic_candidate or '（空）'}”不在候选列表中。"
    if not topic_suggestion:
        raise RuntimeError(topic_last_error or "LLM 未能返回有效的议题分类")

    field_retry_hint: Optional[str] = None
    field_last_error: Optional[str] = None
    field_suggestion: Optional[str] = None
    for _ in range(2):
        try:
            field_resp = _invoke_llm_json(
                _format_field_prompt(
                    row,
                    structured_summary,
                    available_fields,
                    field_retry_hint,
                ),
                base_url,
                api_key,
                model,
            )
        except ValueError as ve:
            field_last_error = str(ve)
            field_retry_hint = str(ve)
            continue
        except RuntimeError:
            raise
        field_candidate = str(field_resp.get("field_suggestion", "")).strip()
        if field_candidate in available_fields:
            field_suggestion = field_candidate
            break
        field_last_error = f"LLM 领域分类不在候选列表中：{field_candidate or '（空）'}"
        field_retry_hint = f"返回的领域“{field_candidate or '（空）'}”不在候选列表中。"
    if not field_suggestion:
        raise RuntimeError(field_last_error or "LLM 未能返回有效的领域分类")

    return {
        "topic_suggestion": topic_suggestion,
        "field_suggestion": field_suggestion,
        "structured_summary": structured_summary or safe_get(row, "结构化总结"),
    }


@app.post("/autosuggest")
def autosuggest(payload: Dict[str,Any]):
    """调用外部 LLM，根据题目/摘要生成分类建议。"""
    sid = payload.get("session_id"); idx = int(payload.get("index", -1))
    if sid not in DATASTORE:
        try:
            with _get_lock(sid):
                DATASTORE[sid] = load_session_from_disk(sid)
                _touch(sid); _evict_if_needed()
        except Exception:
            return PlainTextResponse("无效 session_id", 400)
    df = DATASTORE[sid]
    if idx<0 or idx>=df.shape[0]: return PlainTextResponse("行号越界", 400)
    row = df.iloc[idx]

    topic_blacklist = payload.get("topic_blacklist")
    field_blacklist = payload.get("field_blacklist")

    try:
        result = _llm_autosuggest(row, topic_blacklist=topic_blacklist, field_blacklist=field_blacklist)
    except Exception as e:
        return PlainTextResponse(str(e), 500)

    return JSONResponse(result)

if __name__ == "__main__":
    # Windows 某些环境下 127.0.0.1 权限更稳妥
    uvicorn.run(app, host="127.0.0.1", port=8000)
