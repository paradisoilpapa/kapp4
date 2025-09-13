# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import unicodedata, re
import math, json, requests
from datetime import datetime, date, time, timedelta, timezone

# ==============================
# ページ設定
# ==============================
st.set_page_config(page_title="ヴェロビ：級別×日程ダイナミクス（5〜9車・買い目付き / 統合版）", layout="wide")

# ==============================
# ★ 新規パラメータ（偏差値＆推奨ロジック）
# ==============================
HEN_W_SB   = 0.20   # SB重み
HEN_W_PROF = 0.30   # 脚質重み
HEN_W_IN   = 0.50   # 入着重み（縮約3着内率）
HEN_DEC_PLACES = 1  # 偏差値 小数一桁

HEN_THRESHOLD = 55.0     # 偏差値クリア閾値
HEN_STRONG_ONE = 60.0    # 単独強者の目安

MAX_TICKETS = 6          # 買い目最大点数

# 推奨ラベル判定用（クリア台数→方針）
# k>=5:「2車複・ワイド」中心（広く） / k=3,4:「3連複」 / k=1,2:「状況次第（軸流し寄り）」 / k=0:ケン
LABEL_MAP = {
    "wide_qn": lambda k: k >= 5,
    "trio":    lambda k: 3 <= k <= 4,
    "axis":    lambda k: k in (1,2),
    "ken":     lambda k: k == 0,
}

# 期待値レンジ（内部基準で使用可。画面非表示）
P_FLOOR = {"sanpuku": 0.06, "nifuku": 0.12, "wide": 0.25, "nitan": 0.07, "santan": 0.03}
E_MIN, E_MAX = 0.10, 0.60

# ==============================
# 既存：風・会場・マスタ
# ==============================
WIND_COEFF = {
    "左上": -0.03, "上": -0.05, "右上": -0.035,
    "左": +0.05,  "右": -0.05,
    "左下": +0.035, "下": +0.05, "右下": +0.035,
    "無風": 0.0
}
WIND_MODE = "speed_only"
WIND_SIGN = -1
WIND_GAIN = 3.0
WIND_CAP  = 0.10
WIND_ZERO = 1.5
SPECIAL_DIRECTIONAL_VELODROMES = {"弥彦", "前橋"}

SESSION_HOUR = {"モーニング": 8, "デイ": 11, "ナイター": 18, "ミッドナイト": 22}
JST = timezone(timedelta(hours=9))

BASE_BY_KAKU = {"逃":1.58, "捲":1.65, "差":1.79, "マ":1.45}

KEIRIN_DATA = {
    "函館":{"bank_angle":30.6,"straight_length":51.3,"bank_length":400},
    "青森":{"bank_angle":32.3,"straight_length":58.9,"bank_length":400},
    "いわき平":{"bank_angle":32.9,"straight_length":62.7,"bank_length":400},
    "弥彦":{"bank_angle":32.4,"straight_length":63.1,"bank_length":400},
    "前橋":{"bank_angle":36.0,"straight_length":46.7,"bank_length":335},
    "取手":{"bank_angle":31.5,"straight_length":54.8,"bank_length":400},
    "宇都宮":{"bank_angle":25.8,"straight_length":63.3,"bank_length":500},
    "大宮":{"bank_angle":26.3,"straight_length":66.7,"bank_length":500},
    "西武園":{"bank_angle":29.4,"straight_length":47.6,"bank_length":400},
    "京王閣":{"bank_angle":32.2,"straight_length":51.5,"bank_length":400},
    "立川":{"bank_angle":31.2,"straight_length":58.0,"bank_length":400},
    "松戸":{"bank_angle":29.8,"straight_length":38.2,"bank_length":333},
    "川崎":{"bank_angle":32.2,"straight_length":58.0,"bank_length":400},
    "平塚":{"bank_angle":31.5,"straight_length":54.2,"bank_length":400},
    "小田原":{"bank_angle":35.6,"straight_length":36.1,"bank_length":333},
    "伊東":{"bank_angle":34.7,"straight_length":46.6,"bank_length":333},
    "静岡":{"bank_angle":30.7,"straight_length":56.4,"bank_length":400},
    "名古屋":{"bank_angle":34.0,"straight_length":58.8,"bank_length":400},
    "岐阜":{"bank_angle":32.3,"straight_length":59.3,"bank_length":400},
    "大垣":{"bank_angle":30.6,"straight_length":56.0,"bank_length":400},
    "豊橋":{"bank_angle":33.8,"straight_length":60.3,"bank_length":400},
    "富山":{"bank_angle":33.7,"straight_length":43.0,"bank_length":333},
    "松坂":{"bank_angle":34.4,"straight_length":61.5,"bank_length":400},
    "四日市":{"bank_angle":32.3,"straight_length":62.4,"bank_length":400},
    "福井":{"bank_angle":31.5,"straight_length":52.8,"bank_length":400},
    "奈良":{"bank_angle":33.4,"straight_length":38.0,"bank_length":333},
    "向日町":{"bank_angle":30.5,"straight_length":47.3,"bank_length":400},
    "和歌山":{"bank_angle":32.3,"straight_length":59.9,"bank_length":400},
    "岸和田":{"bank_angle":30.9,"straight_length":56.7,"bank_length":400},
    "玉野":{"bank_angle":30.6,"straight_length":47.9,"bank_length":400},
    "広島":{"bank_angle":30.8,"straight_length":57.9,"bank_length":400},
    "防府":{"bank_angle":34.7,"straight_length":42.5,"bank_length":333},
    "高松":{"bank_angle":33.3,"straight_length":54.8,"bank_length":400},
    "小松島":{"bank_angle":29.8,"straight_length":55.5,"bank_length":400},
    "高知":{"bank_angle":24.5,"straight_length":52.0,"bank_length":500},
    "松山":{"bank_angle":34.0,"straight_length":58.6,"bank_length":400},
    "小倉":{"bank_angle":34.0,"straight_length":56.9,"bank_length":400},
    "久留米":{"bank_angle":31.5,"straight_length":50.7,"bank_length":400},
    "武雄":{"bank_angle":32.0,"straight_length":64.4,"bank_length":400},
    "佐世保":{"bank_angle":31.5,"straight_length":40.2,"bank_length":400},
    "別府":{"bank_angle":33.7,"straight_length":59.9,"bank_length":400},
    "熊本":{"bank_angle":34.3,"straight_length":60.3,"bank_length":400},
    "手入力":{"bank_angle":30.0,"straight_length":52.0,"bank_length":400},
}
VELODROME_MASTER = {
    "函館":{"lat":41.77694,"lon":140.76283,"home_azimuth":None},
    "青森":{"lat":40.79717,"lon":140.66469,"home_azimuth":None},
    "いわき平":{"lat":37.04533,"lon":140.89150,"home_azimuth":None},
    "弥彦":{"lat":37.70778,"lon":138.82886,"home_azimuth":None},
    "前橋":{"lat":36.39728,"lon":139.05778,"home_azimuth":None},
    "取手":{"lat":35.90175,"lon":140.05631,"home_azimuth":None},
    "宇都宮":{"lat":36.57197,"lon":139.88281,"home_azimuth":None},
    "大宮":{"lat":35.91962,"lon":139.63417,"home_azimuth":None},
    "西武園":{"lat":35.76983,"lon":139.44686,"home_azimuth":None},
    "京王閣":{"lat":35.64294,"lon":139.53372,"home_azimuth":None},
    "立川":{"lat":35.70214,"lon":139.42300,"home_azimuth":None},
    "松戸":{"lat":35.80417,"lon":139.91119,"home_azimuth":None},
    "川崎":{"lat":35.52844,"lon":139.70944,"home_azimuth":None},
    "平塚":{"lat":35.32547,"lon":139.36342,"home_azimuth":None},
    "小田原":{"lat":35.25089,"lon":139.14947,"home_azimuth":None},
    "伊東":{"lat":34.954667,"lon":139.092639,"home_azimuth":None},
    "静岡":{"lat":34.973722,"lon":138.419417,"home_azimuth":None},
    "名古屋":{"lat":35.175560,"lon":136.854028,"home_azimuth":None},
    "岐阜":{"lat":35.414194,"lon":136.783917,"home_azimuth":None},
    "大垣":{"lat":35.361389,"lon":136.628444,"home_azimuth":None},
    "豊橋":{"lat":34.770167,"lon":137.417250,"home_azimuth":None},
    "富山":{"lat":36.757250,"lon":137.234833,"home_azimuth":None},
    "松坂":{"lat":34.564611,"lon":136.533833,"home_azimuth":None},
    "四日市":{"lat":34.965389,"lon":136.634500,"home_azimuth":None},
    "福井":{"lat":36.066889,"lon":136.253722,"home_azimuth":None},
    "奈良":{"lat":34.681111,"lon":135.823083,"home_azimuth":None},
    "向日町":{"lat":34.949222,"lon":135.708389,"home_azimuth":None},
    "和歌山":{"lat":34.228694,"lon":135.171833,"home_azimuth":None},
    "岸和田":{"lat":34.477500,"lon":135.369389,"home_azimuth":None},
    "玉野":{"lat":34.497333,"lon":133.961389,"home_azimuth":None},
    "広島":{"lat":34.359778,"lon":132.502889,"home_azimuth":None},
    "防府":{"lat":34.048778,"lon":131.568611,"home_azimuth":None},
    "高松":{"lat":34.345936,"lon":134.061994,"home_azimuth":None},
    "小松島":{"lat":34.005667,"lon":134.594556,"home_azimuth":None},
    "高知":{"lat":33.566694,"lon":133.526083,"home_azimuth":None},
    "松山":{"lat":33.808889,"lon":132.742333,"home_azimuth":None},
    "小倉":{"lat":33.885722,"lon":130.883167,"home_azimuth":None},
    "久留米":{"lat":33.316667,"lon":130.547778,"home_azimuth":None},
    "武雄":{"lat":33.194083,"lon":130.023083,"home_azimuth":None},
    "佐世保":{"lat":33.161667,"lon":129.712833,"home_azimuth":None},
    "別府":{"lat":33.282806,"lon":131.460472,"home_azimuth":None},
    "熊本":{"lat":32.789167,"lon":130.754722,"home_azimuth":None},
    "手入力":{"lat":None,"lon":None,"home_azimuth":None},
}

# --- 最新の印別実測率（表示はしないが内部保持）
RANK_STATS = {
    "◎": {"p1": 0.216, "pTop2": 0.456, "pTop3": 0.624},
    "〇": {"p1": 0.193, "pTop2": 0.360, "pTop3": 0.512},
    "▲": {"p1": 0.208, "pTop2": 0.384, "pTop3": 0.552},
    "△": {"p1": 0.152, "pTop2": 0.248, "pTop3": 0.384},
    "×": {"p1": 0.128, "pTop2": 0.256, "pTop3": 0.384},
    "α": {"p1": 0.088, "pTop2": 0.152, "pTop3": 0.312},
    "β": {"p1": 0.076, "pTop2": 0.151, "pTop3": 0.244},
}
RANK_FALLBACK_MARK = "△"
if RANK_FALLBACK_MARK not in RANK_STATS:
    RANK_FALLBACK_MARK = next(iter(RANK_STATS.keys()))
FALLBACK_DIST = RANK_STATS.get(RANK_FALLBACK_MARK, {"p1": 0.15, "pTop2": 0.30, "pTop3": 0.45})

# KO(勝ち上がり)関連
KO_GIRLS_SCALE = 0.0
KO_HEADCOUNT_SCALE = {5:0.6, 6:0.8, 7:1.0, 8:1.0, 9:1.0}
KO_GAP_DELTA = 0.010
KO_STEP_SIGMA = 0.4

# ◎ライン格上げ
LINE_BONUS_ON_TENKAI = {"優位"}
LINE_BONUS = {"second": 0.08, "thirdplus": 0.04}
LINE_BONUS_CAP = 0.10
PROB_U = {"second": 0.00, "thirdplus": 0.00}

# ==============================
# ユーティリティ
# ==============================
def clamp(x,a,b): return max(a, min(b, x))
def zscore_list(arr):
    arr = np.array(arr, dtype=float)
    m, s = float(np.mean(arr)), float(np.std(arr))
    return np.zeros_like(arr) if s==0 else (arr-m)/s
def zscore_val(x, xs):
    xs = np.array(xs, dtype=float); m, s = float(np.mean(xs)), float(np.std(xs))
    return 0.0 if s==0 else (float(x)-m)/s
def extract_car_list(s, nmax):
    s = str(s or "").strip()
    return [int(c) for c in s if c.isdigit() and 1 <= int(c) <= nmax]
def build_line_maps(lines):
    labels = "ABCDEFG"
    line_def = {labels[i]: lst for i,lst in enumerate(lines) if lst}
    car_to_group = {c:g for g,mem in line_def.items() for c in mem}
    return line_def, car_to_group
def role_in_line(car, line_def):
    for g, mem in line_def.items():
        if car in mem:
            if len(mem)==1: return 'single'
            idx = mem.index(car)
            return ['head','second','thirdplus'][idx] if idx<3 else 'thirdplus'
    return 'single'
def pos_coeff(role, line_factor):
    base = {'head':1.0,'second':0.7,'thirdplus':0.5,'single':0.9}.get(role,0.9)
    return base * line_factor
def tenscore_correction(tenscores):
    n = len(tenscores)
    if n<=2: return [0.0]*n
    df = pd.DataFrame({"得点":tenscores})
    df["順位"] = df["得点"].rank(ascending=False, method="min").astype(int)
    hi = min(n,8)
    baseline = df[df["順位"].between(2,hi)]["得点"].mean()
    def corr(row):
        return round(abs(baseline-row["得点"])*0.03, 3) if row["順位"] in [2,3,4] else 0.0
    return df.apply(corr, axis=1).tolist()

def wind_adjust(wind_dir, wind_speed, role, prof_escape):
    s = max(0.0, float(wind_speed))
    if s <= WIND_ZERO:
        base = 0.0
    elif s <= 5.0:
        base = 0.006 * (s - WIND_ZERO)
    elif s <= 8.0:
        base = 0.021 + 0.008 * (s - 5.0)
    else:
        base = 0.045 + 0.010 * min(s - 8.0, 4.0)
    pos = {'head':1.00,'second':0.85,'single':0.75,'thirdplus':0.65}.get(role, 0.75)
    prof = 0.35 + 0.65*float(prof_escape)
    val = base * pos * prof
    if (WIND_MODE == "directional") or (s >= 7.0 and st.session_state.get("track", "") in SPECIAL_DIRECTIONAL_VELODROMES):
        wd = WIND_COEFF.get(wind_dir, 0.0)
        dir_term = clamp(s * wd * (0.30 + 0.70*float(prof_escape)) * 0.6, -0.03, 0.03)
        val += dir_term
    val = (val * float(WIND_SIGN)) * float(WIND_GAIN)
    return round(clamp(val, -float(WIND_CAP), float(WIND_CAP)), 3)

def bank_character_bonus(bank_angle, straight_length, prof_escape, prof_sashi):
    straight_factor = (float(straight_length)-40.0)/10.0
    angle_factor = (float(bank_angle)-25.0)/5.0
    total = clamp(-0.1*straight_factor + 0.1*angle_factor, -0.05, 0.05)
    return round(total*prof_escape - 0.5*total*prof_sashi, 3)
def bank_length_adjust(bank_length, prof_oikomi):
    delta = clamp((float(bank_length)-411.0)/100.0, -0.05, 0.05)
    return round(delta*prof_oikomi, 3)

def compute_lineSB_bonus(line_def, S, B, line_factor=1.0, exclude=None, cap=0.06, enable=True):
    if not enable or not line_def:
        return {g:0.0 for g in line_def.keys()} if line_def else {}, {}
    w_pos_base = {'head':1.0,'second':0.4,'thirdplus':0.2,'single':0.7}
    Sg, Bg = {}, {}
    for g, mem in line_def.items():
        s=b=0.0
        for car in mem:
            if exclude is not None and car==exclude: continue
            w = w_pos_base[role_in_line(car, line_def)] * line_factor
            s += w*float(S.get(car,0)); b += w*float(B.get(car,0))
        Sg[g]=s; Bg[g]=b
    raw={}
    for g in line_def.keys():
        s, b = Sg[g], Bg[g]
        ratioS = s/(s+b+1e-6)
        raw[g] = (0.6*b + 0.4*s) * (0.6 + 0.4*ratioS)
    zz = zscore_list(list(raw.values())) if raw else []
    bonus={g: clamp(0.02*float(zz[i]), -cap, cap) for i,g in enumerate(raw.keys())}
    return bonus, raw

def input_float_text(label: str, key: str, placeholder: str = "") -> float | None:
    s = st.text_input(label, value=st.session_state.get(key, ""), key=key, placeholder=placeholder)
    ss = unicodedata.normalize("NFKC", str(s)).replace(",", "").strip()
    if ss == "": return None
    if not re.fullmatch(r"[+-]?\d+(\.\d+)?", ss):
        st.warning(f"{label} は数値で入力してください（入力値: {s}）")
        return None
    return float(ss)

# KO Utilities
def _role_of(car, mem):
    if len(mem)==1: return 'single'
    i = mem.index(car)
    return ['head','second','thirdplus'][i] if i<3 else 'thirdplus'
def _line_strength_raw(line_def, S, B, line_factor=1.0):
    if not line_def: return {}
    w_pos = {'head':1.0,'second':0.4,'thirdplus':0.2,'single':0.7}
    raw={}
    for g, mem in line_def.items():
        s=b=0.0
        for c in mem:
            w = w_pos[_role_of(c, mem)] * line_factor
            s += w*float(S.get(c,0)); b += w*float(B.get(c,0))
        ratioS = s/(s+b+1e-6)
        raw[g] = (0.6*b + 0.4*s) * (0.6 + 0.4*ratioS)
    return raw
def _top2_lines(line_def, S, B, line_factor=1.0):
    raw = _line_strength_raw(line_def, S, B, line_factor)
    order = sorted(raw.keys(), key=lambda g: raw[g], reverse=True)
    return (order[0], order[1]) if len(order)>=2 else (order[0], None) if order else (None, None)
def _extract_role_car(line_def, gid, role_name):
    if gid is None or gid not in line_def: return None
    mem = line_def[gid]
    if role_name=='head':    return mem[0] if len(mem)>=1 else None
    if role_name=='second':  return mem[1] if len(mem)>=2 else None
    return None
def _ko_order(v_base_map, line_def, S, B, line_factor=1.0, gap_delta=0.010):
    cars = list(v_base_map.keys())
    if not line_def or len(line_def)<1:
        return [c for c,_ in sorted(v_base_map.items(), key=lambda x:x[1], reverse=True)]
    g1, g2 = _top2_lines(line_def, S, B, line_factor)
    head1 = _extract_role_car(line_def, g1, 'head');  head2 = _extract_role_car(line_def, g2, 'head')
    sec1  = _extract_role_car(line_def, g1, 'second');sec2  = _extract_role_car(line_def, g2, 'second')
    others=[]
    if g1:
        mem = line_def[g1]
        if len(mem)>=3: others += mem[2:]
    if g2:
        mem = line_def[g2]
        if len(mem)>=3: others += mem[2:]
    for g, mem in line_def.items():
        if g not in {g1,g2}:
            others += mem
    order = []
    head_pair = [x for x in [head1, head2] if x is not None]
    order += sorted(head_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    sec_pair = [x for x in [sec1, sec2] if x is not None]
    order += sorted(sec_pair, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    others = list(dict.fromkeys([c for c in others if c is not None]))
    others_sorted = sorted(others, key=lambda c: v_base_map.get(c, -1e9), reverse=True)
    order += [c for c in others_sorted if c not in order]
    for c in cars:
        if c not in order:
            order.append(c)
    def _same_group(a,b):
        if a is None or b is None: return False
        ga = next((g for g,mem in line_def.items() if a in mem), None)
        gb = next((g for g,mem in line_def.items() if b in mem), None)
        return ga is not None and ga==gb
    i=0
    while i < len(order)-2:
        a, b, c = order[i], order[i+1], order[i+2]
        if _same_group(a, b):
            vx = v_base_map.get(b,0.0) - v_base_map.get(c,0.0)
            if vx >= -gap_delta:
                order.pop(i+2)
                order.insert(i+1, b)
        i += 1
    return order

def _zone_from_p(p: float):
    needed = 1.0 / max(p, 1e-12)
    return needed, needed*(1.0+E_MIN), needed*(1.0+E_MAX)

def apply_anchor_line_bonus(score_raw: dict[int,float], line_of: dict[int,int], role_map: dict[int,str], anchor: int, tenkai: str) -> dict[int,float]:
    a_line = line_of.get(anchor, None)
    is_on = (tenkai in LINE_BONUS_ON_TENKAI) and (a_line is not None)
    score_adj: dict[int,float] = {}
    for i, s in score_raw.items():
        bonus = 0.0
        if is_on and line_of.get(i) == a_line and i != anchor:
            role = role_map.get(i, "single")
            bonus = min(max(0.0, LINE_BONUS.get(role, 0.0)), LINE_BONUS_CAP)
        score_adj[i] = s + bonus
    return score_adj

def format_rank_all(score_map: dict[int,float], P_floor_val: float | None = None) -> str:
    order = sorted(score_map.keys(), key=lambda k: (-score_map[k], k))
    rows = []
    for i in order:
        if P_floor_val is None:
            rows.append(f"{i}")
        else:
            rows.append(f"{i}" if score_map[i] >= P_floor_val else f"{i}(P未満)")
    return " ".join(rows)

# ==============================
# 風の自動取得（Open-Meteo / 時刻固定）
# ==============================
def make_target_dt_naive(jst_date, race_slot: str):
    h = SESSION_HOUR.get(race_slot, 11)
    if isinstance(jst_date, datetime):
        jst_date = jst_date.date()
    try:
        y, m, d = jst_date.year, jst_date.month, jst_date.day
    except Exception:
        dt = pd.to_datetime(str(jst_date))
        y, m, d = dt.year, dt.month, dt.day
    return datetime(y, m, d, h, 0, 0)

def fetch_openmeteo_hour(lat, lon, target_dt_naive):
    import numpy as np
    d = target_dt_naive.strftime("%Y-%m-%d")
    base = "https://api.open-meteo.com/v1/forecast"
    urls = [
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m"
         "&timezone=Asia%2FTokyo"
         f"&start_date={d}&end_date={d}", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m"
         "&timezone=Asia%2FTokyo"
         f"&start_date={d}&end_date={d}", False),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m,wind_direction_10m"
         "&timezone=Asia%2FTokyo&past_days=2&forecast_days=2", True),
        (f"{base}?latitude={lat:.5f}&longitude={lon:.5f}"
         "&hourly=wind_speed_10m"
         "&timezone=Asia%2FTokyo&past_days=2&forecast_days=2", False),
    ]
    last_err = None
    for url, with_dir in urls:
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            j = r.json().get("hourly", {})
            times = [datetime.fromisoformat(t) for t in j.get("time", [])]
            if not times: raise RuntimeError("empty hourly times")
            diffs = [abs((t - target_dt_naive).total_seconds()) for t in times]
            k = int(np.argmin(diffs))
            sp = j.get("wind_speed_10m", [])
            di = j.get("wind_direction_10m", []) if with_dir else []
            speed = float(sp[k]) if k < len(sp) else float("nan")
            deg   = (float(di[k]) if with_dir and k < len(di) else None)
            return {"time": times[k], "speed_ms": speed, "deg": deg, "diff_min": diffs[k]/60.0}
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(f"Open-Meteo取得失敗（最後のエラー: {last_err}）")

# ==============================
# サイドバー：開催情報 / バンク・風・頭数
# ==============================
st.sidebar.header("開催情報 / バンク・風・頭数")
n_cars = st.sidebar.selectbox("出走数（5〜9）", [5,6,7,8,9], index=2)
track_names = list(KEIRIN_DATA.keys())
track = st.sidebar.selectbox("競輪場（プリセット）", track_names, index=track_names.index("川崎") if "川崎" in track_names else 0)
info = KEIRIN_DATA[track]
st.session_state["track"] = track

race_time = st.sidebar.selectbox("開催区分", ["モーニング","デイ","ナイター","ミッドナイト"], 1)
race_day = st.sidebar.date_input("開催日（風の取得基準日）", value=date.today())

wind_dir = st.sidebar.selectbox("風向", ["無風","左上","上","右上","左","右","左下","下","右下"], index=0, key="wind_dir_input")
wind_speed_default = st.session_state.get("wind_speed", 3.0)
wind_speed = st.sidebar.number_input("風速(m/s)", 0.0, 30.0, float(wind_speed_default), 0.1)

with st.sidebar.expander("🌀 風をAPIで自動取得（Open-Meteo）", expanded=False):
    api_date = st.date_input("開催日（風の取得基準日）", value=pd.to_datetime("today").date(), key="api_date")
    st.caption("基準時刻：モ=8時 / デ=11時 / ナ=18時 / ミ=22時（JST・tzなしで取得）")
    if st.button("APIで取得→風速に反映", use_container_width=True):
        info_xy = VELODROME_MASTER.get(track)
        if not info_xy or info_xy.get("lat") is None or info_xy.get("lon") is None:
            st.error(f"{track} の座標が未登録です（VELODROME_MASTER に lat/lon を入れてください）")
        else:
            try:
                target = make_target_dt_naive(api_date, race_time)
                data = fetch_openmeteo_hour(info_xy["lat"], info_xy["lon"], target)
                st.session_state["wind_speed"] = round(float(data["speed_ms"]), 2)
                st.success(f"{track} {target:%Y-%m-%d %H:%M} 風速 {st.session_state['wind_speed']:.1f} m/s （API側と{data['diff_min']:.0f}分ズレ）")
                st.rerun()
            except Exception as e:
                st.error(f"取得に失敗：{e}")

straight_length = st.sidebar.number_input("みなし直線(m)", 30.0, 80.0, float(info["straight_length"]), 0.1)
bank_angle = st.sidebar.number_input("バンク角(°)", 20.0, 45.0, float(info["bank_angle"]), 0.1)
bank_length = st.sidebar.number_input("周長(m)", 300.0, 500.0, float(info["bank_length"]), 0.1)

base_laps = st.sidebar.number_input("周回（通常4）", 1, 10, 4, 1)
day_label = st.sidebar.selectbox("開催日", ["初日","2日目","最終日"], 0)
eff_laps = int(base_laps) + {"初日":1,"2日目":2,"最終日":3}[day_label]

race_class = st.sidebar.selectbox("級別", ["Ｓ級","Ａ級","Ａ級チャレンジ","ガールズ"], 0)

angles = [KEIRIN_DATA[k]["bank_angle"] for k in KEIRIN_DATA]
straights = [KEIRIN_DATA[k]["straight_length"] for k in KEIRIN_DATA]
lengths = [KEIRIN_DATA[k]["bank_length"] for k in KEIRIN_DATA]
angle_z = zscore_val(bank_angle, angles)
straight_z = zscore_val(straight_length, straights)
length_z = zscore_val(bank_length, lengths)
style_raw = clamp(0.50*angle_z - 0.35*straight_z - 0.30*length_z, -1.0, +1.0)

override = st.sidebar.slider("会場バイアス補正（−2差し ←→ +2先行）", -2.0, 2.0, 0.0, 0.1)
style = clamp(style_raw + 0.25*override, -1.0, +1.0)

CLASS_FACTORS = {
    "Ｓ級":           {"spread":1.00, "line":1.00},
    "Ａ級":           {"spread":0.90, "line":0.85},
    "Ａ級チャレンジ": {"spread":0.80, "line":0.70},
    "ガールズ":       {"spread":0.85, "line":1.00},
}
cf = CLASS_FACTORS[race_class]

DAY_FACTOR = {"初日":1.00, "2日目":0.60, "最終日":0.85}
day_factor = DAY_FACTOR[day_label]

cap_base = clamp(0.06 + 0.02*style, 0.04, 0.08)
line_factor_eff = cf["line"] * day_factor
cap_SB_eff = cap_base * day_factor
if race_time == "ミッドナイト":
    line_factor_eff *= 0.95
    cap_SB_eff *= 0.95

line_sb_enable = (race_class != "ガールズ")

st.sidebar.caption(
    f"会場スタイル: {style:+.2f}（raw {style_raw:+.2f}） / "
    f"級別: spread={cf['spread']:.2f}, line={cf['line']:.2f} / "
    f"日程係数(line)={day_factor:.2f} → line係数={line_factor_eff:.2f}, SBcap±{cap_SB_eff:.2f}"
)

# ==============================
# メイン：入力
# ==============================
st.title("⭐ ヴェロビ（級別×日程ダイナミクス / 5〜9車・買い目付き：統合版）⭐")
st.caption(f"風補正モード: {WIND_MODE}（'speed_only'=風速のみ / 'directional'=向きも薄く考慮）")

st.subheader("レース番号（直前にサクッと変更）")
if "race_no_main" not in st.session_state:
    st.session_state["race_no_main"] = 1
c1, c2, c3 = st.columns([6,2,2])
with c1:
    race_no_input = st.number_input("R", min_value=1, max_value=12, step=1,
                                    value=int(st.session_state["race_no_main"]),
                                    key="race_no_input")
with c2:
    prev_clicked = st.button("◀ 前のR", use_container_width=True)
with c3:
    next_clicked = st.button("次のR ▶", use_container_width=True)
if prev_clicked:
    st.session_state["race_no_main"] = max(1, int(race_no_input) - 1); st.rerun()
elif next_clicked:
    st.session_state["race_no_main"] = min(12, int(race_no_input) + 1); st.rerun()
else:
    st.session_state["race_no_main"] = int(race_no_input)
race_no = int(st.session_state["race_no_main"])

st.subheader("ライン構成（最大7：単騎も1ライン）")
line_inputs = [
    st.text_input("ライン1（例：317）", key="line_1", max_chars=9),
    st.text_input("ライン2（例：6）", key="line_2", max_chars=9),
    st.text_input("ライン3（例：425）", key="line_3", max_chars=9),
    st.text_input("ライン4（任意）", key="line_4", max_chars=9),
    st.text_input("ライン5（任意）", key="line_5", max_chars=9),
    st.text_input("ライン6（任意）", key="line_6", max_chars=9),
    st.text_input("ライン7（任意）", key="line_7", max_chars=9),
]
n_cars = int(n_cars)
lines = [extract_car_list(x, n_cars) for x in line_inputs if str(x).strip()]
line_def, car_to_group = build_line_maps(lines)
active_cars = sorted({c for lst in lines for c in lst}) if lines else list(range(1, n_cars+1))

st.subheader("個人データ（直近4か月：回数）")
cols = st.columns(n_cars)
ratings, S, B = {}, {}, {}
k_esc, k_mak, k_sashi, k_mark = {}, {}, {}, {}
x1, x2, x3, x_out = {}, {}, {}, {}

for i, no in enumerate(active_cars):
    with cols[i]:
        st.markdown(f"**{no}番**")
        ratings[no] = input_float_text("得点（空欄可）", key=f"pt_{no}", placeholder="例: 55.0")
        S[no] = st.number_input("S", 0, 99, 0, key=f"s_{no}")
        B[no] = st.number_input("B", 0, 99, 0, key=f"b_{no}")
        k_esc[no]   = st.number_input("逃", 0, 99, 0, key=f"ke_{no}")
        k_mak[no]   = st.number_input("捲", 0, 99, 0, key=f"km_{no}")
        k_sashi[no] = st.number_input("差", 0, 99, 0, key=f"ks_{no}")
        k_mark[no]  = st.number_input("マ", 0, 99, 0, key=f"kk_{no}")
        x1[no]  = st.number_input("1着", 0, 99, 0, key=f"x1_{no}")
        x2[no]  = st.number_input("2着", 0, 99, 0, key=f"x2_{no}")
        x3[no]  = st.number_input("3着", 0, 99, 0, key=f"x3_{no}")
        x_out[no]= st.number_input("着外", 0, 99, 0, key=f"xo_{no}")

ratings_val = {no: (ratings[no] if ratings[no] is not None else 55.0) for no in active_cars}

# 1着・2着の縮約（級別×会場の事前分布を混ぜる）
def prior_by_class(cls, style_adj):
    if "ガール" in cls: p1,p2 = 0.18,0.24
    elif "Ｓ級" in cls: p1,p2 = 0.22,0.26
    elif "チャレンジ" in cls: p1,p2 = 0.18,0.22
    else: p1,p2 = 0.20,0.25
    p1 += 0.010*style_adj; p2 -= 0.005*style_adj
    return clamp(p1,0.05,0.60), clamp(p2,0.05,0.60)
def n0_by_n(n):
    if n<=6: return 12
    if n<=14: return 8
    if n<=29: return 5
    return 3

p1_eff, p2_eff = {}, {}
for no in active_cars:
    n = x1[no]+x2[no]+x3[no]+x_out[no]
    p1_prior, p2_prior = prior_by_class(race_class, style)
    n0 = n0_by_n(n)
    if n==0:
        p1_eff[no], p2_eff[no] = p1_prior, p2_prior
    else:
        p1_eff[no] = clamp((x1[no] + n0*p1_prior)/(n+n0), 0.0, 0.40)
        p2_eff[no] = clamp((x2[no] + n0*p2_prior)/(n+n0), 0.0, 0.50)

Form = {no: 0.7*p1_eff[no] + 0.3*p2_eff[no] for no in active_cars}

# 脚質プロフィール（会場適性）
prof_base, prof_escape, prof_sashi, prof_oikomi = {}, {}, {}, {}
for no in active_cars:
    tot = k_esc[no]+k_mak[no]+k_sashi[no]+k_mark[no]
    if tot==0: esc=mak=sashi=mark = 0.25
    else:
        esc=k_esc[no]/tot; mak=k_mak[no]/tot; sashi=k_sashi[no]/tot; mark=k_mark[no]/tot
    prof_escape[no]=esc; prof_sashi[no]=sashi; prof_oikomi[no]=mark
    base = esc*BASE_BY_KAKU["逃"] + mak*BASE_BY_KAKU["捲"] + sashi*BASE_BY_KAKU["差"] + mark*BASE_BY_KAKU["マ"]
    k = 0.06
    venue_bonus = k * style * ( +1.00*esc +0.40*mak -0.60*sashi -0.25*mark )
    prof_base[no] = base + clamp(venue_bonus, -0.06, +0.06)

# ======== 個人補正（得点/脚質上位/着順分布） ========
ratings_sorted = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank = {no: i+1 for i,no in enumerate(ratings_sorted)}
def tenscore_bonus(no):
    r = ratings_rank[no]
    top_n = min(3, len(active_cars))
    bottom_n = min(3, len(active_cars))
    if r <= top_n: return +0.03
    if r >= len(active_cars)-bottom_n+1: return -0.02
    return 0.0
def topk_bonus(k_dict, topn=3, val=0.02):
    order = sorted(k_dict.items(), key=lambda x:(x[1], -x[0]), reverse=True)
    grant = set([no for i,(no,v) in enumerate(order) if i<topn])
    return {no:(val if no in grant else 0.0) for no in k_dict}
esc_bonus   = topk_bonus(k_esc,   topn=3, val=0.02)
mak_bonus   = topk_bonus(k_mak,   topn=3, val=0.02)
sashi_bonus = topk_bonus(k_sashi, topn=3, val=0.015)
mark_bonus  = topk_bonus(k_mark,  topn=3, val=0.01)
def finish_bonus(no):
    tot = x1[no]+x2[no]+x3[no]+x_out[no]
    if tot == 0: return 0.0
    in3 = (x1[no]+x2[no]+x3[no]) / tot
    out = x_out[no] / tot
    bonus = 0.0
    if in3 > 0.50: bonus += 0.03
    if out > 0.70: bonus -= 0.03
    if out < 0.40: bonus += 0.02
    return bonus
extra_bonus = {}
for no in active_cars:
    total = (tenscore_bonus(no) +
             esc_bonus.get(no,0.0) + mak_bonus.get(no,0.0) +
             sashi_bonus.get(no,0.0) + mark_bonus.get(no,0.0) +
             finish_bonus(no))
    extra_bonus[no] = clamp(total, -0.10, +0.10)

# ===== SBなし合計（環境補正 + 得点微補正 + 個人補正 + 周回疲労） =====
tens_list = [ratings_val[no] for no in active_cars]
t_corr = tenscore_correction(tens_list) if active_cars else []
tens_corr = {no:t_corr[i] for i,no in enumerate(active_cars)} if active_cars else {}

rows = []
_wind_func = wind_adjust
eff_wind_dir   = globals().get("eff_wind_dir",   wind_dir)
eff_wind_speed = globals().get("eff_wind_speed", wind_speed)

for no in active_cars:
    role = role_in_line(no, line_def)
    wind = _wind_func(eff_wind_dir, float(eff_wind_speed or 0.0), role, float(prof_escape[no]))
    extra = max(eff_laps - 2, 0)
    fatigue_scale = (1.0 if race_class == "Ｓ級" else 1.1 if race_class == "Ａ級" else 1.2 if race_class == "Ａ級チャレンジ" else 1.05)
    laps_adj = (-0.10 * extra * (1.0 if prof_escape[no] > 0.5 else 0.0)
                + 0.05 * extra * (1.0 if prof_oikomi[no] > 0.4 else 0.0)) * fatigue_scale
    bank_b = bank_character_bonus(bank_angle, straight_length, prof_escape[no], prof_sashi[no])
    length_b = bank_length_adjust(bank_length, prof_oikomi[no])
    indiv = extra_bonus.get(no, 0.0)
    total_raw = (prof_base[no] + wind + cf["spread"] * tens_corr.get(no, 0.0)
                 + bank_b + length_b + laps_adj + indiv)
    rows.append([int(no), role, round(prof_base[no],3), round(wind,3),
                 round(cf["spread"] * tens_corr.get(no, 0.0),3),
                 round(bank_b,3), round(length_b,3), round(laps_adj,3),
                 round(indiv,3), total_raw])

df = pd.DataFrame(rows, columns=[
    "車番","役割","脚質基準(会場)","風補正","得点補正","バンク補正",
    "周長補正","周回補正","個人補正","合計_SBなし_raw",
])
mu = float(df["合計_SBなし_raw"].mean()) if not df.empty else 0.0
df["合計_SBなし"] = mu + 1.0 * (df["合計_SBなし_raw"] - mu)

# ===== KO方式：最終並びの反映（既存）
v_wo = {int(k): float(v) for k, v in zip(df["車番"].astype(int), df["合計_SBなし"].astype(float))}
_is_girls = (race_class == "ガールズ")
head_scale = KO_HEADCOUNT_SCALE.get(int(n_cars), 1.0)
ko_scale_raw = (KO_GIRLS_SCALE if _is_girls else 1.0) * head_scale
KO_SCALE_MAX = 0.45
ko_scale = min(ko_scale_raw, KO_SCALE_MAX)
if ko_scale > 0.0 and line_def and len(line_def) >= 1:
    ko_order = _ko_order(v_wo, line_def, S, B, line_factor=line_factor_eff, gap_delta=KO_GAP_DELTA)
    vals = [v_wo[c] for c in v_wo.keys()]
    mu0  = float(np.mean(vals)); sd0 = float(np.std(vals) + 1e-12)
    KO_STEP_SIGMA_LOCAL = max(0.25, KO_STEP_SIGMA * 0.7)
    step = KO_STEP_SIGMA_LOCAL * sd0
    new_scores = {}
    for rank, car in enumerate(ko_order, start=1):
        rank_adjust = step * (len(ko_order) - rank)
        blended = (1.0 - ko_scale) * v_wo[car] + ko_scale * (mu0 + rank_adjust - (len(ko_order)/2.0 - 0.5)*step)
        new_scores[car] = blended
    v_final = {int(k): float(v) for k, v in new_scores.items()}
else:
    v_final = {int(k): float(v) for k, v in v_wo.items()}

# --- 純SBなしランキング（KOまで／格上げ前）
df_sorted_pure = pd.DataFrame({
    "車番": list(v_final.keys()),
    "合計_SBなし": [round(float(v_final[c]), 6) for c in v_final.keys()]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

# ===== 印用（既存の安全弁を維持）
FINISH_WEIGHT   = globals().get("FINISH_WEIGHT", 6.0)
FINISH_WEIGHT_G = globals().get("FINISH_WEIGHT_G", 3.0)
POS_BONUS  = globals().get("POS_BONUS", {0: 0.0, 1: -0.6, 2: -0.9, 3: -1.2, 4: -1.4})
POS_WEIGHT = globals().get("POS_WEIGHT", 1.0)
SMALL_Z_RATING = globals().get("SMALL_Z_RATING", 0.01)
FINISH_CLIP = globals().get("FINISH_CLIP", 4.0)
TIE_EPSILON  = globals().get("TIE_EPSILON", 0.8)

p2_list = [float(p2_eff.get(n, 0.0)) for n in active_cars]
if len(p2_list) >= 1:
    mu_p2  = float(np.mean(p2_list))
    sd_p2  = float(np.std(p2_list) + 1e-12)
else:
    mu_p2, sd_p2 = 0.0, 1.0
p2z_map = {n: (float(p2_eff.get(n, 0.0)) - mu_p2) / sd_p2 for n in active_cars}
p1_eff_safe = {n: float(p1_eff.get(n, 0.0)) if 'p1_eff' in globals() and p1_eff is not None else 0.0 for n in active_cars}
p2only_map = {n: max(0.0, float(p2_eff.get(n, 0.0)) - float(p1_eff_safe.get(n, 0.0))) for n in active_cars}
zt = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
zt_map = {n: float(zt[i]) for i, n in enumerate(active_cars)} if active_cars else {}

def _pos_idx(no:int) -> int:
    g = car_to_group.get(no, None)
    if g is None or g not in line_def:
        return 0
    grp = line_def[g]
    try:
        return max(0, int(grp.index(no)))
    except Exception:
        return 0

bonus_init,_ = compute_lineSB_bonus(line_def, S, B, line_factor=line_factor_eff, exclude=None, cap=cap_SB_eff, enable=line_sb_enable)

def anchor_score(no:int) -> float:
    base = float(v_final.get(no, -1e9))
    role = role_in_line(no, line_def)
    sb = float(bonus_init.get(car_to_group.get(no, None), 0.0) * (pos_coeff(role, 1.0) if line_sb_enable else 0.0))
    pos_term = POS_WEIGHT * POS_BONUS.get(_pos_idx(no), 0.0)
    if _is_girls:
        raw_finish = FINISH_WEIGHT_G * float(p2z_map.get(no, 0.0))
    else:
        raw_finish = FINISH_WEIGHT * float(p2z_map.get(no, 0.0))
    finish_term = max(-FINISH_CLIP, min(FINISH_CLIP, raw_finish))
    return base + sb + pos_term + finish_term + SMALL_Z_RATING * zt_map.get(no, 0.0)

# ===== ◎候補抽出（既存ロジック維持）
cand_sorted = sorted(active_cars, key=lambda n: anchor_score(n), reverse=True)
C = cand_sorted[:min(3, len(cand_sorted))]
ratings_sorted2 = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
ratings_rank2 = {n: i+1 for i,n in enumerate(ratings_sorted2)}
ALLOWED_MAX_RANK = globals().get("ALLOWED_MAX_RANK", 5)

guarantee_top_rating = True
if guarantee_top_rating and (race_class == "ガールズ") and len(ratings_sorted2) >= 1:
    top_rating_car = ratings_sorted2[0]
    if top_rating_car not in C:
        C = [top_rating_car] + [c for c in C if c != top_rating_car]
        C = C[:min(3, len(cand_sorted))]

# SB上位制限
ANCHOR_CAND_SB_TOPK   = globals().get("ANCHOR_CAND_SB_TOPK", 5)
ANCHOR_REQUIRE_TOP_SB = globals().get("ANCHOR_REQUIRE_TOP_SB", 3)
rank_pure = {int(df_sorted_pure.loc[i, "車番"]): i+1 for i in range(len(df_sorted_pure))}
cand_pool = [c for c in C if rank_pure.get(c, 999) <= ANCHOR_CAND_SB_TOPK]
if not cand_pool:
    cand_pool = [int(df_sorted_pure.loc[i, "車番"]) for i in range(min(ANCHOR_CAND_SB_TOPK, len(df_sorted_pure)))]
anchor_no_pre = max(cand_pool, key=lambda x: anchor_score(x)) if cand_pool else int(df_sorted_pure.loc[0, "車番"])
anchor_no = anchor_no_pre
top2 = sorted(cand_pool, key=lambda x: anchor_score(x), reverse=True)[:2]
if len(top2) >= 2:
    s1 = anchor_score(top2[0]); s2 = anchor_score(top2[1])
    if (s1 - s2) < TIE_EPSILON:
        better_by_rating = min(top2, key=lambda x: ratings_rank2.get(x, 999))
        anchor_no = better_by_rating
if rank_pure.get(anchor_no, 999) > ANCHOR_REQUIRE_TOP_SB:
    pool = [c for c in cand_pool if rank_pure.get(c, 999) <= ANCHOR_REQUIRE_TOP_SB]
    if pool:
        anchor_no = max(pool, key=lambda x: anchor_score(x))
    else:
        anchor_no = int(df_sorted_pure.loc[0, "車番"])
    st.caption(f"※ ◎は『SBなし 上位{ANCHOR_REQUIRE_TOP_SB}位以内』縛りで {anchor_no_pre}→{anchor_no} に調整。")

role_map = {no: role_in_line(no, line_def) for no in active_cars}
cand_scores = [anchor_score(no) for no in C] if len(C) >= 2 else [0, 0]
cand_scores_sorted = sorted(cand_scores, reverse=True)
conf_gap = cand_scores_sorted[0] - cand_scores_sorted[1] if len(cand_scores_sorted) >= 2 else 0.0
spread = float(np.std(list(v_final.values()))) if len(v_final) >= 2 else 0.0
norm = conf_gap / (spread if spread > 1e-6 else 1.0)
confidence = "優位" if norm >= 1.0 else ("互角" if norm >= 0.5 else "混戦")

score_adj_map = apply_anchor_line_bonus(v_final, car_to_group, role_map, anchor_no, confidence)

df_sorted_wo = pd.DataFrame({
    "車番": active_cars,
    "合計_SBなし": [round(float(score_adj_map.get(int(c), v_final.get(int(c), float("-inf")))), 6) for c in active_cars]
}).sort_values("合計_SBなし", ascending=False).reset_index(drop=True)

velobi_wo = list(zip(df_sorted_wo["車番"].astype(int).tolist(),
                     df_sorted_wo["合計_SBなし"].round(3).tolist()))

# ===== 既存の印集約（維持）
def _shrink_p3in(no: int, k: int = 12) -> float:
    n = x1.get(no,0)+x2.get(no,0)+x3.get(no,0)+x_out.get(no,0)
    s = x1.get(no,0)+x2.get(no,0)+x3.get(no,0)
    if n_cars <= 6: p0 = 0.40
    elif n_cars == 7: p0 = 0.35
    else: p0 = 0.30
    return (s + k*p0) / (n + k) if (n+k)>0 else p0

def _pos_penalty(no: int) -> float:
    role = role_in_line(no, line_def)
    return 0.08 if role == 'thirdplus' else (0.05 if role == 'single' else 0.0)
def _score_neg(no: int) -> float:
    zs = zscore_list([ratings_val[n] for n in active_cars]) if active_cars else []
    zmap = {n: float(zs[i]) for i,n in enumerate(active_cars)} if active_cars else {}
    z = zmap.get(no, 0.0)
    if z <= -1.0: return 0.10
    if z <= -0.5: return 0.05
    return 0.0
def _sb_ineff(no: int) -> float:
    sb = float(S.get(no,0)) + float(B.get(no,0))
    return 0.05 if (sb >= 5 and _shrink_p3in(no) < 0.25) else 0.0
def select_beta(cars: list[int]) -> int | None:
    if not cars: return None
    ko = {}
    for no in cars:
        p3 = _shrink_p3in(no)
        ko[no] = (0.70 * max(0.25 - p3, 0.0) + 0.15 * _pos_penalty(no)
                  + 0.10 * _score_neg(no) + 0.05 * _sb_ineff(no))
    return max(ko, key=ko.get) if len(ko)>0 else None
def _alpha_forbidden(no: int) -> bool:
    role = role_in_line(no, line_def)
    if role == 'second': return True
    n = x1.get(no,0)+x2.get(no,0)+x3.get(no,0)+x_out.get(no,0)
    if n >= 10 and x3.get(no,0) >= 3: return True
    order = sorted(active_cars, key=lambda n: ratings_val[n], reverse=True)
    top2 = set(order[:min(2, len(order))])
    if no in top2: return True
    return False
def enforce_alpha_eligibility(result_marks: dict[str,int]) -> dict[str,int]:
    marks = dict(result_marks)
    used = set(marks.values())
    beta_id = marks.get("β", None)
    alpha_id = marks.get("α", None)
    if alpha_id is not None and _alpha_forbidden(alpha_id):
        if "×" not in marks:
            marks["×"] = alpha_id
        del marks["α"]
        used = set(marks.values())
    if "α" not in marks:
        pool_sorted = [int(df_sorted_wo.loc[i,"車番"]) for i in range(len(df_sorted_wo))]
        for no in reversed(pool_sorted):
            if no in used: continue
            if beta_id is not None and no == beta_id: continue
            if not _alpha_forbidden(no):
                marks["α"] = no
                used.add(no)
                break
    return marks

beta_id = select_beta([c for c in active_cars if c != anchor_no])
rank_wo = {int(df_sorted_wo.loc[i, "車番"]): i+1 for i in range(len(df_sorted_wo))}
result_marks, reasons = {}, {}
result_marks["◎"] = anchor_no
reasons[anchor_no] = "本命(C上位3→得点ゲート→ラインSB重視＋KO並び＋着順z＋位置加点)"
if beta_id is not None:
    result_marks["β"] = beta_id
    reasons[beta_id] = "β（来ない枠：低3着率×位置×得点×SB空回り）"

beta_gid = car_to_group.get(beta_id, None) if beta_id is not None else None
old_anchor = None

score_map = {int(df_sorted_wo.loc[i, "車番"]): float(df_sorted_wo.loc[i, "合計_SBなし"]) for i in range(len(df_sorted_wo))}
pool_all = [int(df_sorted_wo.loc[i, "車番"]) for i in range(len(df_sorted_wo))]
overall_rest = [c for c in pool_all if c not in {anchor_no, beta_id}]
a_gid = car_to_group.get(anchor_no, None)
mates_sorted = []
if a_gid is not None and a_gid in line_def:
    mates_sorted = sorted([c for c in line_def[a_gid] if c not in {anchor_no, beta_id}],
                          key=lambda x: (-score_map.get(x, -1e9), x))
preferred_second = old_anchor if (old_anchor is not None and old_anchor != beta_id) else None
if overall_rest:
    if preferred_second is not None and preferred_second in overall_rest:
        pick2 = preferred_second
        src = "旧◎優先"
    else:
        non_anchor_rest = [c for c in overall_rest if car_to_group.get(c, None) != a_gid]
        if non_anchor_rest:
            pick2 = non_anchor_rest[0]; src = "◎ライン外優先"
        else:
            pick2 = overall_rest[0]; src = "全体トップ"
    result_marks["〇"] = pick2
    reasons[pick2] = f"対抗（{src}）"

used = set(result_marks.values())
mate_candidates = [c for c in mates_sorted if c not in used]
if mate_candidates:
    pick = mate_candidates[0]
    result_marks["▲"] = pick
    reasons[pick] = "単穴（◎ライン優先：同ライン最上位を採用）"
else:
    rest_global = [c for c in overall_rest if c not in used]
    if rest_global:
        pick = rest_global[0]
        result_marks["▲"] = pick
        reasons[pick] = "単穴（格上げ後SBなしスコア順）"

used = set(result_marks.values())
tail_priority = [c for c in mates_sorted if c not in used]
tail_priority += [c for c in overall_rest if c not in used and c not in tail_priority]
for mk in ["△","×","α"]:
    if mk in result_marks: continue
    if not tail_priority: break
    no = tail_priority.pop(0)
    result_marks[mk] = no
    reasons[no] = f"{mk}（◎ライン優先→残りスコア順）"
result_marks = enforce_alpha_eligibility(result_marks)
if "α" not in result_marks:
    used_now = set(result_marks.values())
    pool = [int(df_sorted_wo.loc[i, "車番"]) for i in range(len(df_sorted_wo))]
    pool = [c for c in pool if c not in used_now and c != beta_id]
    if pool:
        alpha_pick = pool[-1]
        result_marks["α"] = alpha_pick
        reasons[alpha_pick] = "α（フォールバック：禁止条件全滅→最弱を採用）"

# ==============================
# ★ 偏差値フィルタ閾値 & 目標回収率
# ==============================
import math
import numpy as np
import pandas as pd
import streamlit as st
from itertools import combinations

# しきい値（S＝偏差値の合算）
S_TRIO_MIN = 170.0
S_QN_MIN   = 120.0
S_WIDE_MIN = 110.0

# 目標回収率（ROI）
TARGET_ROI = {"trio":1.20, "qn":1.10, "wide":1.05}

# 表示用小数
HEN_DEC_PLACES = 1
EPS = 1e-12  # 数値安定

# === NaN/欠損に強いスコア整形 & Tスコア（NaN安全） =========================
def coerce_score_map(d, n_cars: int) -> dict[int, float]:
    """velobi_wo を {車番:int -> スコア:float} に正規化し、1..n_cars を埋める"""
    out: dict[int, float] = {}
    if d is None:
        pass
    elif hasattr(d, "items"):  # dict / Mapping
        for k, v in d.items():
            try: i = int(k)
            except Exception: continue
            try: x = float(v)
            except Exception: x = np.nan
            out[i] = x
    elif "pandas.core.series" in str(type(d)).lower():  # Series
        try:
            for k, v in d.to_dict().items():
                try: i = int(k)
                except Exception: continue
                try: x = float(v)
                except Exception: x = np.nan
                out[i] = x
        except Exception: pass
    elif "pandas.core.frame" in str(type(d)).lower():   # DataFrame
        try:
            df_ = d.copy()
            car_col = "車番" if "車番" in df_.columns else None
            if car_col is None:
                for c in df_.columns:
                    if np.issubdtype(df_[c].dtype, np.integer):
                        car_col = c; break
            score_col = None
            for cand in ["SBなし", "合計_SBなし", "スコア", "score", "SB_wo", "SB"]:
                if cand in df_.columns: score_col = cand; break
            if score_col is None:
                for c in df_.columns:
                    if c == car_col: continue
                    if np.issubdtype(df_[c].dtype, np.floating) or np.issubdtype(df_[c].dtype, np.number):
                        score_col = c; break
            if car_col is not None and score_col is not None:
                for _, r in df_.iterrows():
                    try: i = int(r[car_col])
                    except Exception: continue
                    try: x = float(r[score_col])
                    except Exception: x = np.nan
                    out[i] = x
        except Exception: pass
    elif isinstance(d, (list, tuple, np.ndarray)):      # list/tuple/ndarray
        arr = list(d)
        if len(arr) == n_cars and all(not isinstance(x, (list, tuple, dict)) for x in arr):
            for idx, v in enumerate(arr, start=1):
                try: out[idx] = float(v)
                except Exception: out[idx] = np.nan
        else:
            for it in arr:
                try:
                    if isinstance(it, (list, tuple)) and len(it) >= 2:
                        i = int(it[0]); x = float(it[1])
                        out[i] = x
                except Exception: continue
    # 埋め
    for i in range(1, int(n_cars) + 1):
        out.setdefault(i, np.nan)
    return out

def fill_nans_with_median(arr: np.ndarray) -> np.ndarray:
    a = arr.astype(float, copy=True)
    finite = np.isfinite(a)
    if not finite.any():
        a[:] = 0.0
        return a
    med = float(np.nanmedian(a))
    a[~finite] = med
    return a

def t_score_nan_safe(values: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    v = values.astype(float, copy=True)
    mu = float(np.nanmean(v))
    sd = float(np.nanstd(v))
    if not np.isfinite(mu): return np.full_like(v, 50.0)
    if not np.isfinite(sd) or sd < eps: return np.full_like(v, 50.0)
    return 50.0 + 10.0 * (v - mu) / sd

# === 全体基準Tのフォールバック =========================
def _coerce_tscore_map_any(d, n_cars: int) -> dict[int, float] | None:
    if d is None: return None
    out = {}
    if hasattr(d, "items"):                 # dict / Mapping
        it = d.items()
    elif "pandas.core.series" in str(type(d)).lower():  # Series
        it = d.to_dict().items()
    elif "pandas.core.frame" in str(type(d)).lower():   # DataFrame
        df_ = d.copy()
        car_col = "車番" if "車番" in df_.columns else None
        if car_col is None:
            for c in df_.columns:
                if np.issubdtype(df_[c].dtype, np.integer):
                    car_col = c; break
        val_col = None
        for cand in ["偏差値","tscore","T","全体偏差値","score"]:
            if cand in df_.columns: val_col = cand; break
        if val_col is None:
            for c in df_.columns:
                if c == car_col: continue
                if np.issubdtype(df_[c].dtype, np.floating) or np.issubdtype(df_[c].dtype, np.number):
                    val_col = c; break
        if car_col is None or val_col is None: return None
        for _, r in df_.iterrows():
            try:   i = int(r[car_col]); x = float(r[val_col])
            except Exception: continue
            out[i] = x
        for i in range(1, n_cars+1): out.setdefault(i, np.nan)
        return out
    else:
        return None
    for k, v in it:
        try:   i = int(k)
        except Exception: continue
        try:   x = float(v)
        except Exception: x = np.nan
        out[i] = x
    for i in range(1, n_cars+1): out.setdefault(i, np.nan)
    return out

def make_global_t_fixed(n_cars: int, xs_sb_wo_raw: np.ndarray, decimals: int = 1) -> dict[int, float | str]:
    """
    優先順位:
      (1) global_tscore_dict を使用（正規化＆丸め）
      (2) GLOBAL_SB_WO_MEAN / GLOBAL_SB_WO_SD があれば SBなし → 全体Tへ変換
      (3) どちらも無ければ '—'
    """
    gmap = globals().get("global_tscore_dict", None)
    coerced = _coerce_tscore_map_any(gmap, n_cars) if gmap is not None else None
    if coerced:
        arr = np.array([coerced.get(i, np.nan) for i in range(1, n_cars+1)], dtype=float)
        if np.isfinite(arr).any():
            arr = fill_nans_with_median(arr)
            return {i: round(float(arr[i-1]), decimals) for i in range(1, n_cars+1)}

    mean_ = globals().get("GLOBAL_SB_WO_MEAN", None)
    sd_   = globals().get("GLOBAL_SB_WO_SD", None)
    if isinstance(mean_, (int,float)) and isinstance(sd_, (int,float)) and sd_ and sd_ > 0:
        t = 50.0 + 10.0 * ((xs_sb_wo_raw - float(mean_)) / float(sd_))
        t = fill_nans_with_median(np.asarray(t, dtype=float))
        return {i: round(float(t[i-1]), decimals) for i in range(1, n_cars+1)}

    return {i: "—" for i in range(1, n_cars+1)}

# ==============================
# ★ 偏差値セクション（従来：△=50固定・◎≤80・風/ライン込み）
# ==============================
def zscore_list(vals):
    arr = np.array(vals, dtype=float)
    mu = float(arr.mean()) if arr.size else 0.0
    sd = float(arr.std(ddof=0)) if arr.size else 1.0
    if sd < EPS: sd = 1.0
    return [(x - mu) / sd for x in arr]

def t_score(values: np.ndarray, use_sample: bool = False, eps: float = 1e-9):
    ddof = 1 if use_sample else 0
    mu = float(np.mean(values))
    sd = float(np.std(values, ddof=ddof))
    sd = max(sd, eps)
    return 50.0 + 10.0 * (values - mu) / sd

# 1) 素のスコア（SB・環境込み脚質・入着）
sb_raw, prof_env_raw, in3_raw = {}, {}, {}
for no in active_cars:
    role = role_in_line(no, line_def)
    wpos = {'head':1.0,'second':0.7,'thirdplus':0.5,'single':0.9}.get(role,0.9)
    sb_raw[no] = wpos * (float(S.get(no,0)) + float(B.get(no,0)))
    rec = df[df["車番"] == no].iloc[0]
    env_sum = float(rec["風補正"]) + float(rec["バンク補正"]) + float(rec["周長補正"]) + float(rec["周回補正"])
    g = car_to_group.get(no, None)
    line_bonus_g = float(bonus_init.get(g, 0.0)) if line_sb_enable else 0.0
    line_bonus_i = line_bonus_g * pos_coeff(role, 1.0)
    prof_env_raw[no] = float(prof_base[no]) + env_sum + line_bonus_i
    in3_raw[no] = float(_shrink_p3in(no))

# 2) z合成（SB 0.2 / 脚質(環境込み) 0.3 / 入着 0.5）→ 偏差値
HEN_W_SB, HEN_W_PROF, HEN_W_IN = 0.2, 0.3, 0.5
z_sb   = zscore_list([sb_raw[n]       for n in active_cars]) if active_cars else []
z_prof = zscore_list([prof_env_raw[n] for n in active_cars]) if active_cars else []
z_in   = zscore_list([in3_raw[n]      for n in active_cars]) if active_cars else []

hen_raw = {}
for i, no in enumerate(active_cars):
    z = HEN_W_SB*z_sb[i] + HEN_W_PROF*z_prof[i] + HEN_W_IN*z_in[i]
    hen_raw[no] = 50.0 + 10.0*float(z)

# 3) △を50に平行移動
base_car = result_marks.get("△", None)
if base_car is None or base_car not in hen_raw:
    items = sorted(hen_raw.items(), key=lambda x: x[1])
    base_car = items[len(items)//2][0]
shift = 50.0 - hen_raw[base_car]
hen_shifted = {no: hen_raw[no] + shift for no in active_cars}

# 4) ◎が80を超えないようシフト
hmax = max(hen_shifted.values()) if hen_shifted else 80.0
if hmax > 80.0:
    down = hmax - 80.0
    hen_shifted = {no: val - down for no, val in hen_shifted.items()}

# 5) 丸め（画面表示用）
hen_map = {no: round(hen_shifted[no], HEN_DEC_PLACES) for no in active_cars}

# 表示
hen_df = pd.DataFrame({
    "車": active_cars,
    "偏差値": [hen_map[n] for n in active_cars],
    "SB原点": [round(sb_raw[n],3) for n in active_cars],
    "脚質(環境+ライン)": [round(prof_env_raw[n],3) for n in active_cars],
    "入着(縮約3内)": [round(in3_raw[n],3) for n in active_cars],
}).sort_values(["偏差値","車"], ascending=[False, True]).reset_index(drop=True)
st.markdown("### 偏差値（△=50・◎≤80・風/ライン込み）")
st.dataframe(hen_df, use_container_width=True)

# ==============================
# ★ レース内基準（SBなし）→ PLモデル用の強さ（購入計算で使用）
# ==============================
# SBなしスコア（風・ライン補正後）
score_map_sb_wo = coerce_score_map(velobi_wo, n_cars)
xs_raw = np.array([score_map_sb_wo[i] for i in range(1, n_cars + 1)], dtype=float)

# レース内T（購入計算用）
xs = fill_nans_with_median(xs_raw.copy())
xs_race_t = t_score_nan_safe(xs)
race_t = {i: round(float(xs_race_t[i-1]), 1) for i in range(1, n_cars + 1)}
race_z = (xs_race_t - 50.0) / 10.0

# ★ 全体基準T：三段フォールバックで生成（← これで '—' を回避）
global_t_fixed = make_global_t_fixed(n_cars, xs_raw, decimals=HEN_DEC_PLACES)

# PL用の重み
tau = 1.0
w = np.exp(race_z * tau)
S_w = float(np.sum(w))
w_idx = {i: float(w[i-1]) for i in range(1, n_cars + 1)}

def prob_top2_pair_pl(i: int, j: int) -> float:
    wi, wj = w_idx[i], w_idx[j]
    d_i = max(S_w - wi, EPS); d_j = max(S_w - wj, EPS)
    return (wi / S_w) * (wj / d_i) + (wj / S_w) * (wi / d_j)

def prob_top3_triple_pl(i: int, j: int, k: int) -> float:
    a, b, c = w_idx[i], w_idx[j], w_idx[k]
    total = 0.0
    for x, y, z in [(a,b,c),(a,c,b),(b,a,c),(b,c,a),(c,a,b),(c,b,a)]:
        d1 = max(S_w - x, EPS)
        d2 = max(S_w - x - y, EPS)
        total += (x / S_w) * (y / d1) * (z / d2)
    return total

def prob_wide_pair_pl(i: int, j: int) -> float:
    total = 0.0
    for k in range(1, n_cars + 1):
        if k == i or k == j: continue
        total += prob_top3_triple_pl(i, j, k)
    return total

# ==============================
# ★ 買い目生成（全候補）＋ 最低限オッズ（PL確率×TARGET_ROI）
# ==============================
# ←←← ここが変更点：Sは「このレース内偏差値（race_t）」起点に統一
S_BASE_MAP = {i: float(race_t[i]) for i in range(1, n_cars+1)}

def _pair_score(a,b):   return float(S_BASE_MAP.get(a,0.0) + S_BASE_MAP.get(b,0.0))
def _trio_score(a,b,c): return float(S_BASE_MAP.get(a,0.0) + S_BASE_MAP.get(b,0.0) + S_BASE_MAP.get(c,0.0))

pairs, trios = [], []
for i in range(len(active_cars)):
    for j in range(i+1, len(active_cars)):
        a, b = active_cars[i], active_cars[j]
        pairs.append((a,b,_pair_score(a,b)))
for i in range(len(active_cars)):
    for j in range(i+1, len(active_cars)):
        for k in range(j+1, len(active_cars)):
            a, b, c = active_cars[i], active_cars[j], active_cars[k]
            trios.append((a,b,c,_trio_score(a,b,c)))

# Sしきいで候補抽出（Sはレース内基準）
pairs_qn  = sorted([(a,b,s)   for (a,b,s)   in pairs  if s >= S_QN_MIN],  key=lambda x:(-x[2], x[0], x[1]))
pairs_w   = sorted([(a,b,s)   for (a,b,s)   in pairs  if s >= S_WIDE_MIN], key=lambda x:(-x[2], x[0], x[1]))
trios_all = sorted([(a,b,c,s) for (a,b,c,s) in trios  if s >= S_TRIO_MIN], key=lambda x:(-x[3], x[0], x[1], x[2]))

# ---- 各候補のPL確率から「最低限オッズ（単一値）」を計算（TARGET_ROI / p の最小）
def _min_required_from_pairs(rows, p_func, roi: float) -> float|None:
    reqs = []
    for a,b,*_ in rows:
        p = p_func(a,b)
        if p > EPS: reqs.append(roi / p)
    if not reqs: return None
    m = min(reqs)
    return math.floor(m*2 + 0.5) / 2.0  # 0.5刻み丸め

def _min_required_from_trios(rows, p_func, roi: float) -> float|None:
    reqs = []
    for a,b,c,*_ in rows:
        p = p_func(a,b,c)
        if p > EPS: reqs.append(roi / p)
    if not reqs: return None
    m = min(reqs)
    return math.floor(m*2 + 0.5) / 2.0

min_odds_qn   = _min_required_from_pairs(pairs_qn,  prob_top2_pair_pl,   TARGET_ROI["qn"])
min_odds_wide = _min_required_from_pairs(pairs_w,   prob_wide_pair_pl,   TARGET_ROI["wide"])
min_odds_trio = _min_required_from_trios(trios_all, prob_top3_triple_pl, TARGET_ROI["trio"])

# 表示テーブル（S=レース内基準）
def _df_trio(rows):
    return pd.DataFrame([{"買い目":"-".join(map(str,sorted([a,b,c]))), "偏差値S":round(s,1)} for (a,b,c,s) in rows])
def _df_pair(rows):
    return pd.DataFrame([{"買い目":f"{a}-{b}", "偏差値S":round(s,1)} for (a,b,s) in rows])

if trios_all:
    hdr = "#### 三連複（偏差値S＝レース内基準）"
    if min_odds_trio is not None:
        hdr += f"　/　**最低限オッズ {min_odds_trio:.1f}倍以上**（目標回収率{int(TARGET_ROI['trio']*100)}%）"
    st.markdown(hdr); st.dataframe(_df_trio(trios_all), use_container_width=True)
else:
    st.markdown("#### 三連複（該当なし）")

if pairs_qn:
    hdr = "#### 二車複（偏差値S＝レース内基準）"
    if min_odds_qn is not None:
        hdr += f"　/　**最低限オッズ {min_odds_qn:.1f}倍以上**（目標回収率{int(TARGET_ROI['qn']*100)}%）"
    st.markdown(hdr); st.dataframe(_df_pair(pairs_qn), use_container_width=True)
else:
    st.markdown("#### 二車複（該当なし）")

if pairs_w:
    hdr = "#### ワイド（偏差値S＝レース内基準）"
    if min_odds_wide is not None:
        hdr += f"　/　**最低限オッズ {min_odds_wide:.1f}倍以上**（目標回収率{int(TARGET_ROI['wide']*100)}%）"
    st.markdown(hdr); st.dataframe(_df_pair(pairs_w), use_container_width=True)
else:
    st.markdown("#### ワイド（該当なし）")

# ==============================
# ★ note用 出力（縦並び偏差値・全候補・最低限オッズ帯）
# ==============================
st.markdown("### 📋 note用（コピーエリア）")

line_text = "　".join([x for x in line_inputs if str(x).strip()])
marks_line = " ".join(f"{m}{result_marks[m]}" for m in ["◎","〇","▲","△","×","α","β"] if m in result_marks)
score_map_for_note = {int(r["車番"]): float(r["合計_SBなし"]) for _, r in df_sorted_wo.iterrows()}
score_order_text = format_rank_all(score_map_for_note, P_floor_val=None)

def _fmt_hen_lines(ts_map: dict) -> str:
    out = []
    for n in range(1, n_cars+1):
        v = ts_map.get(n, "—")
        if isinstance(v, (int, float)):
            out.append(f"{n}: {float(v):.{HEN_DEC_PLACES}f}")
        else:
            out.append(f"{n}: —")
    return "\n".join(out)

hen_lines_global = _fmt_hen_lines(global_t_fixed)  # 全体基準（フォールバック適用済み）
hen_lines_race   = _fmt_hen_lines(race_t)          # レース内基準（SBなし→T）

def _lines_from_df_note(df: pd.DataFrame, title: str, min_odds: float|None) -> str:
    if df is None or df.empty:
        tail = "対象外"
    else:
        rows = [f"{row['買い目']}（S={row['偏差値S']:.1f}）" for _, row in df.iterrows()]
        tail = "\n".join(rows)
    head = title
    if min_odds is not None:
        base = (S_TRIO_MIN if "三連複" in title else S_QN_MIN if "二車複" in title else S_WIDE_MIN)
        head += f"（基準{base:.0f}以上／最低限オッズ {min_odds:.1f}倍以上）"
    return f"{head}\n{tail}"

note_text = (
    f"{'推奨 3連複' if len(trios_all)>=1 else ('推奨 2車複・ワイド' if (len(pairs_qn)+len(pairs_w))>=1 else '推奨 ケン')}\n"
    f"競輪場　{track}{race_no}R\n"
    f"展開評価：{confidence}\n\n"
    f"{race_time}　{race_class}\n"
    f"ライン　{line_text}\n"
    f"スコア順（SBなし）　{score_order_text}\n"
    f"{marks_line}\n\n"
    "偏差値（風・ライン込み）\n"
    "— 全体基準 —\n"
    f"{hen_lines_global}\n\n"
    "— レース内基準（平均50・SD10） —\n"
    f"{hen_lines_race}\n\n"
    + _lines_from_df_note(_df_trio(trios_all), "三連複", min_odds_trio) + "\n\n"
    + _lines_from_df_note(_df_pair(pairs_qn),   "二車複", min_odds_qn)   + "\n\n"
    + _lines_from_df_note(_df_pair(pairs_w),    "ワイド", min_odds_wide)
)
st.text_area("ここを選択してコピー", note_text, height=520)
