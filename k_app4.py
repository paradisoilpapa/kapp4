
# ✅ Streamlit版：完全再構成（風補正・ライン補正・印・雨・得点・着順・バンク・欠番・UI入力完全対応）

import streamlit as st
import pandas as pd

st.set_page_config(page_title="ライン競輪スコア計算（完全版）", layout="wide")

st.title("⭐ ライン競輪スコア計算（完全構成UI）⭐")

# --- 定義部分 ---
wind_coefficients = {
    "左上": +0.7, "上": +1.0, "右上": +0.7,
    "左": 0.0, "右": 0.0,
    "左下": -0.7, "下": -1.0, "右下": -0.7
}
position_multipliers = {1: 1.0, 2: 0.3, 3: 0.1, 0: 1.2}
base_score = {'逃': 8, '両': 6, '追': 5}
symbol_bonus = {'◎': 2.0, '〇': 1.5, '▲': 1.0, '△': 0.5, '×': 0.2, '無': 0.0}

# --- 状態保持 ---
if "selected_wind" not in st.session_state:
    st.session_state.selected_wind = None

# --- UI入力欄 ---
st.header("【風向き選択】")
cols_top = st.columns(3)
cols_mid = st.columns(3)
cols_bot = st.columns(3)

with cols_top[0]:
    if st.button("左上"): st.session_state.selected_wind = "左上"
with cols_top[1]:
    if st.button("上"): st.session_state.selected_wind = "上"
with cols_top[2]:
    if st.button("右上"): st.session_state.selected_wind = "右上"
with cols_mid[0]:
    if st.button("左"): st.session_state.selected_wind = "左"
with cols_mid[1]:
    st.markdown("<div style='text-align:center;'>□ ホーム</div>", unsafe_allow_html=True)
with cols_mid[2]:
    if st.button("右"): st.session_state.selected_wind = "右"
with cols_bot[0]:
    if st.button("左下"): st.session_state.selected_wind = "左下"
with cols_bot[1]:
    if st.button("下"): st.session_state.selected_wind = "下"
with cols_bot[2]:
    if st.button("右下"): st.session_state.selected_wind = "右下"

st.markdown("---")

wind_speed = st.number_input("風速 (m/s)", 0.0, 10.0, 3.0, step=0.1)
straight_length = st.number_input("みなし直線(m)", 30, 80, 52)
bank_angle = st.number_input("バンク角(°)", 20.0, 45.0, 30.0, step=0.1)
rain = st.checkbox("雨（滑走・慎重傾向あり）")

st.markdown("---")
st.header("【選手別データ入力（欠番対応）】")

kakushitsu_options = ['逃', '両', '追']
symbol_options = ['◎', '〇', '▲', '△', '×', '無']

input_data = []

for i in range(7):
    with st.expander(f"{i+1}番選手の情報"):
        valid = st.checkbox(f"{i+1}番選手は出走", value=True, key=f"valid_{i}")
        if valid:
            kaku = st.selectbox("脚質", kakushitsu_options, key=f"kaku_{i}")
            chaku = st.number_input("前走着順", 1, 7, 4, key=f"chaku_{i}")
            rating = st.number_input("競争得点", 40.0, 70.0, 55.0, step=0.1, key=f"rate_{i}")
            symbol = st.selectbox("政春印", symbol_options, key=f"sym_{i}")
            line_pos = st.selectbox("ライン位置 (0=単騎 1=先頭 2=2番手 3=3番手)", [0,1,2,3], key=f"line_{i}")

            input_data.append({"車番": i+1, "脚質": kaku, "着順": chaku, "得点": rating, "印": symbol, "ライン位置": line_pos})

if st.button("スコア計算実行"):
    if not input_data:
        st.warning("出走選手の情報を入力してください。")
    else:
        df = pd.DataFrame(input_data)
        avg_rating = df["得点"].mean()

        results = []
        for i, row in df.iterrows():
            num = row["車番"]
            kaku = row["脚質"]
            chaku = row["着順"]
            rating = row["得点"]
            symbol = row["印"]
            line_pos = row["ライン位置"]

            base = base_score[kaku]
            wind_corr = 0.0
if st.session_state.selected_wind:
    base_wind = wind_coefficients.get(st.session_state.selected_wind, 0.0) * wind_speed * position_multipliers[line_pos]
    wind_corr = base_wind * { '逃': 1.2, '両': 1.0, '追': 0.8 }.get(kaku, 1.0)
            tai_corr = max(0, round(3.0 - 0.5 * i, 1)) + (1.5 if kaku == '追' and 2 <= i <= 4 else 0)
            chaku_corr = 4 - chaku
            rating_corr = round((avg_rating - rating) * 0.2, 1)
            rain_corr = { '逃': 2.5, '両': 0.5, '追': -2.5 }.get(kaku, 0) if rain else 0
            symb_corr = symbol_bonus[symbol]
            line_corr = {0: -1.0, 1: 2.0, 2: 1.5, 3: 1.0}.get(line_pos, 0)

            bank_corr = 0.0
            if straight_length <= 50 and bank_angle >= 32:
                bank_corr = {'逃': 1.0, '両': 0, '追': -1.0}[kaku]
            elif straight_length >= 58 and bank_angle <= 31:
                bank_corr = {'逃': -1.0, '両': 0, '追': 1.0}[kaku]

            total = round(base + wind_corr + tai_corr + chaku_corr + rating_corr + rain_corr + symb_corr + line_corr + bank_corr, 2)

            results.append((num, kaku, base, wind_corr, tai_corr, chaku_corr, rating_corr, rain_corr, symb_corr, line_corr, bank_corr, total))

        df_out = pd.DataFrame(results, columns=["車番", "脚質", "基本", "風補正", "隊列補正", "着順補正", "得点補正", "雨補正", "政春印補正", "ライン補正", "バンク補正", "合計スコア"])
        st.dataframe(df_out)
